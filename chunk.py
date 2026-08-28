#!/usr/bin/env python3
"""Stage 3 — page text to overlapping, section-aware chunks.

    ../ceynex-core/.venv/bin/python chunk.py [--target-chars 3200] [--overlap 0.15]

**This stage is the single largest relevancy fix in the pipeline.** The prototype
embedded one vector per OCR'd page. A page of a trade strategy holds several
unrelated topics — an objective, a table, half a paragraph about market access —
and averaging them into one 384-dimension point produces a vector that is close
to everything and specific to nothing. Every retrieval then returns pages that
are vaguely on-topic instead of the passage that answers the question.

Chunks are built to respect structure first and length second: split on headings,
then on paragraphs, and only fall back to a hard character cut inside a paragraph
that is longer than the target on its own. Adjacent chunks overlap so a fact
stated across a boundary is not lost by both of them.
"""

from __future__ import annotations

import argparse
import re
import sys

from pipeline_common import CHUNKS, EXTRACTED, read_jsonl, write_jsonl

#: Conservative char equivalent of the 512-token window, at 3.5 chars/token for
#: number-dense policy prose. Only used to warn — see the report at the end.
WINDOW_CHARS = 1_800

#: Sized to the embedding model's window, which is the constraint that actually
#: binds. `BAAI/bge-base-en-v1.5` truncates at **512 tokens** — measured, not
#: assumed: its tokenizer reports
#: `{'max_length': 512, 'strategy': 'longest_first', 'direction': 'right'}` and
#: encoding 2000 words returns exactly 512 ids. Anything past that is silently
#: dropped from the dense vector while still sitting in the payload, so a chunk
#: over the window looks fine everywhere except in the one place that decides
#: whether it is ever retrieved.
#:
#: Policy text tokenizes denser than plain prose — percentages, HS codes,
#: "duty-free" — so the 4-chars-per-token rule of thumb is optimistic here. 1600
#: chars is roughly 450 tokens, leaving headroom for the heading prefix and for
#: the reranker, whose cross-encoder shares the same 512-token budget across the
#: query *and* the passage together.
TARGET_CHARS = 1_600
OVERLAP_RATIO = 0.15
MIN_CHUNK_CHARS = 200

#: Heading shapes common in government documents: numbered sections, ALL-CAPS
#: banners, and short title-case lines. Deliberately conservative — a false
#: heading only splits a chunk early, while a missed one merges two topics, and
#: merging is the failure this stage exists to prevent.
HEADING = re.compile(
    r"^(?:"
    r"\s*\d+(?:\.\d+)*\.?\s+\S.{0,80}"      # 3.1 Market access
    r"|\s*[A-Z][A-Z \-&/,']{6,80}"           # TRADE POLICY OBJECTIVES
    r"|\s*(?:chapter|section|part|annex|appendix)\s+[\dIVXLC]+\b.{0,60}"
    r")$",
    re.IGNORECASE | re.MULTILINE,
)


def split_sections(text: str) -> list[tuple[str, str]]:
    """`[(heading, body)]`. The leading body before any heading gets an empty one."""
    matches = list(HEADING.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group().strip(), text[match.end() : end]))
    return sections


def split_paragraphs(body: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def pack(paragraphs: list[str], target: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs up to `target`, carrying `overlap` chars forward."""
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        # A paragraph longer than the target on its own is the only case that
        # needs a blind cut. Splitting on sentence ends keeps the cut off the
        # middle of a figure like "16.5%".
        pieces = [paragraph]
        if len(paragraph) > target:
            pieces = re.findall(rf"(?s).{{1,{target}}}(?:(?<=[.!?])\s+|$)", paragraph) or [paragraph]

        for piece in pieces:
            if current and len(current) + len(piece) + 2 > target:
                chunks.append(current.strip())
                tail = current[-overlap:] if overlap else ""
                current = f"{tail}\n\n{piece}" if tail else piece
            else:
                current = f"{current}\n\n{piece}" if current else piece

    if current.strip():
        chunks.append(current.strip())
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-chars", type=int, default=TARGET_CHARS)
    parser.add_argument("--overlap", type=float, default=OVERLAP_RATIO)
    args = parser.parse_args()

    overlap_chars = int(args.target_chars * args.overlap)
    records: list[dict] = []
    per_doc: dict[str, int] = {}

    for page in read_jsonl(EXTRACTED):
        for heading, body in split_sections(page["text"]):
            paragraphs = split_paragraphs(body)
            if not paragraphs:
                continue
            for text in pack(paragraphs, args.target_chars, overlap_chars):
                if len(text) < MIN_CHUNK_CHARS:
                    continue
                doc_id = page["doc_id"]
                index = per_doc.get(doc_id, 0)
                per_doc[doc_id] = index + 1
                records.append(
                    {
                        "doc_id": doc_id,
                        "chunk_index": index,
                        "page": page["page"],
                        # The heading rides in the text as well as the payload:
                        # "Rules of origin" as context changes what the passage
                        # embeds to, and the embedding is what retrieval sees.
                        "section": heading,
                        "text": f"{heading}\n\n{text}".strip() if heading else text,
                    }
                )

    write_jsonl(CHUNKS, records)
    print(f"wrote {len(records)} chunks to {CHUNKS}")
    for doc_id, count in sorted(per_doc.items()):
        print(f"  {doc_id}: {count}")
    if records:
        lengths = sorted(len(r["text"]) for r in records)
        print(
            f"\nchunk length: min {lengths[0]}, median {lengths[len(lengths) // 2]}, "
            f"max {lengths[-1]} chars"
        )
        # The guard that keeps the 512-token truncation from coming back. A chunk
        # over the window is not an error anywhere — it embeds, it indexes, it
        # retrieves — it just does so from a prefix of itself, which is invisible
        # unless something says so here.
        over = [n for n in lengths if n > WINDOW_CHARS]
        if over:
            print(
                f"\nWARNING: {len(over)} chunks exceed ~{WINDOW_CHARS} chars and will be "
                f"truncated to 512 tokens by the dense model — their tails are indexed in "
                f"the payload and in BM25, but not in the dense vector.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
