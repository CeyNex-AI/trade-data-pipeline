#!/usr/bin/env python3
"""Stage 2 — raw/ documents to page-level plain text.

    ../ceynex-core/.venv/bin/python extract.py

`pdfplumber` for PDFs, `BeautifulSoup` for HTML. Both are already dependencies of
ceynex-core, so this stage needs nothing installed that the project did not
already have.

**Qwen3-VL OCR is the fallback, not the default.** The prototype ran a 2-billion
parameter vision model over all 133 pages of a PDF that has a perfectly good text
layer, which is why it took as long as it did. `pdfplumber` extracts the same
pages in about a second. Pages that come back near-empty are genuinely scanned
and are the only ones worth a VLM: this stage lists them and stops, rather than
loading a model most runs do not need. Run `ocr_pages.py` on that list and re-run
with `--use-ocr` to fold the results in.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

from pipeline_common import EXTRACTED, RAW_DIR, read_manifest, write_jsonl

#: A page yielding less than this has no usable text layer. Blank pages and
#: full-page figures also land here, which is why the OCR list is advisory —
#: OCR-ing a page that really is blank costs a little time and no correctness.
MIN_PAGE_CHARS = 80

#: A *document* yielding less than this was not really extracted.
#:
#: Several of the fetched pages are JavaScript shells: the HTML served to a
#: non-browser client contains the chrome and none of the policy text, so
#: extraction "succeeds" and returns 93 characters. Indexing that produces a
#: handful of chunks that are retrievable, citable, and say nothing — the worst
#: outcome available, because the citation looks real. Measured on the first
#: full fetch: the German BMWK page gave 93 chars, the UAE MoFA page 123, and
#: the Netherlands policy document 456, against 201,830 for the UK strategy and
#: 119,642 for the Canadian briefing book.
#:
#: Reported as a corpus gap, which is what it is. Fixing it needs either a
#: headless browser or, better, the PDF these pages link to.
MIN_DOCUMENT_CHARS = 2_000

#: HTML elements that are navigation, not document. Government sites wrap the
#: actual policy text in a great deal of this, and chunking a nav menu produces
#: text that is similar to every query and informative for none.
STRIP_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")


#: pdfplumber emits `(cid:N)` for a glyph whose embedded font carries no
#: ToUnicode map. They are not text and they are not rare — the National Export
#: Strategy is full of them — and left in they are embedded, indexed, reranked
#: and eventually shown to a reader as though they were part of the document.
CID_ARTIFACT = re.compile(r"\(cid:\d+\)")

#: The ~40 commonest English function words. Their share of a passage is a
#: cheap, dictionary-free measure of whether it is prose a model can read.
#: Measured over this corpus: pages pdfplumber extracts cleanly average 20%,
#: pages whose text layer is scrambled average 3%, because scrambling merges
#: every short word into a neighbour.
STOPWORDS = frozenset(
    "the of and to in for a is are on with by as at from that this will be has have "
    "it its or an not their our we can which was were been".split()
)
WORD = re.compile(r"[A-Za-z]+")


def clean(text: str) -> str:
    """Strip font artifacts and collapse the whitespace they leave behind."""
    text = CID_ARTIFACT.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def prose_score(text: str) -> float:
    """Share of words that are common English function words. 0.0 for gibberish.

    Used only to choose between two extractions of the same page, never as an
    absolute quality gate — a table of HS codes and tariff rates scores near
    zero and is exactly the content this corpus most needs.
    """
    words = WORD.findall(text)
    if not words:
        return 0.0
    return sum(1 for w in words if w.lower() in STOPWORDS) / len(words)


def extract_pdf(path: Path) -> tuple[list[tuple[int, str]], list[int]]:
    """Returns `(pages, needs_ocr)`. Page numbers are 1-based, as a reader sees them."""
    import pdfplumber

    pages: list[tuple[int, str]] = []
    needs_ocr: list[int] = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = clean(page.extract_text() or "")
            if len(text) < MIN_PAGE_CHARS:
                needs_ocr.append(number)
                continue
            pages.append((number, text))
    return pages, needs_ocr


def extract_html(path: Path) -> list[tuple[int, str]]:
    """HTML is one 'page'. Page numbers are meaningless for it, so it gets None later."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(list(STRIP_TAGS)):
        tag.decompose()
    # `main` or `article` when the page marks it, because that is the document;
    # falling back to the whole body only when it does not.
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = clean(root.get_text("\n", strip=True))
    return [(1, text)] if len(text) >= MIN_PAGE_CHARS else []


def prefer_ocr(
    doc_id: str, pages: list[tuple[int, str]], needs_ocr: list[int], ocr_dir: Path
) -> list[tuple[int, str]]:
    """Take the better of the pdfplumber and OCR extractions, page by page.

    Two distinct problems, one fix:

    - **A page with no text layer** (scanned, or a full-page image). pdfplumber
      returns nothing and OCR is the only source.
    - **A page whose text layer is scrambled.** Parts of the National Export
      Strategy interleave two text runs *character by character* — "Limited" and
      "Trade" come out as "LTirmaditee" — because the PDF draws them as
      overlapping objects and pdfplumber reads them in document order. Cropping
      into columns does not fix it; the interleaving is below the word level.
      OCR reads the rendered image and never sees the text layer at all, so it
      is immune.

    The second is the dangerous one. It produces plausible-looking output of
    roughly the right length that no reader would accept and no length check
    catches, and it embeds and retrieves like any other text.

    Choosing per page rather than per document, and by score rather than by
    rule, because neither extractor wins everywhere: OCR is better on the
    scrambled pages and pdfplumber is better on dense tables the VLM abridges.
    """
    by_number = dict(pages)
    swapped = 0
    recovered = 0

    for number in sorted(set(by_number) | set(needs_ocr)):
        candidate = ocr_dir / f"page_{number:03d}.txt"
        if not candidate.is_file():
            continue
        ocr_text = clean(candidate.read_text(encoding="utf-8"))
        if len(ocr_text) < MIN_PAGE_CHARS:
            continue
        existing = by_number.get(number)
        if existing is None:
            by_number[number] = ocr_text
            recovered += 1
        elif prose_score(ocr_text) > prose_score(existing):
            by_number[number] = ocr_text
            swapped += 1

    if recovered or swapped:
        print(
            f"  {doc_id}: {recovered} pages recovered from OCR, "
            f"{swapped} replaced where the text layer scored worse"
        )
    return sorted(by_number.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="fold in ocr_output/page_NNN.txt for pages with no text layer",
    )
    parser.add_argument("--ocr-dir", type=Path, default=Path("ocr_output"))
    args = parser.parse_args()

    records: list[dict] = []
    ocr_wanted: dict[str, list[int]] = {}
    skipped: list[str] = []

    for doc in read_manifest():
        candidates = sorted(RAW_DIR.glob(f"{doc.doc_id}.*"))
        if not candidates:
            skipped.append(f"{doc.doc_id} (not fetched)")
            continue
        if not doc.indexable:
            skipped.append(f"{doc.doc_id} (language={doc.language}, English-only index)")
            continue

        path = candidates[0]
        if path.suffix == ".pdf":
            pages, needs_ocr = extract_pdf(path)
            if needs_ocr:
                ocr_wanted[doc.doc_id] = needs_ocr
            if args.use_ocr:
                pages = prefer_ocr(doc.doc_id, pages, needs_ocr, args.ocr_dir)
        else:
            pages, needs_ocr = extract_html(path), []

        if not pages:
            skipped.append(f"{doc.doc_id} (no extractable text)")
            continue

        total_chars = sum(len(text) for _, text in pages)
        if total_chars < MIN_DOCUMENT_CHARS:
            skipped.append(
                f"{doc.doc_id} ({total_chars} chars — a JavaScript shell, not the document; "
                f"needs the underlying PDF or a rendered fetch)"
            )
            continue

        for number, text in sorted(pages):
            records.append(
                {
                    "doc_id": doc.doc_id,
                    "page": number if path.suffix == ".pdf" else None,
                    "text": text,
                }
            )
        print(f"  {doc.doc_id}: {len(pages)} pages, {sum(len(t) for _, t in pages):,} chars")

    write_jsonl(EXTRACTED, records)
    print(f"\nwrote {len(records)} pages to {EXTRACTED}")

    if skipped:
        print(f"\nskipped {len(skipped)}:", file=sys.stderr)
        for note in skipped:
            print(f"  - {note}", file=sys.stderr)

    if ocr_wanted and not args.use_ocr:
        total = sum(len(v) for v in ocr_wanted.values())
        print(
            f"\n{total} pages across {len(ocr_wanted)} documents have no text layer and were "
            f"dropped. These are the only pages worth running the VLM over:",
            file=sys.stderr,
        )
        for doc_id, numbers in ocr_wanted.items():
            preview = ", ".join(str(n) for n in numbers[:12])
            more = f" (+{len(numbers) - 12} more)" if len(numbers) > 12 else ""
            print(f"  {doc_id}: {preview}{more}", file=sys.stderr)
        print("\n  python ocr_pages.py --images-dir pages && python extract.py --use-ocr", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
