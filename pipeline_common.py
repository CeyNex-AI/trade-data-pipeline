#!/usr/bin/env python3
"""Shared paths, manifest reading and JSONL helpers for the policy corpus pipeline.

Five stages, each a script, each reading the previous stage's file:

    fetch.py   manifest          -> raw/<doc_id>.{pdf,html}
    extract.py raw/              -> build/extracted.jsonl   (one record per page)
    chunk.py   extracted.jsonl   -> build/chunks.jsonl      (one record per chunk)
    enrich.py  chunks.jsonl      -> build/enriched.jsonl    (+ entity tags)
    index.py   enriched.jsonl    -> qdrant collection

Separate stages rather than one script because they fail for different reasons
and cost different amounts. Fetching is network-bound and rate-limited,
extraction is CPU-bound, and indexing is the only one that touches a service.
Re-running the cheap end after fixing a chunking rule should not re-download 15
documents from government websites.

**Run these with ceynex-core's interpreter**, not a bare `python`:

    ../ceynex-core/.venv/bin/python fetch.py

They import `ceynex.retrieval.schema` and `ceynex.retrieval.tagging` from the
installed package on purpose. The payload keys the indexer writes and the
filters the agent queries with must come from one definition — a second copy
here would drift, and a drifted filter key returns an empty result silently.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
BUILD_DIR = HERE / "build"

EXTRACTED = BUILD_DIR / "extracted.jsonl"
CHUNKS = BUILD_DIR / "chunks.jsonl"
ENRICHED = BUILD_DIR / "enriched.jsonl"

#: The committed manifest. Lives in ceynex-core because the KG loader reads the
#: same file to create :PolicyDocument nodes — the corpus and the graph must
#: agree on which documents exist.
MANIFEST = HERE.parent / "ceynex-core" / "ceynex" / "data" / "reference" / "policy_documents.csv"


@dataclass
class Document:
    """One row of the manifest."""

    doc_id: str
    iso3: tuple[str, ...]
    title: str
    publisher: str
    url: str
    doc_type: str
    hs_focus: tuple[str, ...]
    agreements: tuple[str, ...]
    language: str
    published: str
    retrieved_at: str
    sha256: str
    verified: str

    @property
    def indexable(self) -> bool:
        """English-only, because the dense model is.

        `BAAI/bge-base-en-v1.5` has no useful representation of German legal
        text, so indexing AWG would put chunks into the collection that can only
        ever be retrieved by accident. The row stays in the manifest and the node
        stays in the graph — the document is recorded as found and consciously
        skipped, which is a different thing from missing.
        """
        return self.language == "en"


def _split(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(";") if part.strip())


def read_manifest(path: Path = MANIFEST) -> list[Document]:
    """Read the manifest, skipping the `#` provenance header block.

    Same shape as `ceynex.kg.loaders.trade_agreements._rows` — the header is
    prose for a human and must not reach the CSV reader.
    """
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    return [
        Document(
            doc_id=row["doc_id"],
            iso3=_split(row["iso3"]),
            title=row["title"],
            publisher=row["publisher"],
            url=row["url"],
            doc_type=row["doc_type"],
            hs_focus=_split(row["hs_focus"]),
            agreements=_split(row["agreements"]),
            language=row["language"],
            published=row["published"],
            retrieved_at=row["retrieved_at"],
            sha256=row["sha256"],
            verified=row["verified"],
        )
        for row in rows
    ]


def update_manifest(updates: dict[str, dict[str, str]], path: Path = MANIFEST) -> None:
    """Write `sha256` / `retrieved_at` back in place, preserving the header.

    In place rather than into a lock file: one source of truth for what was
    fetched and when. A committed manifest whose hashes match what is in Qdrant
    is what makes an answer citing a document reproducible six months later.
    """
    text = path.read_text(encoding="utf-8")
    header = [line for line in text.splitlines(keepends=True) if line.startswith("#")]
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("#"))
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for row in rows:
        row.update(updates.get(row["doc_id"], {}))

    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.writelines(header)
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing {path} — run the previous stage first")
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
