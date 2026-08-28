#!/usr/bin/env python3
"""Stage 1 — download every manifest document into raw/, recording sha256.

    ../ceynex-core/.venv/bin/python fetch.py [--update-manifest] [--only DOC_ID]

**Fails loudly on a bad response.** A 404 page, a Cloudflare interstitial or a
login wall is still 200-with-a-body from `httpx`'s point of view, and quietly
ingesting one produces a document full of navigation text that matches every
query and answers none. Anything that is not a real document is reported and
skipped, never written.

`--update-manifest` writes `sha256` and `retrieved_at` back into the committed
CSV. Run it once the fetch is clean and commit the result: an answer citing a
policy document is only reproducible if the exact bytes it was built from can be
identified later.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date

import httpx

from pipeline_common import RAW_DIR, Document, read_manifest, update_manifest

# Government sites reject the default httpx agent often enough to be worth
# setting. This is an ordinary identifying string, not an attempt to look like a
# browser — the requests are for published public documents at a human pace.
HEADERS = {
    "User-Agent": "CeyNex/0.1 (University of Moratuwa CS3501 research project)",
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
}

TIMEOUT_S = 60.0

#: Below this, the response is a stub — an error page, a redirect notice or a
#: consent wall. Real policy documents are never this short.
MIN_BYTES = 2_000


def suffix_for(response: httpx.Response) -> str:
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type or response.content[:5] == b"%PDF-":
        return ".pdf"
    return ".html"


def fetch_one(client: httpx.Client, doc: Document) -> tuple[str, str] | None:
    """Download one document. Returns `(sha256, path)` or None if unusable."""
    try:
        response = client.get(doc.url, headers=HEADERS, timeout=TIMEOUT_S, follow_redirects=True)
    except httpx.HTTPError as exc:
        print(f"  FAIL {doc.doc_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    if response.status_code != 200:
        print(f"  FAIL {doc.doc_id}: HTTP {response.status_code}", file=sys.stderr)
        return None
    if len(response.content) < MIN_BYTES:
        print(
            f"  FAIL {doc.doc_id}: {len(response.content)} bytes — too short to be the "
            f"document, probably an error or consent page",
            file=sys.stderr,
        )
        return None

    path = RAW_DIR / f"{doc.doc_id}{suffix_for(response)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    digest = hashlib.sha256(response.content).hexdigest()
    print(f"  ok   {doc.doc_id}: {len(response.content):,} bytes -> {path.name}")
    return digest, str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="fetch a single doc_id")
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="write sha256 and retrieved_at back into the committed CSV",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download documents already present in raw/",
    )
    args = parser.parse_args()

    documents = read_manifest()
    if args.only:
        documents = [d for d in documents if d.doc_id == args.only]
        if not documents:
            raise SystemExit(f"no manifest row with doc_id {args.only!r}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    updates: dict[str, dict[str, str]] = {}
    failures: list[str] = []

    print(f"fetching {len(documents)} documents into {RAW_DIR}")
    with httpx.Client() as client:
        for doc in documents:
            existing = list(RAW_DIR.glob(f"{doc.doc_id}.*"))
            if existing and not args.force:
                print(f"  skip {doc.doc_id}: already present ({existing[0].name})")
                continue
            result = fetch_one(client, doc)
            if result is None:
                failures.append(doc.doc_id)
                continue
            digest, _ = result
            updates[doc.doc_id] = {"sha256": digest, "retrieved_at": date.today().isoformat()}

    if args.update_manifest and updates:
        update_manifest(updates)
        print(f"\nmanifest updated for {len(updates)} documents")

    if failures:
        print(f"\n{len(failures)} of {len(documents)} failed: {', '.join(failures)}", file=sys.stderr)
        print(
            "These are corpus gaps, not crashes. Fix the URL in the manifest or record "
            "the gap in its header, then re-run.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
