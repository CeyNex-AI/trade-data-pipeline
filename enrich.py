#!/usr/bin/env python3
"""Stage 4 — tag every chunk with the graph entities it is about.

    ../ceynex-core/.venv/bin/python enrich.py

The tags come from `ceynex.retrieval.tagging`, imported rather than reimplemented:
the values written here are the values the agent's filters compare against, and
two copies of that vocabulary would drift into an empty result set that raises
nothing.

**`iso3` is the only manifest tag stamped onto every chunk, and that asymmetry is
deliberate.** A passage inside the UK trade strategy is UK policy whether or not
the word "UK" appears in it, so the issuer's iso3 is always correct. It is
unioned with countries named in the text, because a US tariff notice that
mentions Sri Lanka is about both and a question about either should find it.

`hs_prefix` and `agreement` are taken from the **chunk text only**. Measured on
the first run, unioning the manifest's document-level `hs_focus` into every chunk
tagged all 293 chunks of the National Export Strategy with all six of its HS
codes — which makes the goods filter match everything and therefore filter
nothing. A tag that is true of the document is not true of each of its
paragraphs, and a filter that always passes is worse than no filter: it costs the
same and buys a false sense that retrieval was scoped.

Document-level `hs_focus` and `agreements` are still used — by the KG loader, to
build `APPLIES_TO` and `DESCRIBES` edges from the :PolicyDocument node. That is
the right level for them.
"""

from __future__ import annotations

from ceynex.retrieval.tagging import agreements_in, countries_in, hs_prefixes_in, measure_type_of

from pipeline_common import CHUNKS, ENRICHED, read_jsonl, read_manifest, write_jsonl


def main() -> int:
    documents = {doc.doc_id: doc for doc in read_manifest()}
    records: list[dict] = []
    measure_counts: dict[str, int] = {}
    untagged_hs = 0

    for chunk in read_jsonl(CHUNKS):
        doc = documents.get(chunk["doc_id"])
        if doc is None:
            continue
        text = chunk["text"]

        iso3 = tuple(dict.fromkeys(doc.iso3 + countries_in(text)))
        hs_prefix = hs_prefixes_in(text)
        agreements = agreements_in(text)
        measure = measure_type_of(text)

        measure_counts[measure] = measure_counts.get(measure, 0) + 1
        if not hs_prefix:
            untagged_hs += 1

        records.append(
            {
                **chunk,
                "iso3": list(iso3),
                "hs_prefix": list(hs_prefix),
                "agreement": list(agreements),
                "measure_type": measure,
                "title": doc.title,
                "publisher": doc.publisher,
                "url": doc.url,
                "language": doc.language,
            }
        )

    write_jsonl(ENRICHED, records)
    print(f"wrote {len(records)} enriched chunks to {ENRICHED}\n")

    print("measure_type distribution:")
    for measure, count in sorted(measure_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {measure:18s} {count:5d}  ({count / max(1, len(records)):.0%})")

    # Not a warning — a chunk with no HS tag is general trade policy, which most
    # of a strategy document is. It stays searchable; it is simply only reachable
    # by a query that does not filter on goods. Reported because a corpus where
    # this is near 100% would mean the tagger is failing, and that is worth
    # noticing before the evaluation blames the retriever.
    print(f"\nchunks with no HS tag: {untagged_hs} of {len(records)} ({untagged_hs / max(1, len(records)):.0%})")
    print("  (general trade policy — reachable, but not by a goods-filtered query)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
