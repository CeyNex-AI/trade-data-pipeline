# trade-data-pipeline

Builds the **policy-document index** that CeyNex searches when a question concerns a destination market's trade policy. It takes the trade and foreign-policy documents published by Sri Lanka's main export markets, plus Sri Lanka's own National Export Strategy, and processes them in five steps:

- downloads them;
- extracts their text;
- cuts it into section-aware passages;
- tags each passage with the countries, HS codes and trade agreements it mentions;
- indexes the passages in Qdrant for hybrid (dense + BM25) retrieval.

In production the collection `ceynex_policy` holds about 544 passages. The `trade_economics` agent in [`ceynex-core`](https://github.com/CeyNex-AI/ceynex-core) retrieves them and shows them as evidence with a link to the source page.

Group 07, Project P16, CS3501 Data Science and Engineering Project, University of Moratuwa.

## How it fits

```
ceynex-core/ceynex/data/reference/policy_documents.csv   (the manifest, committed in ceynex-core)
        │
        ▼
fetch.py  ──▶ raw/<doc_id>.pdf|html           (sha256 recorded)
extract.py ─▶ build/extracted.jsonl           (one record per page)
chunk.py  ──▶ build/chunks.jsonl              (section-aware, overlapping)
enrich.py ──▶ build/enriched.jsonl            (+ iso3, hs_prefix, agreement, measure tags)
index.py  ──▶ Qdrant collection ceynex_policy  (dense + sparse vectors)
        │
        ▼
ceynex-core: kg/queries.py picks the documents for the asked market in Cypher,
             retrieval/client.py searches only those passages,
             trade_economics turns hits into Evidence(source_id="POLICY", url=...)
```

The manifest lives in `ceynex-core` because the knowledge-graph loader reads the same file to create `:PolicyDocument` nodes. That way the graph and the index always agree on which documents exist.

## The stages

| Script | Stage | Notes |
|---|---|---|
| `fetch.py` | Download every manifest document into `raw/` | Rejects stubs, error pages and consent walls instead of saving them. `--update-manifest` writes `sha256` and `retrieved_at` back into the manifest so every cited passage can be traced to exact bytes. `--only DOC_ID` fetches one document. |
| `extract.py` | Text per page | `pdfplumber` for PDFs and BeautifulSoup for HTML. Pages with almost no text are listed as OCR candidates, and the stage stops there. Re-run with `--use-ocr` after OCR. |
| `pdf_to_png.py`, `ocr_pages.py` | OCR fallback for scanned pages | Renders pages to PNG and transcribes them with a local Qwen3-VL-2B-Instruct model, preserving layout. Needs a GPU-capable PyTorch install and the model weights in `Qwen3-VL-2B-Instruct/`. |
| `chunk.py` | Passages | Splits on headings, then paragraphs, and only cuts by length inside an over-long paragraph. Default target is 1,600 characters with 15% overlap (`--target-chars`, `--overlap`), which keeps each passage inside the embedding model's 512-token window. |
| `enrich.py` | Entity tags | Imports the tagging vocabulary from `ceynex.retrieval.tagging`, so the indexed tags match the agent's filters exactly. The issuing country's iso3 is stamped on every passage. HS codes and agreements come from the passage text only. |
| `index.py` | Qdrant | Two named vectors: `dense` (`BAAI/bge-base-en-v1.5`) and `sparse` (`Qdrant/bm25`), fused with RRF at query time. Point ids are a uuid5 of `(doc_id, chunk_index)`, so re-running is idempotent. `--recreate` drops the collection first. |
| `pipeline_common.py` | Shared paths, manifest reader, JSONL helpers | |
| `prototype/` | The first single-vector MiniLM prototype | Kept for reference. Not used by the pipeline. |

Only English documents are indexed, because the dense model is English-only. Other documents stay in the manifest and the graph, recorded as found and deliberately skipped.

## Running it

The scripts import `ceynex.retrieval` and `ceynex.settings` from `ceynex-core`, so run them with that checkout's interpreter. Clone the repos side by side:

```
ceynex-contracts/
ceynex-core/            (make install done, Qdrant up via make up)
trade-data-pipeline/
```

```bash
cd trade-data-pipeline
PY=../ceynex-core/.venv/bin/python

$PY fetch.py                 # add --update-manifest once the fetch is clean
$PY extract.py               # lists scanned pages, if any
#   optional OCR for those pages:
#   python pdf_to_png.py && python ocr_pages.py --images-dir pages && $PY extract.py --use-ocr
$PY chunk.py
$PY enrich.py
$PY index.py                 # --recreate when the embedding model changes
```

Then, in `ceynex-core`, run `make kg-load` (it includes `--policy`) so the `:PolicyDocument` nodes record their chunk counts.

`index.py` reads the Qdrant URL and collection from `ceynex-core`'s settings (`QDRANT_URL`, `QDRANT_COLLECTION`). It can also take `--url` and `--collection`.

Generated folders (`raw/`, `pages/`, `ocr_output/`, `build/`, `qdrant_storage/`) and downloaded model weights are git-ignored. Everything in them can be reproduced from the manifest.

## Checking the result

In `ceynex-core`:

```bash
make eval-policy-baseline   # 15 policy questions with retrieval off
make eval-policy            # the same with retrieval on
```

The difference between the two runs is the accuracy claim for policy retrieval. Method and results are in `ceynex-core/docs/POLICY_RETRIEVAL.md` and `docs/EVALUATION.md`.

## Team

Pipeline: Thisen Ekanayake (230170B). Team: Senindu Dinapura (230151T), Dhinanjaya Fernando (230181J). Supervisor: Dr. Chathuranga Hettiarachchi. Teaching Assistant: Birunthaban Rajendram. University of Moratuwa.
