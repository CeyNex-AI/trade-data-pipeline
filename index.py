#!/usr/bin/env python3
"""Stage 5 — embed the enriched chunks and upsert them into Qdrant.

    ../ceynex-core/.venv/bin/python index.py [--recreate]

Builds a collection with two named vectors, `dense` and `sparse`, because policy
text needs both: the dense model finds passages that mean the same thing, BM25
finds passages containing the exact token — "GSP+", "MFN", "HS 6109" — and those
are the tokens a trade question turns on. Qdrant fuses the two rankings with RRF
server-side at query time.

**Idempotent, like the KG loaders.** Point ids are a uuid5 of `(doc_id,
chunk_index)`, so re-running upserts over the same points rather than adding a
second copy of the corpus. `--recreate` drops the collection first, which is only
needed when the embedding model or vector size changes.

Writes back nothing: the manifest's `chunk_count` is reported here and set on the
:PolicyDocument node by ceynex-core's loader, which reads this collection.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from qdrant_client import QdrantClient, models

from ceynex.retrieval.schema import (
    DENSE_DIM,
    DENSE_MODEL,
    DENSE_VECTOR,
    INDEXED_KEYWORD_FIELDS,
    SPARSE_MODEL,
    SPARSE_VECTOR,
    point_id,
)
from ceynex.settings import qdrant_collection, qdrant_url

from pipeline_common import ENRICHED, read_jsonl

BATCH = 64


def ensure_collection(client: QdrantClient, name: str, *, recreate: bool) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
        print(f"dropped existing collection {name}")

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
            },
            # IDF is what makes BM25 a BM25: without it every term is weighted
            # equally and "the" counts as much as "GSP+", which is the opposite
            # of why sparse retrieval is here.
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        print(f"created collection {name}")

    for field in INDEXED_KEYWORD_FIELDS:
        # Idempotent in practice: creating an index that exists is a no-op error
        # on some versions, so it is swallowed rather than guarded with a read.
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:  # noqa: BLE001 - "already exists" is the normal case
            if "already exists" not in str(exc).lower():
                print(f"  payload index {field}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=qdrant_collection())
    parser.add_argument("--url", default=qdrant_url() or "http://localhost:6333")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop the collection first — needed only when the vector size changes",
    )
    args = parser.parse_args()

    chunks = list(read_jsonl(ENRICHED))
    if not chunks:
        raise SystemExit("no enriched chunks — run enrich.py first")

    from fastembed import SparseTextEmbedding, TextEmbedding

    print(f"loading {DENSE_MODEL} and {SPARSE_MODEL} ...")
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    client = QdrantClient(url=args.url, timeout=120)
    ensure_collection(client, args.collection, recreate=args.recreate)

    texts = [c["text"] for c in chunks]
    print(f"embedding {len(texts)} chunks ...")
    dense_vectors = list(dense_model.embed(texts, batch_size=BATCH))
    sparse_vectors = list(sparse_model.embed(texts, batch_size=BATCH))

    points = [
        models.PointStruct(
            id=point_id(chunk["doc_id"], chunk["chunk_index"]),
            vector={
                DENSE_VECTOR: dense.tolist(),
                SPARSE_VECTOR: models.SparseVector(
                    indices=sparse.indices.tolist(), values=sparse.values.tolist()
                ),
            },
            payload={
                "doc_id": chunk["doc_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "page": chunk["page"],
                "section": chunk["section"],
                "iso3": chunk["iso3"],
                "hs_prefix": chunk["hs_prefix"],
                "measure_type": chunk["measure_type"],
                "agreement": chunk["agreement"],
                "title": chunk["title"],
                "publisher": chunk["publisher"],
                "url": chunk["url"],
                "language": chunk["language"],
            },
        )
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True)
    ]

    for start in range(0, len(points), BATCH):
        client.upsert(collection_name=args.collection, points=points[start : start + BATCH])
        print(f"  upserted {min(start + BATCH, len(points))}/{len(points)}")

    info = client.get_collection(args.collection)
    print(f"\ncollection {args.collection}: {info.points_count} points")
    print("\nchunks per document (chunk_count for the :PolicyDocument node):")
    for doc_id, count in sorted(Counter(c["doc_id"] for c in chunks).items()):
        print(f"  {doc_id:32s} {count:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
