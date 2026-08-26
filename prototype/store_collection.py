from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


# Configuration
TEXT_FOLDER = Path("./ocr_output")
COLLECTION_NAME = "ceynex"

# Connect to your ceynex Qdrant
client = QdrantClient(
    host="localhost",
    port=6337
)

# Load MiniLM
model = SentenceTransformer("all-MiniLM-L6-v2")


# Read files
texts = []
metadata = []

for file_path in TEXT_FOLDER.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")

    texts.append(text)
    metadata.append({
        "filename": file_path.name,
        "path": str(file_path),
    })


print(f"Found {len(texts)} text files")


# Generate embeddings
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# Create Qdrant points
points = []

for i, (text, embedding, meta) in enumerate(
    zip(texts, embeddings, metadata)
):
    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": text,
                **meta
            }
        )
    )


# Upload to Qdrant
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} documents to Qdrant")