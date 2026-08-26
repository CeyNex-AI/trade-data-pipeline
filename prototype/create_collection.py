from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(
    host="localhost",
    port=6337,
)

client.create_collection(
    collection_name="ceynex",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
    )
)