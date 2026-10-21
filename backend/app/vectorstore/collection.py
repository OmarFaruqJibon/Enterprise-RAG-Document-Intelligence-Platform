from qdrant_client.models import Distance, VectorParams

from app.vectorstore.qdrant import get_qdrant_client


COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE = 384


def create_collection() -> str:
    client = get_qdrant_client()

    existing_collections = [
        collection.name
        for collection in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing_collections:
        return "Collection already exists"

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    return "Collection created successfully"