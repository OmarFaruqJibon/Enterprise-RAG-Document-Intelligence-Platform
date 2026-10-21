from app.embeddings.model import get_embedding_model
from app.vectorstore.qdrant import get_qdrant_client
from app.vectorstore.collection import COLLECTION_NAME


def search_documents(
    question: str,
    limit: int = 5,
) -> list[dict]:

    embedding_model = get_embedding_model()

    # Convert question into a vector
    query_vector = embedding_model.embed_query(question)

    client = get_qdrant_client()

    # Search for similar document vectors
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    matches = []

    for point in results.points:

        payload = point.payload or {}

        matches.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", "Unknown"),
                "page": payload.get("page", 0),
                "score": point.score,
            }
        )

    return matches