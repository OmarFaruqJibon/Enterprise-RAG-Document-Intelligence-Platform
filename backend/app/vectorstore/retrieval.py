import re

from app.embeddings.model import get_embedding_model
from app.vectorstore.qdrant import get_qdrant_client
from app.vectorstore.collection import COLLECTION_NAME


# Starting value for testing
DEFAULT_SCORE_THRESHOLD = 0.45

# Normalize text for exact duplicate detection
def normalize_text(text: str) -> str:
    
    return re.sub(r"\s+", " ", text).strip().casefold()


def search_documents(
    question: str,
    limit: int = 5,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict]:
    if not question or not question.strip():
        raise ValueError("Search question cannot be empty.")

    if not 1 <= limit <= 20:
        raise ValueError("Limit must be between 1 and 20.")

    if not -1.0 <= score_threshold <= 1.0:
        raise ValueError("Score threshold must be between -1 and 1.")

    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(question.strip())

    client = get_qdrant_client()

    candidate_limit = min(limit * 3, 60)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=candidate_limit,
        with_payload=True,
    )

    matches = []
    seen_chunks = set()

    for point in results.points:
        payload = point.payload or {}
        text = payload.get("text", "")
        score = point.score

        if not text.strip() or score is None:
            continue

        if score < score_threshold:
            continue

        source = str(payload.get("source", "Unknown"))
        normalized = normalize_text(text)

        duplicate_key = (source, normalized)
        if duplicate_key in seen_chunks:
            continue

        seen_chunks.add(duplicate_key)

        matches.append({
            "text": text,
            "source": source,
            "page": payload.get("page", 0),
            "score": float(score),
        })

        if len(matches) >= limit:
            break

    return matches
