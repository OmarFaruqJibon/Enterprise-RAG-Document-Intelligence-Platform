# retrieval.py
import re

from rank_bm25 import BM25Okapi

from app.embeddings.model import get_embedding_model
from app.vectorstore.qdrant import get_qdrant_client
from app.vectorstore.collection import COLLECTION_NAME


# Starting value for testing
DEFAULT_SCORE_THRESHOLD = 0.45

# Number of chunks to fetch per Qdrant scroll request
SCROLL_BATCH_SIZE = 256


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def tokenize_text(text: str) -> list[str]:
    """Convert text into lowercase word tokens for BM25."""
    return re.findall(r"\b\w+\b", text.casefold())


def search_documents(
    question: str,
    limit: int = 5,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict]:
    """Retrieve relevant chunks using dense vector search."""

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

        if not isinstance(text, str) or not text.strip():
            continue

        if score is None or score < score_threshold:
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
            "retrieval_method": "vector",
        })

        if len(matches) >= limit:
            break

    return matches


def search_documents_bm25(
    question: str,
    limit: int = 5,
) -> list[dict]:
    """Retrieve relevant chunks using BM25 keyword ranking."""

    if not question or not question.strip():
        raise ValueError("Search question cannot be empty.")

    if not 1 <= limit <= 20:
        raise ValueError("Limit must be between 1 and 20.")

    client = get_qdrant_client()

    # Load all stored chunks from Qdrant in batches.
    # This simple implementation is suitable for a small corpus.
    all_points = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        all_points.extend(points)

        if next_offset is None:
            break

        offset = next_offset

    # Extract text and metadata, skipping empty and duplicate chunks.
    documents = []
    seen_chunks = set()

    for point in all_points:
        payload = point.payload or {}
        text = payload.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        source = str(payload.get("source", "Unknown"))
        normalized = normalize_text(text)
        duplicate_key = (source, normalized)

        if duplicate_key in seen_chunks:
            continue

        seen_chunks.add(duplicate_key)

        tokens = tokenize_text(text)

        if not tokens:
            continue

        documents.append({
            "text": text,
            "source": source,
            "page": payload.get("page", 0),
            "tokens": tokens,
        })

    if not documents:
        return []

    query_tokens = tokenize_text(question)

    if not query_tokens:
        return []

    # Build a BM25 index over the document chunks.
    bm25 = BM25Okapi(
        [document["tokens"] for document in documents]
    )

    scores = bm25.get_scores(query_tokens)

    # Sort candidate chunk indices by descending BM25 score.
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    matches = []

    for index in ranked_indices:
        score = float(scores[index])

        # No shared query terms means no keyword match.
        if score <= 0:
            continue

        document = documents[index]

        matches.append({
            "text": document["text"],
            "source": document["source"],
            "page": document["page"],
            "score": score,
            "retrieval_method": "bm25",
        })

        if len(matches) >= limit:
            break

    return matches