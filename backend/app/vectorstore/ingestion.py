from uuid import uuid5, NAMESPACE_URL

from qdrant_client.models import PointStruct

from app.documents.loader import load_pdf
from app.documents.splitter import split_documents
from app.embeddings.model import get_embedding_model
from app.vectorstore.qdrant import get_qdrant_client
from app.vectorstore.collection import (
    COLLECTION_NAME,
    create_collection,
)


def ingest_pdf(file_path: str) -> dict:

    # Load PDF
    documents = load_pdf(file_path)

    if not documents:
        raise ValueError("The PDF contains no readable pages.")

    # Split the pages into smaller documents
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were generated from the PDF.")

    # embedding model
    embedding_model = get_embedding_model()

    # Convert chunk into a vector
    texts = [chunk.page_content for chunk in chunks]

    vectors = embedding_model.embed_documents(texts)

    create_collection()

    client = get_qdrant_client()

    # Qdrant points
    points = []

    for chunk, vector in zip(chunks, vectors):

        metadata = dict(chunk.metadata)

        point_id = str(
            uuid5(
                NAMESPACE_URL,
                f"{metadata.get('source', file_path)}:"
                f"{metadata.get('page', 0)}:"
                f"{chunk.page_content}",
            )
        )

        payload = {
            "text": chunk.page_content,
            "source": metadata.get("source", file_path),
            "page": metadata.get("page", 0),
            "metadata": metadata,
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    # 7. Store in Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return {
        "source": file_path,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "vectors_stored": len(points),
        "collection": COLLECTION_NAME,
    }