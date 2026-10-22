from fastapi import FastAPI,HTTPException, Query
from app.services.llm import ask_llm

from app.documents.loader import load_pdf
from app.documents.splitter import split_documents

from app.vectorstore.qdrant import get_qdrant_client
from app.vectorstore.collection import create_collection

from app.vectorstore.ingestion import ingest_pdf
from app.vectorstore.retrieval import search_documents


from app.rag.graph import run_rag_graph


from pydantic import BaseModel, Field
from fastapi import HTTPException

from app.rag.pipeline import answer_question

app = FastAPI(
    title="KnowledgeFlow AI",
    description="Enterprise RAG & Document Intelligence Platform",
    version="0.0.1",
)


@app.get("/")
def root():
    return {
        "message": "KnowledgeFlow AI API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/ask")
def ask(question: str):

    answer = ask_llm(question)

    return {
        "question": question,
        "answer": answer,
    }
    
    
@app.get("/documents/test")
def test_document_ingestion():

    documents = load_pdf("data/employee.pdf")

    chunks = split_documents(documents)

    return {
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "first_chunk": chunks[0].page_content,
        "first_chunk_metadata": chunks[0].metadata,
    }


#-------------   QDRANT DATABASE  ------------
@app.get("/qdrant/test")
def test_qdrant():

    client = get_qdrant_client()

    collections = client.get_collections()

    return {
        "status": "connected",
        "collections": [
            collection.name
            for collection in collections.collections
        ],
    }


@app.post("/qdrant/create-collection")
def create_qdrant_collection():

    result = create_collection()

    return {
        "message": result
    }
    

@app.post("/documents/ingest")
def ingest_document(
    file_path: str = Query(..., description="Path to a PDF file")
):
    try:
        return ingest_pdf(file_path)

    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.get("/documents/search")
def search_document(
    question: str,
    limit: int = Query(default=5, ge=1, le=20),
):
    try:
        results = search_documents(
            question=question,
            limit=limit,
        )

        return {
            "question": question,
            "results_count": len(results),
            "results": results,
        }

    except (ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# RAG

class RAGQuestion(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to answer from uploaded documents",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve",
    )


@app.post("/rag/ask")
def ask_documents(request: RAGQuestion):
    try:
        return answer_question(
            question=request.question,
            limit=request.limit,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        print(f"RAG pipeline error: {error}")
        raise HTTPException(
            status_code=500,
            detail="The RAG pipeline failed. Check the backend logs.",
        ) from error



# langgraph
@app.post("/rag/graph/ask")
def ask_documents_with_graph(request: RAGQuestion):
    try:
        result = run_rag_graph(
            question=request.question,
            limit=request.limit,
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "citation_warnings": result["citation_warnings"],
            "workflow_status": result["status"],
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"LangGraph workflow error: {error}")
        raise HTTPException(
            status_code=500,
            detail="The LangGraph workflow failed. Check backend logs.",
        ) from error
