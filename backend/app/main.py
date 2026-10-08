from fastapi import FastAPI
from app.services.llm import ask_llm

from app.documents.loader import load_pdf
from app.documents.splitter import split_documents


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