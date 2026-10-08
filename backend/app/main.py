from fastapi import FastAPI
from app.services.llm import ask_llm


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