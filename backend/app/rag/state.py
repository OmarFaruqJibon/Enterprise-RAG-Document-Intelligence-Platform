
from typing import Any, TypedDict


class RAGState(TypedDict):
    question: str
    limit: int

    # Retrieved document chunks
    matches: list[dict[str, Any]]

    # Generated answer and validated citations
    answer: str
    sources: list[dict[str, Any]]
    citation_warnings: list[str]

    # Workflow status
    status: str

    # Evidence assessment
    evidence_sufficient: bool
    evidence_reason: str
