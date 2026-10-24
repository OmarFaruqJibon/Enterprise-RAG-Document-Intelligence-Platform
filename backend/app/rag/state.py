
from typing import Annotated, Any, TypedDict
 
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    question: str
    limit: int

    search_question: str

    # Conversation history
    messages: Annotated[list[AnyMessage], add_messages]

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
