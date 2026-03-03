from pydantic import BaseModel
from typing import Annotated, TypedDict, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class DocumentMetadata(BaseModel):
    source: str
    page: Optional[int] = None

class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str

    retrieved_docs: list[str]
    doc_metadata: list[DocumentMetadata]

    answer: str
