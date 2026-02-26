from pydantic import BaseModel, Field
from typing import Annotated

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

"""State definition for RAG graph."""
class ChatState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    user_id: str | None = None
    pdf_id: str | None = None
    retrieved_docs: list[str] = Field(default_factory=list)
    

