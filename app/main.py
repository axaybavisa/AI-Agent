from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from uuid import uuid4
from app.graph.graph_builder import build_graph
from pydantic import BaseModel


app = FastAPI()

# ⚠️ Use Redis/DB in production
sessions = {}


@app.post("/upload")
async def upload_endpoint(
    session_id: Optional[str] = Form(None),
    upload: Optional[UploadFile] = File(None),
):
    """
    Build graph.
    - If PDF uploaded → RAG enabled
    - If not → only other tools enabled
    """

    # Generate session if not provided
    if not session_id:
        session_id = str(uuid4())

    graph = await build_graph(upload)

    sessions[session_id] = graph

    return {
        "status": "Graph built",
        "session_id": session_id,
        "rag_enabled": upload is not None,
    }




class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):

    graph = sessions.get(request.session_id)

    # If session doesn't exist → build default (no PDF)
    if not graph:
        graph = await build_graph()
        sessions[request.session_id] = graph

    result = await graph.ainvoke({
        "messages": [
            {"role": "user", "content": request.message}
        ]
    })

    return {
        "response": result["messages"][-1].content
    }