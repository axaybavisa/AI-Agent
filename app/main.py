from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from app.graph.graph_builder import build_graph
from app.services.postgres_db import get_checkpointer


# ── state shared across requests ─────────────────────────────────────────────
_checkpointer = None   # set during lifespan
_sessions: dict = {}   # session_id → compiled graph (per-PDF tools differ)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _checkpointer
    async with get_checkpointer() as cp:
        _checkpointer = cp
        yield
    _checkpointer = None


app = FastAPI(lifespan=lifespan)


# ── request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_endpoint(
    session_id: Optional[str] = Form(None),
    upload: Optional[UploadFile] = File(None),
):
    if not session_id:
        session_id = str(uuid4())

    graph = await build_graph(upload, checkpointer=_checkpointer)
    _sessions[session_id] = graph

    return {
        "status": "Graph built",
        "session_id": session_id,
        "rag_enabled": upload is not None,
    }


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    graph = _sessions.get(request.session_id)

    # No prior upload for this session → build a plain (web-search-only) graph
    if not graph:
        graph = await build_graph(checkpointer=_checkpointer)
        _sessions[request.session_id] = graph

    config = {"configurable": {"thread_id": request.session_id}}

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": request.message}]},
        config=config,
    )

    return {"response": result["messages"][-1].content}
