import os
from typing import Dict
from fastapi import FastAPI, UploadFile
from app.graph.graph_builder import build_graph


"""This name has been show on langsmith traking."""
os.environ['LANGCHAIN_PROJECT'] = "RAG-App"

app = FastAPI()

sessions: Dict[str, any] = {} 

@app.post("/upload")
async def upload_pdf(session_id: str, upload: UploadFile):

    graph = await build_graph(upload)

    sessions[session_id] = graph

    return {"status": "Graph built successfully"}


@app.post("/chat")
async def chat(session_id: str, message: str):

    graph = sessions.get(session_id)

    if not graph:
        return {"error": "Session not found. Upload PDF first."}

    result = await graph.ainvoke({
        "messages": [
            {"role": "user", "content": message}
        ]
    })

    return result["messages"][-1].content



