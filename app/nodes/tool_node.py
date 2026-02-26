import os
import hashlib
import tempfile
import json 
import asyncio
import aiofiles

from pathlib import Path
from fastapi import UploadFile

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage

from langsmith import traceable
from dotenv import load_dotenv

from app.graph.state import ChatState


# ─────────────────────────────────────────
# ENV SETUP
# ─────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────
# LLM INIT
# ─────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
)


"""For vectorestore."""
BASE_VECTORSTORE_PATH = "faiss_index"
VECTORSTORE_CACHE = {}


# ─────────────────────────────────────────
# FILE FINGERPRINT
# ─────────────────────────────────────────
def _file_fingerprint(path: str)-> dict:
    p = Path(path)
    h = hashlib.sha256()

    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    stat = p.stat()

    return {
        "sha256": h.hexdigest(),
        "size": stat.st_size, 
    }        

def _index_key(
        pdf_path: str, 
        chunk_size: int, 
        chunk_overlap: int, 
        embed_model_name: str
    ) -> str:

    fingerprint = _file_fingerprint(pdf_path)

    meta = {
        "pdf_fingerprint": fingerprint["sha256"],
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }

    return hashlib.sha256(
        json.dumps(meta, sort_keys=True).encode("utf-8")
        ).hexdigest()


# ─────────────────────────────────────────
# Load & split PDF
# ─────────────────────────────────────────
@traceable(name="load_pdf")
async def pdf_load(upload: UploadFile) -> str:
    """Save uploaded PDF and return temp path."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name

    async with aiofiles.open(tmp_path, "wb") as f:
       # reads the upload in 1MB chunks and writes each chunk to a file
       while chunk := await upload.read(1024 * 1024):
        await f.write(chunk)

    return tmp_path    


# ─────────────────────────────────────────
# Build FAISS vectorstore
# ─────────────────────────────────────────
@traceable(name="build_vectorstore")
async def build_vectorstore( 
    pdf_temp_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embedding_model: str = "models/gemini-embedding-001"
    ) -> FAISS:

    os.makedirs(BASE_VECTORSTORE_PATH, exist_ok=True)

    """Convert document splits into a FAISS vectorstore."""
    embedding = GoogleGenerativeAIEmbeddings(model=embedding_model)

    index_key = _index_key(
        pdf_temp_path,
        chunk_size,
        chunk_overlap,
        embedding_model,
    )

    persist_path = os.path.join(BASE_VECTORSTORE_PATH, index_key)

    # ✅ In-memory cache
    if index_key in VECTORSTORE_CACHE:
        print("⚡ Using in-memory cached vectorstore")
        os.remove(pdf_temp_path)
        return VECTORSTORE_CACHE[index_key]

    # ✅ If exists → Load
    if os.path.exists(persist_path):
        print(f"✅ Loading existing vectorstore from '{persist_path}'...")

        vectorstore = await asyncio.to_thread(
            FAISS.load_local,
            persist_path, 
            embedding, 
            allow_dangerous_deserialization=True
        )

        VECTORSTORE_CACHE[index_key] = vectorstore
        os.remove(pdf_temp_path)
        return vectorstore

    # ❌ Else → Build
    print("🔨 Building new vectorstore...")

    loader = PyPDFLoader(pdf_temp_path)
    docs = await loader.aload()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
    )

    splits = splitter.split_documents(docs)

    vectorstore = await asyncio.to_thread(
        FAISS.from_documents,
        splits, 
        embedding,
    )

    await asyncio.to_thread(vectorstore.save_local, persist_path)

    VECTORSTORE_CACHE[index_key] = vectorstore

    print(f"💾 Vectorstore saved to '{persist_path}'")

    os.remove(pdf_temp_path)    

    return vectorstore


# ─────────────────────────────────────────
# Build retriever & search
# ─────────────────────────────────────────
@traceable(name="retriever_search")
async def retriver_search(vectorstore: FAISS, query: str) -> list:
    """
    Search the vectorstore for chunks relevant to the query.

    Args:
        vectorstore: Your FAISS vectorstore from build_vectorstore()
        query: The user's question or search string

    Returns:
        A list of the 4 most relevant document chunks
    """
    retriever = vectorstore.as_retriever(
        # here we can use mmr or semalirity
        search_type="mmr", 
        search_kwargs={"score_threshold": 0.7}
    )

    docs = await retriever.ainvoke(query)

    return docs


# ─────────────────────────────────────────
# RAG tool for LangGraph agent
# ─────────────────────────────────────────
def make_rag_tool(vectorstore: FAISS):
    """
    Factory function — builds the rag_tool with vectorstore baked in.
    Call this once after build_vectorstore() and pass the result to your agent.
    """

    @tool 
    async def rag_tool(query: str) -> dict:
        """
        Retrieve relevant information from the pdf document.
        Use this tool when the user asks factual / conceptual questions
        that might be answered from the stored documents.
        """
        docs: list[Document] = await retriver_search(
            vectorstore, 
            query
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        metadata = [doc.metadata for doc in docs]

        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "num_docs_retrieved": len(docs) 
        }
    
    return rag_tool


# ─────────────────────────────────────────
# Bind tools to LLM
# ─────────────────────────────────────────
def bind_tool_to_llm(rag_tool) -> ChatGoogleGenerativeAI:
    """
    Bind tools to the LLM so it knows when and how to call them.
    
    The LLM will automatically decide to call rag_tool
    when the user asks a question that needs document retrieval.

    Args:
        rag_tool: The RAG tool returned from make_rag_tool()

    Returns:
        LLM with tools bound — ready to use in a LangGraph node
    """
    tools = [rag_tool]
    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools


# ─────────────────────────────────────────
# Full pipeline — wire everything together
# ─────────────────────────────────────────
async def setup_rg_pipeline(upload: UploadFile):
    """Run once when a PDF is uploaded. Returns a ready-to-use rag_tool."""
    
    tmp_path = await pdf_load(upload)

    vectorstore = await build_vectorstore(
        pdf_temp_path=tmp_path,
    )

    rag_tool = make_rag_tool(vectorstore)
    llm_with_tools = bind_tool_to_llm(rag_tool)

    return llm_with_tools, rag_tool


SYSTEM_PROMPT = SystemMessage(content="""
You are a helpful AI assistant with access to a document retrieval tool.
When the user asks a question, use the rag_tool to search the uploaded PDF
and answer based on the retrieved context.
If the answer is not found in the documents, say so honestly.
""")


@traceable(name="answer_generation")
async def chat_node(state: ChatState, llm_with_tools):
    """Example LangGraph node that takes user input, runs it through the LLM with tools, and returns a response."""
    messages = state.messages

    full_messages = [SYSTEM_PROMPT] + messages

    response = await llm_with_tools.ainvoke(full_messages)

    return {"messages": [response]}
