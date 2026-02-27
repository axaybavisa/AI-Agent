import os
import asyncio

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.documents import Document

from langsmith import traceable

from app.services.pdf_service import _index_key
from app.services.llm_service import get_embedding


"""For vectorestore."""
BASE_VECTORSTORE_PATH = "faiss_index"
VECTORSTORE_CACHE = {}

# ─────────────────────────────────────────
# Build FAISS vectorstore
# ─────────────────────────────────────────
@traceable(name="build_vectorstore")
async def build_vectorstore( 
    pdf_temp_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    ) -> FAISS:

    os.makedirs(BASE_VECTORSTORE_PATH, exist_ok=True)

    # Create embedding instance (professional pattern)
    embedding = get_embedding()
    embedding_name = "models/gemini-embedding-001"

    index_key = _index_key(
        pdf_temp_path,
        chunk_size,
        chunk_overlap,
        embedding_name,
    )

    persist_path = os.path.join(BASE_VECTORSTORE_PATH, index_key)

    try:
        # ✅ In-memory cache
        if index_key in VECTORSTORE_CACHE:
            print("⚡ Using in-memory cached vectorstore")
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

        return vectorstore

    finally:
        if os.path.exists(pdf_temp_path):
            os.remove(pdf_temp_path)  # always deleted, even if an error occurs



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
        search_kwargs={
             "k": 4,
            "fetch_k": 20,
        }
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
            query,
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
