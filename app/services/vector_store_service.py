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


# ─────────────────────────────────────────────────────────────────────
# VectorStoreService: Builds and manages vectorstores for uploaded PDFs
# ─────────────────────────────────────────────────────────────────────
class VectorStoreService:

    def __init__(self, base_path: str = "faiss_index"):
        self.base_path = base_path
        self.cache: dict[str, FAISS] = {}
        os.makedirs(self.base_path, exist_ok=True)

    @traceable(name="build_vectores")
    async def build_vectorstore(
        self,
        pdf_temp_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100, 
    ) -> FAISS:
            
        # Create embedding instance (professional pattern)
        embedding = get_embedding()
        embedding_name = "models/gemini-embedding-001"

        index_key = _index_key(
            pdf_temp_path,
            chunk_size,
            chunk_overlap,
            embedding_name,
        )

        persist_path = os.path.join(self.base_path, index_key)

        try:
            # ✅ In-memory cache
            if index_key in self.cache:
                print("⚡ Using in-memory cached vectorstore")
                return self.cache[index_key]

            # ✅ If in disk exists → Load
            if os.path.exists(persist_path):
                print(f"✅ Loading existing vectorstore from '{persist_path}'...")

                vectorstore = await asyncio.to_thread(
                    FAISS.load_local,
                    persist_path,
                    embedding,
                    allow_dangerous_deserialization=True,
                )

                self.cache[index_key] = vectorstore
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

            await asyncio.to_thread(
                vectorstore.save_local, 
                persist_path,
            )

            self.cache[index_key] = vectorstore

            print(f"💾 Vectorstore saved to '{persist_path}'")

            return vectorstore

        finally:
            if os.path.exists(pdf_temp_path):
                os.remove(pdf_temp_path)



# ─────────────────────────────────────────
# RAG pipeline components
# ─────────────────────────────────────────
class RetrieverService:

    def __init__(
        self,
        search_type: str = "mmr",
        k: int = 4,
        fetch_k: int = 20,
    ):
        self.search_type = search_type
        self.k = k
        self.fetch_k = fetch_k

    @traceable(name="retriever_search")
    async def search(
        self,
        vectorstore: FAISS,
        query: str,
    ) -> list:

        retriever = vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs={
                "k": self.k,
                "fetch_k": self.fetch_k,
            },
        )

        docs = await retriever.ainvoke(query)
        return docs


# ─────────────────────────────────────────
# RAG tool for LangGraph agent
# ─────────────────────────────────────────
class RAGToolService:

    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        self.retriever = RetrieverService()

    def create_tool(self):

        @tool
        async def rag_tool(query: str) -> dict:
            """
            Search the uploaded PDF and return relevant context.

            Args:
                query: The user's question related to the document.

            Returns:
                A dictionary containing:
                - query
                - context
                - metadata
                - num_docs_retrieved
            """

            docs = await self.retriever.search(
                self.vectorstore,
                query,
            )

            context = "\n\n".join(doc.page_content for doc in docs)
            metadata = [doc.metadata for doc in docs]

            return {
                "query": query,
                "context": context,
                "metadata": metadata,
                "num_docs_retrieved": len(docs),
            }

        return rag_tool