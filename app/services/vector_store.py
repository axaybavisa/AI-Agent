import os
import asyncio

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

from langsmith import traceable

from app.services.llms import get_embedding
from app.services.pdf_loader import IndexKeyGenerator
from app.core.config import get_settings

setting = get_settings()


# ─────────────────────────────────────────────────────────────────────
# VectorStoreService: Builds and manages vectorstores for uploaded PDFs
# ─────────────────────────────────────────────────────────────────────
class VectorStoreService:

    _cache: dict[str, FAISS] = {}  # class-level cache, shared across all instances

    def __init__(self, base_path: str = "faiss_index"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    @traceable(name="build_vectores")
    async def build_vectorstore(self, pdf_temp_path: str) -> FAISS:
            
        # Create embedding instance 
        embedding = get_embedding()

        index_generator = IndexKeyGenerator(
            pdf_path=pdf_temp_path
        )

        index_key: str = index_generator.generate_index_key()
        persist_path = os.path.join(self.base_path, index_key)

        try:
            # ✅ In-memory cache
            if index_key in VectorStoreService._cache:
                print("⚡ Using in-memory cached vectorstore")
                return VectorStoreService._cache[index_key]

            # ✅ If in disk exists → Load
            if os.path.exists(persist_path):
                print(f"✅ Loading existing vectorstore from '{persist_path}'...")

                vectorstore = await asyncio.to_thread(
                    FAISS.load_local,
                    persist_path,
                    embedding,
                    allow_dangerous_deserialization=True,
                )

                VectorStoreService._cache[index_key] = vectorstore
                return vectorstore

            # ❌ Else → Build
            print("🔨 Building new vectorstore...")

            loader = PyPDFLoader(pdf_temp_path)
            docs = await loader.aload()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=setting.CHUNK_SIZE,
                chunk_overlap=setting.CHUNK_OVERLAP,
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

            VectorStoreService._cache[index_key] = vectorstore

            print(f"💾 Vectorstore saved to '{persist_path}'")

            return vectorstore

        finally:
            if os.path.exists(pdf_temp_path):
                os.remove(pdf_temp_path)



# ─────────────────────────────────────────
# RAG pipeline components
# ─────────────────────────────────────────
class RetrieverService:

    def __init__(self):

        self.search_type = setting.SEARCH_TYPE
        self.k = setting.TOP_K
    
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
            Use this tool ONLY when the user asks questions
            about the uploaded PDF document.

            It retrieves relevant sections from the document
            to help answer document-related questions.

            Do NOT use it for general knowledge questions.
            """

            docs = await self.retriever.search(
                self.vectorstore,
                query,
            )

            MAX_CHARS = 4000
            context = ""

            for doc in docs:
                if len(context) + len(doc.page_content) > MAX_CHARS:
                    break
                context += doc.page_content + "\n\n"

            return context.strip()    

        return rag_tool