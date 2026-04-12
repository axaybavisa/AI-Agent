from app.services.pdf_loader import UploadFile, pdf_load
from app.services.vector_store import VectorStoreService, RAGToolService
from app.services.web_search import web_search 


# ─────────────────────────────────────────
# Full pipeline — wire everything together
# ─────────────────────────────────────────
async def build_tools(upload: UploadFile | None = None):
    """
    Returns list of tools.
    No LLM binding here.
    Graph will handle execution.
    """

    tools = []

    if upload:
        tmp_path = await pdf_load(upload)

        if tmp_path:
            vectorstore = await VectorStoreService().build_vectorstore(tmp_path)
            rag_tool = RAGToolService(vectorstore).create_tool()
            tools.append(rag_tool)  

    tools.append(web_search)        

    return tools
