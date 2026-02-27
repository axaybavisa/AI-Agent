from langchain_community.vectorstores import FAISS

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage

from langsmith import traceable

from app.graph.state import ChatState
from app.services.llm_service import get_llm
from app.services.vector_store_service import retriver_search
from app.services.pdf_service import UploadFile, pdf_load
from app.services.vector_store_service import build_vectorstore



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
def bind_tool_to_llm(rag_tool):
    """
    Bind tools to the LLM so it knows when and how to call them.
    
    The LLM will automatically decide to call rag_tool
    when the user asks a question that needs document retrieval.

    Args:
        rag_tool: The RAG tool returned from make_rag_tool()

    Returns:
        LLM with tools bound — ready to use in a LangGraph node
    """

    llm = get_llm()

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
