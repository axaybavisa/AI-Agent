from langchain_core.messages import SystemMessage

from langsmith import traceable

from app.graph.state import ChatState
from app.services.llm_service import get_llm
from app.services.pdf_service import UploadFile, pdf_load
from app.services.vector_store_service import build_vectorstore, make_rag_tool


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
