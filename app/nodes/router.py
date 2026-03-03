from langchain_core.messages import SystemMessage

from langsmith import traceable

from app.graph.state import ChatState
from app.tools.llm_service import get_llm
from app.tools.pdf_service import UploadFile, pdf_load
from app.tools.vector_store_service import VectorStoreService, RAGToolService

from app.tools.web_service import web_search 

# ─────────────────────────────────────────
# Bind tools to LLM
# ─────────────────────────────────────────
def bind_tool_to_llm(rag_tool, web_search):
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

    tools = [rag_tool, web_search]
    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools


# ─────────────────────────────────────────
# Full pipeline — wire everything together
# ─────────────────────────────────────────
async def setup_rg_pipeline(upload: UploadFile | None = None):
    """
    Builds RAG pipeline once per uploaded PDF.
    Returns:
        llm_with_tools -> LLM capable of tool calling
        rag_tool       -> Retrieval tool
    """ 

    rag_tool = None
    tools = []

    tools.append(web_search)

    if upload:
        tmp_path = await pdf_load(upload)

        if tmp_path:
            vectorstore = await VectorStoreService().build_vectorstore(tmp_path)
            rag_tool = RAGToolService(vectorstore).create_tool()
            tools.append(rag_tool)

    base_llm = get_llm()
    
    if tools:
        llm_with_tools = base_llm.bind_tools(tools)
    else:
        llm_with_tools = base_llm    

    return llm_with_tools, tools


SYSTEM_PROMPT = SystemMessage(content="""
You are a helpful AI assistant.

You have access to tools:
- Use the web_search tool when the user asks about current events,
  latest news, real-time information, or things happening today.
- Use the document retrieval tool when the question is about the uploaded PDF.
- If no tool is required, answer directly.

Always prefer tools when the question requires up-to-date information.
""")


@traceable(name="answer_generation")
async def chat_node(state: ChatState, llm_with_tools):
    """Example LangGraph node that takes user input, runs it through the LLM with tools, and returns a response."""
    messages = state['messages']

    full_messages = [SYSTEM_PROMPT] + messages

    response = await llm_with_tools.ainvoke(full_messages)

    return {"messages": [response]}
