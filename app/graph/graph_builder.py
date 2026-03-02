from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.nodes.router import UploadFile, setup_rg_pipeline, chat_node
from app.graph.state import ChatState

async def build_graph(upload: UploadFile | None = None):
    llm_with_tools, rag_tool = await setup_rg_pipeline(upload)

    tool_node = ToolNode([rag_tool])

    async def llm_node(state: ChatState):
        return await chat_node(state, llm_with_tools)
    
    builder = StateGraph(ChatState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)

    builder.set_entry_point("llm")

    builder.add_conditional_edges(
        "llm",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "llm")

    graph = builder.compile()

    return graph

