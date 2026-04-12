from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.services.pipelines import UploadFile, build_tools
from app.nodes.router import create_agent_node
from app.graph.state import AgentState


async def build_graph(
    upload: UploadFile | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    tools = await build_tools(upload)
    agent_node = await create_agent_node(tools)

    builder = StateGraph(AgentState, name="rag_agent_graph")

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("agent")

    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "agent")

    graph = builder.compile(checkpointer=checkpointer)

    return graph
