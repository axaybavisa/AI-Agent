from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.services.pipelines import UploadFile, build_tools
from app.nodes.router import create_agent_node
from app.graph.state import AgentState

async def build_graph(upload: UploadFile | None = None):
   
   tools = await build_tools(upload)

   agent_node = await create_agent_node(tools)

   tool_node = ToolNode(tools)


   builder = StateGraph(AgentState)

   builder.add_node("agent", agent_node)
   builder.add_node("tools", tool_node)

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

   graph = builder.compile()

   return graph