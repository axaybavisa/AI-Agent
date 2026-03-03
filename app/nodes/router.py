from langchain_core.messages import SystemMessage

from langsmith import traceable

from app.graph.state import AgentState
from app.services.llms import get_llm
from langchain_core.tools import BaseTool


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
async def create_agent_node(tools: list[BaseTool]):
    """Example LangGraph node that takes user input, runs it through the LLM with tools, and returns a response."""
    llm = get_llm().bind_tools(tools)

    async def agent(state: AgentState):
        full_messages = [SYSTEM_PROMPT] + state["messages"]

        response = await llm.ainvoke(full_messages)

        return {"messages": [response]}
    
    return agent
