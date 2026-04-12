from langchain_core.messages import SystemMessage

from langchain_core.tools import BaseTool

from langsmith import traceable

from app.graph.state import AgentState
from app.services.llms import get_llm



SYSTEM_PROMPT = SystemMessage(content="""
You are a helpful AI assistant.

You have access to the following tools. Use them in this priority order:

1. **rag_tool** (highest priority) — Use this whenever the user asks anything 
   related to the uploaded PDF document. Always prefer this over other tools 
   for document-related questions.

2. **web_search** — Use this when the user asks about current events, breaking 
   news, real-time data, or anything that requires up-to-date information from 
   the web.

3. **Direct answer** (no tool) — Only answer directly from your own knowledge 
   if the question is general and neither tool is needed.

Never guess when a tool can provide a more accurate answer.
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
