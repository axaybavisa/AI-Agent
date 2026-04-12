from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults


search = DuckDuckGoSearchResults()

@tool
def web_search(query: str) -> str:
    """
    Use this tool to search the web for current news or real-time information.
    """
    return search.invoke(query)
