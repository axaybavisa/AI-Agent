from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults


web_search = DuckDuckGoSearchResults(
    name="web_search",
    description="Search the web for latest news and current events."
)