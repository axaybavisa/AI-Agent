from langchain_community.tools import DuckDuckGoSearchResults

search_engine = DuckDuckGoSearchResults(
    name="web_search",
    description="Search the web for latest news and current events."
)