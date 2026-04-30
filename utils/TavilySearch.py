import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langsmith import traceable
load_dotenv()

class TavilySearch:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set")
        self.client = TavilyClient(api_key=api_key)
    
    @traceable(name="Tavily Web Search")
    def search(self, query):
        response = self.client.search(
            query=query,
            search_depth="advanced",
            max_results=3,
            include_images=False,
            include_raw_content=False
        )
        return response["results"]
