# src/tools.py

import os
from typing import Literal
from tavily import TavilyClient
from langchain_core.tools import tool
from app.config import settings

# Initialize Tavily client once
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search."""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

@tool
def get_code(condition: str) -> str:
    """Diagnosis agent tool: return ICD-10 code for a given condition."""
    # implement actual lookup logic here
    return f"CODE_FOR_{condition.replace(' ', '_').upper() } is 0"
