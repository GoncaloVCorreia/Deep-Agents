# app/services/agent.py
import functools
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from app.config import settings

load_dotenv(override=True)  # loads .env once

# Tavily search tool
def _make_internet_search(tavily_client: TavilyClient):
    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ):
        """Run a web search"""
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    return internet_search

SYSTEM_PROMPT = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.
You have access to an internet search tool as your primary means of gathering information.

## `internet_search`
Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

def get_agent():

    llm = init_chat_model(
        "gemini-2.5-flash",
        model_provider="google_genai",
        api_key=settings.GOOGLE_GENAI_API_KEY,
    )

    tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    internet_search = _make_internet_search(tavily_client)

    agent = create_deep_agent(
        tools=[internet_search],
        system_prompt=SYSTEM_PROMPT,
        model=llm,  
    )
    return agent

async def generate_reply(user_message: str) -> str:
    agent = get_agent()
    # DeepAgents expects a messages list
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    # Result shape can vary by stack; stringify is safest, but try common fields first
    try:
        # if it's a LangChain AIMessage-like
        content = getattr(result, "content", None)
        if content:
            return content
        # if it's a dict with messages
        if isinstance(result, dict) and "messages" in result:
            # last message as reply
            msgs = result["messages"]
            if isinstance(msgs, list) and msgs:
                return str(msgs[-1].get("content") or msgs[-1])
        return str(result)
    except Exception:
        return str(result)
