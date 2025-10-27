# app/services/agent.py
import json
import uuid
from typing import Dict, Any, Literal, Callable, Optional
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from app.config import settings


load_dotenv(override=True)
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
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

SYSTEM_PROMPT = """You are an expert researcher.
You have access to an internet search tool as your primary means of gathering information.

## `internet_search`
Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""
SESSIONS: Dict[str, Dict[str, Any]] = {}
checkpointer = MemorySaver()  # required for human-in-the-loop

def get_agent():
    llm = init_chat_model(
        "gemini-2.5-flash",
        model_provider="google_genai",
        api_key=settings.GOOGLE_GENAI_API_KEY,
    )
    
    agent = create_deep_agent(
        tools=[internet_search, get_weather],
        interrupt_on={
            "get_weather": {"allowed_decisions": ["approve", "edit", "reject"]}
        },
        system_prompt=SYSTEM_PROMPT,
        model=llm,
        checkpointer=checkpointer,
    )
    return agent

def chat(user_message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    agent = get_agent()

    if not thread_id:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config=config)

    if isinstance(result, dict) and result.get("__interrupt__"):
        interrupts = result["__interrupt__"][0].value
        action_requests = interrupts["action_requests"]
        # Store session (or update it)
        SESSIONS[thread_id] = {
            "thread_config": config,
            "messages": result.get("messages", [{"role": "user", "content": user_message}])
        }
        return {
            "status": "pending_approval",
            "thread_id": thread_id,
            "actions": action_requests
        }

    # Completed without interruption
    answer = None
    if hasattr(result, "content"):
        answer = result.content
    elif isinstance(result, dict) and "messages" in result:
        msgs = result["messages"]
        if msgs:
            last = msgs[-1]
            # if last is an object with attribute `content`
            if hasattr(last, "content"):
                answer = last.content
            # else if last is a dict
            elif isinstance(last, dict):
                answer = last.get("content")
    return {
        "status": "completed",
        "answer": answer or str(result),
        "thread_id": thread_id
    }

def decide(thread_id: str, action_name: str, decision: Literal["approve","edit","reject"], args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    session = SESSIONS.get(thread_id)
    if not session:
        return {"status": "failed", "error": "invalid thread_id"}
    
    config = session["thread_config"]
    
    if decision == "edit":
        decision_obj = {
            "type": "edit",
            "edited_action": {"name": action_name, "args": args or {}}
        }
    else:
        decision_obj = {"type": decision}
    
    # Use Command alone - it should resume the interrupted state
    result = get_agent().invoke(
        Command(resume={"decisions": [decision_obj]}),
        config=config
    )
    
    # Clean up session
    del SESSIONS[thread_id]
    
    # Extract answer
    answer = None
    if isinstance(result, dict) and "messages" in result:
        msgs = result["messages"]
        if msgs and len(msgs) > 0:
            last = msgs[-1]
            if hasattr(last, "content"):
                answer = last.content
            elif isinstance(last, dict):
                answer = last.get("content")
    
    return {"status": "completed", "answer": answer or str(result)}