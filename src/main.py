# src/services/agent.py

from typing import Optional, Dict, Any
from app.config import settings
from src.llms import GoogleGenAILLM
from src.models.agent_model import AgentManager

SESSIONS: Dict[str, Any] = {}

llm = GoogleGenAILLM(model_name="gemini-2.5-flash", temperature=0.0)
agent_manager = AgentManager(llm=llm)

def chat(user_message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    result = agent_manager.chat(user_message, thread_id)
    if result["status"] == "pending_approval":
        # store session state, messages, etc.
        SESSIONS[result["thread_id"]] = result
    return result

def decide(thread_id: str, action_name: str, decision: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    session = SESSIONS.get(thread_id)
    if not session:
        return {"status": "failed", "error": "invalid thread_id"}
    
    config = {"configurable": {"thread_id": thread_id}}
    answer = agent_manager.decide(thread_id, action_name, decision, args, config)

    del SESSIONS[thread_id]
    return answer