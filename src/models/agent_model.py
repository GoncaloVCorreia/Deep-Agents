# src/models/agent_model.py

import uuid
from typing import Dict, Any, Literal, Optional
from deepagents import create_deep_agent
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from src.llms import GoogleGenAILLM
from src.tools import internet_search, get_weather, get_code
from src.prompts import SUPERVISOR_PROMPT, DIAGNOSIS_PROMPT

class AgentManager:
    def __init__(self, llm: GoogleGenAILLM):
        self.llm = llm
        self.checkpointer = MemorySaver()
        self.supervisor_agent = self._build_supervisor_agent()
        self.diagnosis_agent = self._build_diagnosis_agent()

    def _build_supervisor_agent(self):
        subagents = [
            {
                "name": "diagnosis_agent",
                "description": "Handles ICD-10 code lookups for conditions",
                "system_prompt": DIAGNOSIS_PROMPT,
                "tools": [get_code],
                "model": self.llm.llm  # uses same LLM by default
            }
        ]
        agent = create_deep_agent(
            tools=[internet_search, get_weather],
            interrupt_on={
            "get_weather": {"allowed_decisions": ["approve", "edit", "reject"]}
             },
            system_prompt=SUPERVISOR_PROMPT,
            model=self.llm.llm,
            checkpointer=self.checkpointer,
            subagents=subagents
        )
        return agent

    def _build_diagnosis_agent(self):
        # Optionally build a separate deep agent for diagnosis if you want isolation
        agent = create_deep_agent(
            tools=[get_code],
            system_prompt=DIAGNOSIS_PROMPT,
            model=self.llm.llm,
            checkpointer=self.checkpointer
        )
        return agent

    def chat(self, user_message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = self.supervisor_agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config
        )
        return self._handle_result(result, thread_id, config)

    def _handle_result(self, result: Any, thread_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        
       
        if isinstance(result, dict) and result.get("__interrupt__"):
            interrupts = result["__interrupt__"][0].value
            action_requests = interrupts["action_requests"]
            
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

    def decide(self, thread_id: str, action_name: str, decision: Literal["approve","edit","reject"], args: Optional[Dict[str, Any]],config) -> Dict[str, Any]:
        decision_obj = {"type": decision}
        if decision == "edit":
            decision_obj = {
                "type": "edit",
                "edited_action": {"name": action_name, "args": args or {}}
            }
        result = self.supervisor_agent.invoke(
            Command(resume={"decisions": [decision_obj]}),
            config=config
        )
        answer = None
        if hasattr(result, "content"):
            answer = result.content
        elif isinstance(result, dict) and "messages" in result:
            msgs = result["messages"]
            if msgs:
                last = msgs[-1]
                if hasattr(last, "content"):
                    answer = last.content
                elif isinstance(last, dict):
                    answer = last.get("content")

        return {"status": "completed", "thread_id": thread_id, "answer": answer or str(result)}
