# src/models/agent_model.py

import uuid
from typing import Dict, Any, Literal, Optional
from deepagents import create_deep_agent
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from src.llms import GoogleGenAILLM
from src.tools import internet_search, get_weather, icd10_query
from src.prompts import SUPERVISOR_PROMPT, DIAGNOSIS_PROMPT, EVAL_PROMPT

def coerce_text(v: Any) -> str:
    """Flatten common LangChain/LangGraph/Gemini shapes into a plain string."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v

    # Objects with `.content`
    if hasattr(v, "content"):
        return coerce_text(getattr(v, "content"))

    # Gemini/LC parts list
    if isinstance(v, list):
        out = []
        for p in v:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict):
                # {'type':'text','text':'...'} or {'text':'...'}
                if isinstance(p.get("text"), str):
                    out.append(p["text"])
                elif "content" in p:
                    out.append(coerce_text(p["content"]))
                elif "parts" in p:
                    out.append(coerce_text(p["parts"]))
            elif hasattr(p, "content"):
                out.append(coerce_text(p.content))
            else:
                out.append(str(p))
        return "\n".join(s for s in out if s)

    # Dict-shaped results
    if isinstance(v, dict):
        # Try common keys first
        for k in ("answer", "output", "text", "content"):
            if k in v:
                return coerce_text(v[k])
        # Gemini candidates → candidates[0].content.parts
        cands = v.get("candidates")
        if isinstance(cands, list) and cands:
            return coerce_text(cands[0].get("content"))
        return str(v)

    return str(v)


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
                "tools": [icd10_query],
                "model": self.llm.llm  # uses same LLM by default
            },
          

        ]
        agent = create_deep_agent(
            tools=[get_weather],
            interrupt_on={
            "get_weather": {"allowed_decisions": ["approve", "edit", "reject"]}
             },
            system_prompt=EVAL_PROMPT,
            model=self.llm.llm,
            checkpointer=self.checkpointer,
            subagents=subagents
        )
        return agent

    def _build_diagnosis_agent(self):
        # Optionally build a separate deep agent for diagnosis if you want isolation
        agent = create_deep_agent(
            tools=[icd10_query],
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
        # Interrupt path
        if isinstance(result, dict) and result.get("__interrupt__"):
            interrupts = result["__interrupt__"][0].value
            return {
                "status": "pending_approval",
                "thread_id": thread_id,
                "actions": interrupts.get("action_requests", []),
            }

        # Completed path – extract last message content if present
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

        # ALWAYS normalize to a string
        answer_str = coerce_text(answer if answer is not None else result)

        return {
            "status": "completed",
            "answer": answer_str,
            "thread_id": thread_id,
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
