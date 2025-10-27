# app/schemas/chat.py
from datetime import datetime
from typing import Dict, Any, List, Union, Literal, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class StartResponse(BaseModel):
    status: Literal["completed", "pending_approval"]
    answer: Optional[str] = None
    thread_id: Optional[str] = None
    actions: Optional[List[Dict[str, Any]]] = None

class DecisionRequest(BaseModel):
    thread_id: str
    action_name: str
    decision: Literal["approve", "edit", "reject"]
    args: Optional[Dict[str, Any]] = None

class DecisionResponse(BaseModel):
    status: Literal["completed", "failed"]
    answer: Optional[str] = None
    error: Optional[str] = None

class ConversationResponse(BaseModel):
    messages: Union[List[Dict[str, Any]], Dict[str, Any]]
    created_at: datetime

    model_config = {"from_attributes": True}
