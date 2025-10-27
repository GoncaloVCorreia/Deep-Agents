# app/schemas/chat.py
from datetime import datetime
from typing import Dict, Any, List, Union
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class ConversationResponse(BaseModel):
    messages: Union[List[Dict[str, Any]], Dict[str, Any]]
    created_at: datetime

    model_config = {"from_attributes": True}
