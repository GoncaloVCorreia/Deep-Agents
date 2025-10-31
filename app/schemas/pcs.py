# app/schemas/pcs.py
from typing import Dict, Any, Optional
from pydantic import BaseModel

class PreviewRequest(BaseModel):
    note: str

class PreviewResponse(BaseModel):
    result: Dict[str, Any]
