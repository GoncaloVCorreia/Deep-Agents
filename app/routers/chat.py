# app/routers/chat.py
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from app.schemas.chat import ChatRequest, StartResponse, DecisionRequest, DecisionResponse
from src.main import chat, decide

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/message", response_model=StartResponse)
async def chat_message(payload: ChatRequest):
    try:
        return chat(payload.message, payload.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decision", response_model=DecisionResponse)
async def chat_decision(payload: DecisionRequest):
    try:
        return decide(payload.thread_id, payload.action_name, payload.decision, payload.args)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
