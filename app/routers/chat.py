# app/routers/chat.py
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from app.schemas.chat import ChatRequest, ConversationResponse
from src.main import generate_reply

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/message", response_model=ConversationResponse)
async def chat_message(payload: ChatRequest):
    try:
        reply = await generate_reply(payload.message)
        return ConversationResponse(
            messages=[
                {"role": "user", "content": payload.message},
                {"role": "assistant", "content": reply},
            ],
            created_at=datetime.now(timezone.utc),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

