# app/routers/pcs.py
from fastapi import APIRouter, HTTPException
from app.schemas.pcs import PreviewRequest, PreviewResponse

# Option A: call the deterministic coder directly (recommended)
from src.procedures.coder import code_note

# Option B: call the LangChain tool wrapper instead
# from src.tools import get_pcs_codes
# import json

router = APIRouter(prefix="/pcs", tags=["pcs"])

@router.post("/preview", response_model=PreviewResponse)
async def preview(req: PreviewRequest):
    try:
        # Option A
        result = code_note(req.note)

        # Option B (uncomment if you prefer to route through the LangChain Tool)
        # raw = get_pcs_codes.invoke({"note": req.note})
        # result = json.loads(raw)

        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
