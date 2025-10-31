import uvicorn
from app.routers.chat import router as chat_router
from app.routers.pcs import router as pcs_router
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings 
from fastapi import FastAPI
from pydantic import BaseModel



class CodeRequest(BaseModel):
    message: str

app = FastAPI(title="ICD-10 Deep Agent")


# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)


# Include routers
app.include_router(chat_router)
app.include_router(pcs_router)   


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)