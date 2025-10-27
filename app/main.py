from fastapi import FastAPI
import uvicorn
from app.routers.chat import router as chat_router
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings  # or wherever your Settings class is



app = FastAPI(title="Digital Twin API")

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


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)