# app/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Deep Agents API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Observability (optional)
    LANGSMITH_TRACING_V2: Optional[str] = None
    LANGSMITH_PROJECT: Optional[str] = None
    LANGSMITH_API_KEY: Optional[str] = None

    # Keys
    GOOGLE_GENAI_API_KEY: str
    TAVILY_API_KEY: str

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    # Flags
    ENABLE_BACKGROUND_TASKS: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()
