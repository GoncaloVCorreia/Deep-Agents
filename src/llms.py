# src/llms.py

from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from app.config import settings

load_dotenv()

class LLM(ABC):
    def __init__(self, model_name: str, temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def invoke(self, messages: list) -> dict:
        pass

class GoogleGenAILLM(LLM):
    def __init__(self, model_name: str, temperature: float = 0.0):
        super().__init__(model_name, temperature)
        api_key = settings.GOOGLE_GENAI_API_KEY
        if not api_key:
            raise RuntimeError("GOOGLE_GENAI_API_KEY not set in environment.")
        self.llm = init_chat_model(
            model_provider="google_genai",
            model=self.model_name,
            api_key=api_key,
            temperature=temperature,
        )

    def invoke(self, messages: list) -> dict:
        return self.llm.invoke({"messages": messages})
