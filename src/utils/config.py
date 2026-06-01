import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Config(BaseModel):
    groq_api_key: str
    gemini_api_key: str
    judge_model: str = "llama-3.3-70b-versatile"
    pipeline_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens: int = 1024

def load_config() -> Config:
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not groq_key:
        raise ValueError("GROQ_API_KEY not found in .env file")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    return Config(
        groq_api_key=groq_key,
        gemini_api_key=gemini_key
    )