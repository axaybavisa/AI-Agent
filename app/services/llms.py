from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from functools import lru_cache

from app.core.config import get_settings

settings = get_settings()

# ─────────────────────────────────────────
# LLM INIT
# ─────────────────────────────────────────
@lru_cache(10)
def get_llm():
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.7,
    )


def get_embedding():
    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )
