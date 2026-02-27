import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# ─────────────────────────────────────────
# LLM INIT
# ─────────────────────────────────────────
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=os.getenv("GOOGLE_API_KEY"),
        thinking_budget=0,  # disables thinking mode → removes signature from output
    )


def get_embedding():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
