# 📄 AI Agent-App

A **Retrieval-Augmented Generation (RAG)** API built with **FastAPI**, **LangGraph**, and **Google Gemini**. Upload a PDF and ask questions about it — the agent automatically retrieves relevant context from the document and generates accurate, grounded answers.

---

## ✨ Features

- 📤 **PDF Upload & Indexing** — Upload any PDF and it gets chunked, embedded, and stored in a FAISS vectorstore automatically.
- ⚡ **Smart Caching** — Vectorstores are fingerprinted by file hash + chunking config. Repeated uploads skip re-indexing via in-memory and on-disk caching.
- 🤖 **Agentic RAG with LangGraph** — Uses a ReAct-style LangGraph agent that decides when to call the retrieval tool and when to answer directly.
- 🔍 **Semantic Search** — Similarity search powered by Google's `gemini-embedding-001` embedding model and FAISS.
- 🧠 **Gemini 2.5 Flash LLM** — Fast, capable responses using `gemini-2.5-flash` via `langchain-google-genai`.
- 📊 **LangSmith Tracing** — End-to-end observability of every node, tool call, and LLM invocation.
- 🗄️ **PostgreSQL Ready** — Async Postgres for Checkpointer used to Memory.

