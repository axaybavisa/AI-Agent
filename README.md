# 📄 RAG-App

A **Retrieval-Augmented Generation (RAG)** API built with **FastAPI**, **LangGraph**, and **Google Gemini**. Upload a PDF and ask questions about it — the agent automatically retrieves relevant context from the document and generates accurate, grounded answers.

---

## ✨ Features

- 📤 **PDF Upload & Indexing** — Upload any PDF and it gets chunked, embedded, and stored in a FAISS vectorstore automatically.
- ⚡ **Smart Caching** — Vectorstores are fingerprinted by file hash + chunking config. Repeated uploads skip re-indexing via in-memory and on-disk caching.
- 🤖 **Agentic RAG with LangGraph** — Uses a ReAct-style LangGraph agent that decides when to call the retrieval tool and when to answer directly.
- 🔍 **Semantic Search** — Similarity search powered by Google's `gemini-embedding-001` embedding model and FAISS.
- 🧠 **Gemini 2.5 Flash LLM** — Fast, capable responses using `gemini-2.5-flash` via `langchain-google-genai`.
- 📊 **LangSmith Tracing** — End-to-end observability of every node, tool call, and LLM invocation.
- 🗄️ **PostgreSQL Ready** — Async SQLAlchemy setup included for future persistence features.

---

## 🏗️ Architecture

```
POST /chat (PDF + message)
        │
        ▼
  build_graph()
        │
        ▼
┌───────────────────────────────────┐
│         LangGraph Agent           │
│                                   │
│  ┌─────────┐     ┌─────────────┐  │
│  │  LLM    │────▶│  rag_tool   │  │
│  │  Node   │◀────│  (FAISS     │  │
│  │(Gemini) │     │  retriever) │  │
│  └─────────┘     └─────────────┘  │
└───────────────────────────────────┘
        │
        ▼
   Final Answer
```

**Flow:**
1. PDF is uploaded → saved to a temp file → loaded with `PyPDFLoader`
2. Document is split into chunks with `RecursiveCharacterTextSplitter` (500 tokens, 100 overlap)
3. Chunks are embedded and stored in a FAISS vectorstore (cached to disk under `faiss_index/`)
4. A LangGraph `StateGraph` is compiled with two nodes: `llm` and `tools`
5. The Gemini LLM decides whether to call `rag_tool` for retrieval or respond directly
6. The agent loops until it produces a final answer

---

## 📁 Project Structure

```
RAG-App/
├── main.py                      # FastAPI app & /chat endpoint
├── app/
│   ├── graph/
│   │   ├── graph_builder.py     # LangGraph StateGraph construction
│   │   └── state.py             # ChatState (Pydantic + message history)
│   ├── nodes/
│   │   └── tool.py              # PDF loading, vectorstore, RAG tool, LLM node
│   └── services/
│       ├── models.py            # SQLAlchemy models (extendable)
│       └── postgres.py          # Async DB engine & session factory
├── faiss_index/                 # Persisted FAISS vectorstores (auto-created)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started


## ⚙️ Configuration

Key parameters in `app/nodes/tool.py` can be tuned:

| Parameter         | Default                        | Description                            |
|-------------------|--------------------------------|----------------------------------------|
| `chunk_size`      | `500`                          | Token size of each document chunk      |
| `chunk_overlap`   | `100`                          | Overlap between consecutive chunks     |
| `embedding_model` | `models/gemini-embedding-001`  | Google embedding model                 |
| `k` (retriever)   | `4`                            | Number of chunks retrieved per query   |
| LLM model         | `gemini-2.5-flash`             | Google Gemini model used for answers   |
| `temperature`     | `0.7`                          | LLM response creativity                |

---

## 🧰 Tech Stack

| Layer           | Technology                              |
|-----------------|-----------------------------------------|
| API Framework   | FastAPI                                 |
| Agent Orchestration | LangGraph                           |
| LLM             | Google Gemini 2.5 Flash                 |
| Embeddings      | Google Generative AI Embeddings         |
| Vector Store    | FAISS (CPU)                             |
| PDF Parsing     | PyPDFLoader (pypdf)                     |
| Observability   | LangSmith                               |
| Database        | PostgreSQL + SQLAlchemy (async)         |
| Runtime         | Python 3.11+, uvicorn                   |

---

