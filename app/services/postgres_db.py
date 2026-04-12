import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import get_settings

load_dotenv()

settings = get_settings()


def _to_psycopg_url(url: str) -> str:
    """
    AsyncPostgresSaver uses psycopg directly and expects a plain
    'postgresql://' URL, not a SQLAlchemy driver URL like
    'postgresql+asyncpg://'. Strip the driver suffix if present.
    """
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


@asynccontextmanager
async def get_checkpointer():
    """Async context manager that yields a ready AsyncPostgresSaver."""
    conn_string = _to_psycopg_url(settings.DATABASE_URL)
    async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()
        yield checkpointer


# ── one-off setup script ──────────────────────────────────────────────────────
async def _setup():
    async with get_checkpointer():
        print("Checkpointer tables created successfully.")

if __name__ == "__main__":
    asyncio.run(_setup())
