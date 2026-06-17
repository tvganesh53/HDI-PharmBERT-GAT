"""
db_adapter.py — switches between MySQL (local Docker) and SQLite (HF Spaces).

Set environment variable:
    DB_BACKEND=sqlite   → uses /app/data/nlp.db  (HF Spaces default)
    DB_BACKEND=mysql    → uses MYSQL_* env vars   (local Docker default)
"""

from __future__ import annotations
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")

if DB_BACKEND == "mysql":
    MYSQL_USER     = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_HOST     = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT     = os.getenv("MYSQL_PORT", "3306")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "nlp_classifier")
    DATABASE_URL   = (
        f"mysql+aiomysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    )
else:
    # SQLite — stored in /app/data so it persists on HF Spaces persistent storage
    DB_PATH      = os.getenv("SQLITE_PATH", "/app/data/nlp.db")
    DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"


def get_engine():
    if DB_BACKEND == "mysql":
        return create_async_engine(
            DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
    else:
        return create_async_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False,
        )


engine        = get_engine()
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
