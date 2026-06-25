"""
Phase G - database.py
Auto-switches between MySQL (local Docker) and SQLite (HF Spaces).
Set DB_BACKEND=sqlite for HF Spaces, DB_BACKEND=mysql for local Docker.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

log = logging.getLogger("phase_g.database")

DB_BACKEND = os.getenv("DB_BACKEND", "mysql")

if DB_BACKEND == "sqlite":
    SQLITE_PATH  = os.getenv("SQLITE_PATH", "/app/data/nlp.db")
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    DATABASE_URL = f"sqlite+aiosqlite:///{SQLITE_PATH}"
    engine = create_async_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=os.getenv("DB_ECHO", "false").lower() == "true",
    )
else:
    from urllib.parse import quote_plus
    MYSQL_USER     = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_HOST     = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT     = os.getenv("MYSQL_PORT", "3306")
    MYSQL_DB       = os.getenv("MYSQL_DB", os.getenv("MYSQL_DATABASE", "nlp_classifier"))
    DATABASE_URL = (
        f"mysql+aiomysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        f"?charset=utf8mb4"
    )
    engine = create_async_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=os.getenv("DB_ECHO", "false").lower() == "true",
    )

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(lambda conn: Base.metadata.create_all(conn, checkfirst=True))
    log.info("Database tables ready")


async def close_db() -> None:
    await engine.dispose()
    log.info("Database pool closed.")

# rebuild 1781059258.3590245
