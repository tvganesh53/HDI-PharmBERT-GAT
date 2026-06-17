"""
Phase G — migrate.py
Standalone script to create all MySQL tables.
Run once before starting the server for the first time.

Usage:
    python migrate.py
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("migrate")


async def main() -> None:
    from database import create_tables, engine

    log.info("Connecting to MySQL …")
    log.info(
        "Host=%s  DB=%s",
        os.getenv("MYSQL_HOST", "localhost"),
        os.getenv("MYSQL_DB", "nlp_classifier"),
    )
    await create_tables()

    # Print table names that were created
    from sqlalchemy import inspect, text
    async with engine.connect() as conn:
        result = await conn.execute(text("SHOW TABLES"))
        tables = [row[0] for row in result.fetchall()]

    log.info("Tables in database: %s", tables)
    log.info("Migration complete ✓")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
