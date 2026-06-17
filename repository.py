"""
Phase G — repository.py
All database queries: save classifications, analytics, trends, top-labels.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, func, select, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from models import AnalyticsDaily, APIKeyLog, Classification

log = logging.getLogger("phase_g.repository")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── Classification CRUD ───────────────────────────────────────────────────────

async def save_classification(
    db: AsyncSession,
    *,
    job_id: str,
    input_text: str,
    top_label: str,
    top_score: float,
    all_labels: list[dict],
    latency_ms: float,
    batch_size: int = 1,
    model_name: str | None = None,
    api_key_name: str | None = None,
) -> Classification:
    """Insert a classification result and return the ORM object."""
    row = Classification(
        job_id=job_id,
        input_text=input_text,
        top_label=top_label,
        top_score=top_score,
        all_labels=all_labels,
        latency_ms=latency_ms,
        batch_size=batch_size,
        model_name=model_name,
        api_key_name=api_key_name,
    )
    db.add(row)
    await db.flush()   # get the auto-increment id without committing
    log.debug("Saved classification id=%s label=%s", row.id, top_label)
    return row


async def get_classification(db: AsyncSession, job_id: str) -> Classification | None:
    result = await db.execute(
        select(Classification).where(Classification.job_id == job_id)
    )
    return result.scalar_one_or_none()


async def get_recent_classifications(
    db: AsyncSession, limit: int = 50
) -> list[Classification]:
    result = await db.execute(
        select(Classification).order_by(desc(Classification.created_at)).limit(limit)
    )
    return list(result.scalars().all())


# ── API Key logging ───────────────────────────────────────────────────────────

async def upsert_api_key_log(
    db: AsyncSession,
    *,
    key_name: str,
    role: str = "user",
) -> None:
    """Increment request_count for a key; insert if not yet seen.
    ORM-level upsert — works on SQLite (tests) and MySQL (production).
    """
    result = await db.execute(select(APIKeyLog).where(APIKeyLog.key_name == key_name))
    row = result.scalar_one_or_none()
    if row is None:
        row = APIKeyLog(key_name=key_name, role=role, request_count=1, last_used_at=_utcnow())
        db.add(row)
    else:
        row.request_count = (row.request_count or 0) + 1
        row.last_used_at = _utcnow()
    await db.flush()


# ── Analytics queries ─────────────────────────────────────────────────────────

async def get_summary(db: AsyncSession) -> dict[str, Any]:
    """Overall stats: total requests, avg / p95 latency, top label."""
    total_q = await db.execute(select(func.count()).select_from(Classification))
    total: int = total_q.scalar_one() or 0

    latency_q = await db.execute(
        select(
            func.avg(Classification.latency_ms),
            func.max(Classification.latency_ms),
        )
    )
    avg_lat, max_lat = latency_q.one()

    top_label_q = await db.execute(
        select(Classification.top_label, func.count().label("cnt"))
        .group_by(Classification.top_label)
        .order_by(desc("cnt"))
        .limit(1)
    )
    top_row = top_label_q.first()

    return {
        "total_requests": total,
        "avg_latency_ms": round(float(avg_lat or 0), 2),
        "max_latency_ms": round(float(max_lat or 0), 2),
        "top_label": top_row[0] if top_row else None,
    }


async def get_label_distribution(
    db: AsyncSession, days: int = 30
) -> list[dict[str, Any]]:
    """Label counts for the last N days, sorted by frequency."""
    since = _utcnow() - timedelta(days=days)
    result = await db.execute(
        select(Classification.top_label, func.count().label("count"))
        .where(Classification.created_at >= since)
        .group_by(Classification.top_label)
        .order_by(desc("count"))
    )
    return [{"label": row[0], "count": row[1]} for row in result.all()]


async def get_daily_trends(
    db: AsyncSession, days: int = 30
) -> list[dict[str, Any]]:
    """Daily request counts + average latency for the last N days."""
    since = _utcnow() - timedelta(days=days)
    result = await db.execute(
        select(
            func.date(Classification.created_at).label("day"),
            func.count().label("total"),
            func.avg(Classification.latency_ms).label("avg_latency"),
        )
        .where(Classification.created_at >= since)
        .group_by("day")
        .order_by("day")
    )
    return [
        {
            "day": str(row[0]),
            "total_requests": row[1],
            "avg_latency_ms": round(float(row[2] or 0), 2),
        }
        for row in result.all()
    ]


async def get_top_inputs(
    db: AsyncSession, limit: int = 10
) -> list[dict[str, Any]]:
    """Most frequently classified inputs (exact match deduplication)."""
    result = await db.execute(
        select(Classification.input_text, func.count().label("count"))
        .group_by(Classification.input_text)
        .order_by(desc("count"))
        .limit(limit)
    )
    return [{"input": row[0], "count": row[1]} for row in result.all()]


async def get_latency_percentiles(db: AsyncSession) -> dict[str, float]:
    """p50, p95, p99 latency using MySQL PERCENTILE_CONT approximation."""
    # MySQL 8+ supports window functions; use a portable approximation
    result = await db.execute(
        text(
            """
            SELECT
                AVG(latency_ms)                                       AS p50,
                MAX(CASE WHEN pct <= 0.95 THEN latency_ms END)        AS p95,
                MAX(CASE WHEN pct <= 0.99 THEN latency_ms END)        AS p99
            FROM (
                SELECT latency_ms,
                       PERCENT_RANK() OVER (ORDER BY latency_ms) AS pct
                FROM classifications
            ) ranked
            """
        )
    )
    row = result.one()
    return {
        "p50_ms": round(float(row[0] or 0), 2),
        "p95_ms": round(float(row[1] or 0), 2),
        "p99_ms": round(float(row[2] or 0), 2),
    }


# ── Daily aggregation (called by background task) ────────────────────────────

async def refresh_analytics_daily(db: AsyncSession, target_date: date | None = None) -> None:
    """Compute and upsert an AnalyticsDaily row for `target_date` (default: today).
    Uses ORM-level upsert so it works on both SQLite (tests) and MySQL (production).
    """
    if target_date is None:
        target_date = datetime.now(tz=timezone.utc).date()

    start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    end   = start + timedelta(days=1)

    total_q = await db.execute(
        select(func.count())
        .select_from(Classification)
        .where(Classification.created_at >= start, Classification.created_at < end)
    )
    total = total_q.scalar_one() or 0

    lat_q = await db.execute(
        select(func.avg(Classification.latency_ms))
        .where(Classification.created_at >= start, Classification.created_at < end)
    )
    avg_lat = float(lat_q.scalar_one() or 0)

    top_q = await db.execute(
        select(Classification.top_label, func.count().label("cnt"))
        .where(Classification.created_at >= start, Classification.created_at < end)
        .group_by(Classification.top_label)
        .order_by(desc("cnt"))
        .limit(1)
    )
    top_row = top_q.first()

    keys_q = await db.execute(
        select(func.count(Classification.api_key_name.distinct()))
        .where(Classification.created_at >= start, Classification.created_at < end)
    )
    unique_keys = keys_q.scalar_one() or 0

    # ORM-level upsert: works on SQLite (tests) and MySQL (production)
    existing_q = await db.execute(
        select(AnalyticsDaily).where(AnalyticsDaily.day == target_date)
    )
    row = existing_q.scalar_one_or_none()
    if row is None:
        row = AnalyticsDaily(
            day=target_date,
            total_requests=total,
            avg_latency_ms=round(avg_lat, 2),
            p95_latency_ms=0.0,
            top_label=top_row[0] if top_row else None,
            error_count=0,
            unique_keys=unique_keys,
        )
        db.add(row)
    else:
        row.total_requests = total
        row.avg_latency_ms = round(avg_lat, 2)
        row.top_label = top_row[0] if top_row else None
        row.unique_keys = unique_keys

    await db.flush()
    log.debug("analytics_daily refreshed for %s (total=%d)", target_date, total)
