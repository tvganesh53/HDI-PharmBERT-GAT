"""
Phase G — models.py (SQLite + MySQL compatible)
Primary keys use Integer (not BigInteger) so SQLite AUTOINCREMENT works correctly.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Float, Integer, String, Text, TypeDecorator
from sqlalchemy.orm import mapped_column

from database import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class JSONText(TypeDecorator):
    """JSON stored as TEXT — works on SQLite and MySQL."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return json.dumps(value if value is not None else [])

    def process_result_value(self, value, dialect):
        if not value:
            return []
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return []


class Classification(Base):
    __tablename__ = "classifications"

    # Integer (not BigInteger) — required for SQLite AUTOINCREMENT
    id           = mapped_column(Integer,     primary_key=True, autoincrement=True)
    job_id       = mapped_column(String(64),  unique=True, index=True, nullable=False, default=_new_uuid)
    input_text   = mapped_column(Text,        nullable=False)
    top_label    = mapped_column(String(128), nullable=False, index=True)
    top_score    = mapped_column(Float,       nullable=False)
    all_labels   = mapped_column(JSONText,    nullable=False, default=list)
    latency_ms   = mapped_column(Float,       nullable=False)
    batch_size   = mapped_column(Integer,     nullable=False, default=1)
    model_name   = mapped_column(String(128), nullable=True)
    api_key_name = mapped_column(String(128), nullable=True, index=True)
    created_at   = mapped_column(DateTime,    nullable=False, default=_utcnow, index=True)


class APIKeyLog(Base):
    __tablename__ = "api_keys_log"

    id            = mapped_column(Integer,     primary_key=True, autoincrement=True)
    key_name      = mapped_column(String(128), unique=True, index=True, nullable=False)
    role          = mapped_column(String(32),  nullable=False, default="user")
    request_count = mapped_column(Integer,     nullable=False, default=0)
    last_used_at  = mapped_column(DateTime,    nullable=True)
    created_at    = mapped_column(DateTime,    nullable=False, default=_utcnow)


class AnalyticsDaily(Base):
    __tablename__ = "analytics_daily"

    id             = mapped_column(Integer,      primary_key=True, autoincrement=True)
    day            = mapped_column(Date,         unique=True, index=True, nullable=False)
    total_requests = mapped_column(Integer,      nullable=False, default=0)
    avg_latency_ms = mapped_column(Float,        nullable=False, default=0.0)
    p95_latency_ms = mapped_column(Float,        nullable=False, default=0.0)
    top_label      = mapped_column(String(128),  nullable=True)
    error_count    = mapped_column(Integer,      nullable=False, default=0)
    unique_keys    = mapped_column(Integer,      nullable=False, default=0)
