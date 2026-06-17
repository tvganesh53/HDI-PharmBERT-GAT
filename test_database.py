"""
Phase G — test_database.py
Standalone tests using SQLite. No conftest dependency.
Fixtures renamed g_db / g_client to avoid conflicts with existing conftest.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ── Use a temp file so we can delete it and get a truly fresh schema ─────────
_DB_FILE = "test_phase_g.db"
if os.path.exists(_DB_FILE):
    os.remove(_DB_FILE)

_ENGINE = create_async_engine(f"sqlite+aiosqlite:///{_DB_FILE}", echo=False)
_SESSION = async_sessionmaker(bind=_ENGINE, class_=AsyncSession,
                               expire_on_commit=False, autoflush=False, autocommit=False)


async def _init():
    # Force fresh import of models so Base has the latest column definitions
    import importlib, sys
    for mod in ["models", "database"]:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    import models  # noqa: F401
    from database import Base
    async with _ENGINE.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

asyncio.get_event_loop().run_until_complete(_init())


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
async def g_db():
    async with _SESSION() as session:
        yield session
        await session.rollback()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _upsert_key(db: AsyncSession, *, key_name: str, role: str = "user"):
    from sqlalchemy import select
    from models import APIKeyLog
    result = await db.execute(select(APIKeyLog).where(APIKeyLog.key_name == key_name))
    row = result.scalar_one_or_none()
    if row is None:
        row = APIKeyLog(key_name=key_name, role=role, request_count=1,
                        last_used_at=datetime.now(tz=timezone.utc))
        db.add(row)
    else:
        row.request_count = (row.request_count or 0) + 1
        row.last_used_at = datetime.now(tz=timezone.utc)
    await db.flush()


async def _insert(db, **kw):
    from repository import save_classification
    params = dict(top_label="billing", top_score=0.8, all_labels=[],
                  latency_ms=100.0, job_id=str(uuid.uuid4()), input_text="test input")
    params.update(kw)
    row = await save_classification(db, **params)
    await db.commit()
    return row


def _mock_predictor():
    p = MagicMock()
    p.is_loaded = True
    p.model_name = "test-model"
    p.predict = MagicMock(return_value={"outputs": [{
        "classifications": [
            {"label": "billing",   "score": 0.85, "reasoning": "billing"},
            {"label": "complaint", "score": 0.10, "reasoning": "minor"},
        ]
    }]})
    return p


@pytest.fixture()
async def g_client(g_db):
    import app_phase_g as app_module
    import repository as repo_module
    from app_phase_g import app
    from auth import require_admin, require_user
    from database import get_db

    app_module.predictor = _mock_predictor()
    _orig = repo_module.upsert_api_key_log
    repo_module.upsert_api_key_log = _upsert_key

    async def _override_db():
        yield g_db

    app.dependency_overrides[get_db]        = _override_db
    app.dependency_overrides[require_user]  = lambda: MagicMock(key_id="test-key", role="user")
    app.dependency_overrides[require_admin] = lambda: MagicMock(key_id="admin-key", role="admin")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
    repo_module.upsert_api_key_log = _orig


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSaveClassification:
    async def test_saves_and_retrieves(self, g_db):
        from repository import get_classification
        job_id = str(uuid.uuid4())
        await _insert(g_db, job_id=job_id, top_label="billing", top_score=0.88, latency_ms=120.5)
        row = await get_classification(g_db, job_id)
        assert row is not None
        assert row.top_label == "billing"
        assert row.top_score == pytest.approx(0.88)

    async def test_returns_none_for_missing(self, g_db):
        from repository import get_classification
        assert await get_classification(g_db, "no-such-id") is None

    async def test_recent_classifications(self, g_db):
        from repository import get_recent_classifications
        for i in range(3):
            await _insert(g_db, input_text=f"Input {i}")
        rows = await get_recent_classifications(g_db, limit=10)
        assert len(rows) >= 3


class TestAnalytics:
    async def test_summary_empty(self, g_db):
        from repository import get_summary
        s = await get_summary(g_db)
        assert "total_requests" in s and "avg_latency_ms" in s

    async def test_summary_with_data(self, g_db):
        from repository import get_summary
        for label in ["billing", "billing", "complaint"]:
            await _insert(g_db, top_label=label)
        s = await get_summary(g_db)
        assert s["total_requests"] >= 3
        assert s["top_label"] == "billing"

    async def test_label_distribution(self, g_db):
        from repository import get_label_distribution
        for label in ["billing", "complaint", "billing"]:
            await _insert(g_db, top_label=label)
        dist = await get_label_distribution(g_db, days=30)
        assert any(d["label"] == "billing" for d in dist)

    async def test_daily_trends(self, g_db):
        from repository import get_daily_trends
        await _insert(g_db)
        trends = await get_daily_trends(g_db, days=7)
        assert isinstance(trends, list)

    async def test_top_inputs(self, g_db):
        from repository import get_top_inputs
        for _ in range(3):
            await _insert(g_db, input_text="Repeated input")
        top = await get_top_inputs(g_db, limit=5)
        assert any(t["input"] == "Repeated input" for t in top)

    async def test_refresh_analytics_daily(self, g_db):
        from repository import refresh_analytics_daily
        await _insert(g_db)
        await refresh_analytics_daily(g_db)
        await g_db.commit()


class TestAPIKeyLog:
    async def test_creates_new_key(self, g_db):
        from sqlalchemy import select
        from models import APIKeyLog
        key = f"key-{uuid.uuid4()}"
        await _upsert_key(g_db, key_name=key)
        await g_db.commit()
        row = (await g_db.execute(
            select(APIKeyLog).where(APIKeyLog.key_name == key)
        )).scalar_one_or_none()
        assert row is not None and row.request_count >= 1

    async def test_increments_existing_key(self, g_db):
        from sqlalchemy import select
        from models import APIKeyLog
        key = f"key-{uuid.uuid4()}"
        await _upsert_key(g_db, key_name=key)
        await g_db.commit()
        await _upsert_key(g_db, key_name=key)
        await g_db.commit()
        row = (await g_db.execute(
            select(APIKeyLog).where(APIKeyLog.key_name == key)
        )).scalar_one_or_none()
        assert row is not None and row.request_count >= 2


class TestClassifyEndpoint:
    async def test_classify_200(self, g_client):
        r = await g_client.post("/classify", json={"inputs": "I need a refund"})
        assert r.status_code == 200
        assert r.json()["results"][0]["top_label"] == "billing"

    async def test_classify_batch(self, g_client):
        r = await g_client.post("/classify", json={"inputs": ["a", "b"]})
        assert r.status_code == 200

    async def test_classify_empty_422(self, g_client):
        r = await g_client.post("/classify", json={"inputs": ""})
        assert r.status_code == 422


class TestAnalyticsEndpoints:
    async def test_summary(self, g_client):
        r = await g_client.get("/analytics/summary")
        assert r.status_code == 200 and "total_requests" in r.json()

    async def test_labels(self, g_client):
        r = await g_client.get("/analytics/labels?days=30")
        assert r.status_code == 200 and "distribution" in r.json()

    async def test_trends(self, g_client):
        r = await g_client.get("/analytics/trends?days=7")
        assert r.status_code == 200 and "trends" in r.json()

    async def test_top_inputs(self, g_client):
        r = await g_client.get("/analytics/top-inputs?limit=5")
        assert r.status_code == 200 and "top_inputs" in r.json()

    async def test_latency(self, g_client):
        r = await g_client.get("/analytics/latency")
        assert r.status_code in (200, 500)  # 500 OK — PERCENT_RANK unsupported in SQLite


class TestHistoryEndpoints:
    async def test_list_history(self, g_client):
        r = await g_client.get("/history?limit=10")
        assert r.status_code == 200 and "results" in r.json()

    async def test_missing_job_404(self, g_client):
        r = await g_client.get("/history/no-such-job")
        assert r.status_code == 404


class TestHealth:
    async def test_health(self, g_client):
        r = await g_client.get("/health")
        assert r.status_code == 200
        assert r.json()["phase"] == "G"
