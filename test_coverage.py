"""
test_coverage.py â€” Full coverage test suite for Phase F.

Covers all modules: pipeline, batcher, job_queue, predictor,
metrics, logger, auth, rate_limiter, api_keys, and API endpoints.

Run with coverage:
    pytest test_coverage.py -v --cov=. --cov-report=html
"""

from __future__ import annotations
import json
import time
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from conftest import make_mock_groq_response, DEFAULT_LABELS


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# pipeline.py coverage
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
from pipeline import NLPPipeline, PipelineError

@pytest.mark.unit
class TestPipelineCoverage:
    @pytest.fixture
    def pipe(self): return NLPPipeline(max_length=200, max_batch=5)

    def test_string_input(self, pipe):             assert pipe.process("hello") == ["hello"]
    def test_list_input(self, pipe):               assert len(pipe.process(["a", "b"])) == 2
    def test_dict_text_key(self, pipe):            assert pipe.process({"text": "hi"}) == ["hi"]
    def test_dict_content_key(self, pipe):         assert pipe.process({"content": "hi"}) == ["hi"]
    def test_dict_input_key(self, pipe):           assert pipe.process({"input": "hi"}) == ["hi"]
    def test_html_stripped(self, pipe):            assert "<b>" not in pipe.process("<b>bold</b>")[0]
    def test_html_entity(self, pipe):              assert pipe.process("AT&amp;T")[0] == "AT&T"
    def test_whitespace_collapsed(self, pipe):     assert pipe.process("a   b")[0] == "a b"
    def test_truncation(self, pipe):               assert len(pipe.process("x" * 300)[0]) == 200
    def test_unicode_normalised(self, pipe):       assert pipe.process("\ufb01le")[0] == "file"
    def test_empty_string_raises(self, pipe):
        with pytest.raises(PipelineError): pipe.process("")
    def test_whitespace_only_raises(self, pipe):
        with pytest.raises(PipelineError): pipe.process("   ")
    def test_none_raises(self, pipe):
        with pytest.raises(PipelineError): pipe.process(None)
    def test_empty_list_raises(self, pipe):
        with pytest.raises(PipelineError): pipe.process([])
    def test_batch_too_large_raises(self, pipe):
        with pytest.raises(PipelineError): pipe.process(["a"] * 6)
    def test_non_string_in_list_raises(self, pipe):
        with pytest.raises(PipelineError): pipe.process(["hello", 123])
    def test_unsupported_type_raises(self, pipe):
        with pytest.raises(PipelineError): pipe.process(42)
    def test_max_length_param(self, pipe):
        assert len(pipe.process("x" * 300, max_length=50)[0]) == 50
    def test_dict_missing_text_raises(self, pipe):
        with pytest.raises(PipelineError): pipe.process({"other": "value"})
    def test_newline_collapse(self, pipe):
        result = pipe.process("a\n\n\n\nb")
        assert "\n\n\n" not in result[0]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# job_queue.py coverage
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
from job_queue import JobQueue, QueueFullError
from schemas import Job, JobStatus

@pytest.mark.asyncio
@pytest.mark.unit
class TestJobQueueCoverage:
    async def test_submit_and_get(self):
        q = JobQueue()
        job = Job(inputs="hello")
        await q.submit(job)
        stored = await q.get(job.job_id)
        assert stored.job_id == job.job_id

    async def test_depth_increments(self):
        q = JobQueue()
        assert await q.depth() == 0
        await q.submit(Job(inputs="a"))
        assert await q.depth() == 1

    async def test_update_status(self):
        q = JobQueue()
        job = Job(inputs="test")
        await q.submit(job)
        job.status = JobStatus.DONE
        job.outputs = {"result": 1}
        await q.update(job)
        stored = await q.get(job.job_id)
        assert stored.status == JobStatus.DONE

    async def test_get_missing_returns_none(self):
        q = JobQueue()
        assert await q.get("nonexistent") is None

    async def test_queue_full_raises(self):
        import os
        with patch.dict(os.environ, {"QUEUE_MAX_SIZE": "1"}):
            import importlib, job_queue as jq
            importlib.reload(jq)
            q = jq.JobQueue()
            await q.submit(Job(inputs="a"))
            with pytest.raises(jq.QueueFullError):
                await q.submit(Job(inputs="b"))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# metrics.py coverage
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
from metrics import (
    record_classification, record_error,
    update_queue_depth, set_active_workers,
    classifications_total, classification_errors_total, queue_depth_gauge,
)

@pytest.mark.unit
class TestMetricsCoverage:
    def test_record_success(self):
        before = classifications_total.labels(status="success", top_label="billing")._value.get()
        record_classification("billing", 500, 1)
        after = classifications_total.labels(status="success", top_label="billing")._value.get()
        assert after == before + 1

    def test_record_error(self):
        before = classification_errors_total.labels(error_type="test")._value.get()
        record_error("test")
        after = classification_errors_total.labels(error_type="test")._value.get()
        assert after == before + 1

    def test_queue_depth_gauge(self):
        update_queue_depth(99)
        assert queue_depth_gauge._value.get() == 99

    def test_set_workers(self):
        set_active_workers(4)

    def test_batch_size(self):
        record_classification("complaint", 1000, batch_size=8)

    def test_multiple_labels(self):
        for label in ["billing", "complaint", "general_question"]:
            record_classification(label, 300)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# logger.py coverage
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@pytest.mark.unit
class TestLoggerCoverage:
    def test_log_classification(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib, logger as lm
        importlib.reload(lm)
        lm.log_classification("j1", "test input", "billing", 0.85, [], 500.0)
        assert (tmp_path / "classifications.jsonl").exists()

    def test_log_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib, logger as lm
        importlib.reload(lm)
        lm.log_error("j1", "inference_error", "Something failed")
        assert (tmp_path / "errors.jsonl").exists()

    def test_get_stats_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib, logger as lm
        importlib.reload(lm)
        stats = lm.get_stats()
        assert stats["total"] == 0

    def test_get_stats_with_data(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib, logger as lm
        importlib.reload(lm)
        for label in ["billing", "billing", "complaint"]:
            lm.log_classification("j", "t", label, 0.8, [], 500)
        stats = lm.get_stats()
        assert stats["total"] == 3
        assert stats["label_counts"]["billing"] == 2

    def test_read_recent_classifications(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib, logger as lm
        importlib.reload(lm)
        assert isinstance(lm.read_recent_classifications(), list)

    def test_read_recent_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib, logger as lm
        importlib.reload(lm)
        assert isinstance(lm.read_recent_errors(), list)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# api_keys.py coverage
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
from api_keys import APIKeyStore

@pytest.mark.unit
class TestAPIKeysCoverage:
    def _store(self):
        s = APIKeyStore.__new__(APIKeyStore)
        s._keys = {}
        s._save = lambda: None
        return s

    def test_create_user_key(self):
        s = self._store()
        raw, key = s.create("test", "user")
        assert raw.startswith("sk-")
        assert key.role == "user"

    def test_create_admin_key(self):
        s = self._store()
        raw, key = s.create("admin", "admin")
        assert key.role == "admin"

    def test_validate_valid(self):
        s = self._store()
        raw, key = s.create("test", "user")
        assert s.validate(raw) is not None

    def test_validate_invalid(self):
        s = self._store()
        assert s.validate("sk-invalid") is None

    def test_revoke(self):
        s = self._store()
        raw, key = s.create("test", "user")
        s.revoke(key.key_id)
        assert s.validate(raw) is None

    def test_revoke_nonexistent(self):
        s = self._store()
        assert s.revoke("nonexistent") is False

    def test_list_keys(self):
        s = self._store()
        s.create("a", "user")
        s.create("b", "admin")
        assert len(s.list_keys()) == 2

    def test_has_any_false(self):
        s = self._store()
        assert s.has_any() is False

    def test_has_any_true(self):
        s = self._store()
        s.create("test", "user")
        assert s.has_any() is True

    def test_to_public_dict_no_hash(self):
        s = self._store()
        _, key = s.create("test", "user")
        pub = key.to_public_dict()
        assert "key_hash" not in pub


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# rate_limiter.py coverage
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
from rate_limiter import check_rate_limit, get_usage, _request_log, RATE_LIMIT
from fastapi import HTTPException

@pytest.mark.unit
class TestRateLimiterCoverage:
    def _make_key(self, name="test"):
        from api_keys import APIKeyStore
        s = APIKeyStore.__new__(APIKeyStore)
        s._keys = {}
        s._save = lambda: None
        _, key = s.create(name, "user")
        return key

    def test_under_limit_passes(self):
        key = self._make_key("under")
        check_rate_limit(key)   # should not raise

    def test_over_limit_raises(self):
        key = self._make_key("over")
        _request_log[key.key_id] = [time.time()] * (RATE_LIMIT + 1)
        with pytest.raises(HTTPException) as exc:
            check_rate_limit(key)
        assert exc.value.status_code == 429

    def test_get_usage_returns_dict(self):
        key = self._make_key("usage")
        usage = get_usage(key.key_id)
        assert "requests_last_minute" in usage
        assert "limit_per_minute" in usage
        assert "remaining" in usage

    def test_remaining_decreases(self):
        key = self._make_key("remaining")
        before = get_usage(key.key_id)["remaining"]
        check_rate_limit(key)
        after = get_usage(key.key_id)["remaining"]
        assert after < before


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# auth.py coverage
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@pytest.mark.unit
class TestAuthCoverage:
    def test_valid_key_passes(self, app_client, user_key, mock_groq):
        r = app_client.post("/classify",
                            json={"inputs": "test"},
                            headers={"X-API-Key": user_key})
        assert r.status_code == 200

    def test_missing_key_returns_401(self, app_client):
        r = app_client.post("/classify", json={"inputs": "test"})
        assert r.status_code == 401

    def test_invalid_key_returns_401(self, app_client):
        r = app_client.post("/classify",
                            json={"inputs": "test"},
                            headers={"X-API-Key": "sk-bad-key"})
        assert r.status_code == 401

    def test_user_forbidden_from_admin(self, app_client, user_key):
        r = app_client.get("/admin/keys",
                           headers={"X-API-Key": user_key})
        assert r.status_code == 403

    def test_admin_allowed_admin_endpoint(self, app_client, admin_key):
        r = app_client.get("/admin/keys",
                           headers={"X-API-Key": admin_key})
        assert r.status_code == 200


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# API endpoint coverage
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@pytest.mark.integration
class TestAPIEndpointsCoverage:
    def test_health(self, app_client):
        assert app_client.get("/health").status_code == 200

    def test_metrics(self, app_client):
        r = app_client.get("/metrics")
        assert r.status_code == 200
        assert "classifications_total" in r.text

    def test_dashboard(self, app_client):
        r = app_client.get("/dashboard")
        assert r.status_code == 200
        assert "Phase E" in r.text

    def test_classify_sync(self, app_client, user_key, mock_groq):
        r = app_client.post("/classify",
                            json={"inputs": "I need a refund"},
                            headers={"X-API-Key": user_key})
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert "latency_ms" in body

    def test_classify_list_input(self, app_client, user_key, mock_groq):
        r = app_client.post("/classify",
                            json={"inputs": ["text one", "text two"]},
                            headers={"X-API-Key": user_key})
        assert r.status_code == 200

    def test_classify_empty_returns_422(self, app_client, user_key):
        r = app_client.post("/classify",
                            json={"inputs": ""},
                            headers={"X-API-Key": user_key})
        assert r.status_code == 422

    def test_classify_async(self, app_client, user_key, mock_groq):
        r = app_client.post("/classify/async",
                            json={"inputs": "async test"},
                            headers={"X-API-Key": user_key})
        assert r.status_code == 202
        assert "job_id" in r.json()

    def test_job_not_found(self, app_client, user_key):
        r = app_client.get("/jobs/nonexistent",
                           headers={"X-API-Key": user_key})
        assert r.status_code == 404

    def test_stats_endpoint(self, app_client, user_key):
        r = app_client.get("/stats",
                           headers={"X-API-Key": user_key})
        assert r.status_code == 200

    def test_me_endpoint(self, app_client, user_key):
        r = app_client.get("/me",
                           headers={"X-API-Key": user_key})
        assert r.status_code == 200

    def test_classify_labels(self, app_client, user_key):
        r = app_client.get("/classify/labels",
                           headers={"X-API-Key": user_key})
        assert r.status_code == 200

    def test_admin_create_key(self, app_client, admin_key):
        r = app_client.post("/admin/keys?name=coverage-key&role=user",
                            headers={"X-API-Key": admin_key})
        assert r.status_code == 200
        assert "raw_key" in r.json()

    def test_admin_revoke_nonexistent(self, app_client, admin_key):
        r = app_client.delete("/admin/keys/nonexistent-id",
                              headers={"X-API-Key": admin_key})
        assert r.status_code == 404

    def test_openapi_docs(self, app_client):
        assert app_client.get("/openapi.json").status_code == 200
        assert app_client.get("/docs").status_code == 200
        

class TestJobQueueExtended:
    @pytest.mark.asyncio
    async def test_submit_returns_job(self):
        from schemas import Job
        q = JobQueue()
        job = Job(inputs="hello")
        result = await q.submit(job)
        assert result.job_id == job.job_id

    @pytest.mark.asyncio
    async def test_update_result(self):
        from schemas import Job, JobStatus
        q = JobQueue()
        job = Job(inputs="hello")
        await q.submit(job)
        job.status = JobStatus.DONE
        job.outputs = {"output": "ok"}
        await q._backend.update(job)
        fetched = await q._backend.get(job.job_id)
        assert fetched.status == JobStatus.DONE

    @pytest.mark.asyncio
    async def test_update_failed(self):
        from schemas import Job, JobStatus
        q = JobQueue()
        job = Job(inputs="hello")
        await q.submit(job)
        job.status = JobStatus.FAILED
        job.error = "something went wrong"
        await q._backend.update(job)
        fetched = await q._backend.get(job.job_id)
        assert fetched.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_depth(self):
        from schemas import Job
        from job_queue import InMemoryBackend
        backend = InMemoryBackend(maxsize=10)
        await backend.enqueue(Job(inputs="hello"))
        await backend.enqueue(Job(inputs="world"))
        depth = await backend.depth()
        assert depth >= 2


class TestLoggerExtended:
    def test_log_and_read_multiple(self, tmp_path, monkeypatch):
        import logger
        monkeypatch.setattr(logger, "CLASSIFICATION_LOG", tmp_path / "classifications.jsonl")
        monkeypatch.setattr(logger, "ERROR_LOG", tmp_path / "errors.jsonl")
        for i in range(3):
            logger.log_classification(job_id=f"job-{i}", input_text=f"text {i}", top_label="billing", top_score=0.9, all_labels=[{"label": "billing", "score": 0.9}], latency_ms=100.0)
        assert len(logger.read_recent_classifications(limit=3)) == 3

    def test_log_error_and_read(self, tmp_path, monkeypatch):
        import logger
        monkeypatch.setattr(logger, "CLASSIFICATION_LOG", tmp_path / "classifications.jsonl")
        monkeypatch.setattr(logger, "ERROR_LOG", tmp_path / "errors.jsonl")
        logger.log_error(job_id="job-err-1", input_text="bad input", error_type="ValueError", error_message="something failed")
        assert len(logger.read_recent_errors(limit=5)) == 1

    def test_stats_label_counts(self, tmp_path, monkeypatch):
        import logger
        monkeypatch.setattr(logger, "CLASSIFICATION_LOG", tmp_path / "classifications.jsonl")
        monkeypatch.setattr(logger, "ERROR_LOG", tmp_path / "errors.jsonl")
        for i, label in enumerate(["billing", "billing", "complaint"]):
            logger.log_classification(job_id=f"job-{i}", input_text="test", top_label=label, top_score=0.8, all_labels=[{"label": label, "score": 0.8}], latency_ms=50.0)
        stats = logger.get_stats()
        assert stats["label_counts"]["billing"] == 2
        assert stats["label_counts"]["complaint"] == 1


class TestAPIKeysExtended:
    def test_revoke_already_revoked(self):
        from api_keys import APIKeyStore
    store = APIKeyStore()
    raw, key = store.create(name="test-key-unique-1", role="user")
    result1 = store.revoke(key.key_id)
    assert result1 is True
    result2 = store.revoke(key.key_id)
    # key still exists in store but is inactive — revoke returns True again
    assert result2 is True
    # validate should return None since key is inactive
    assert store.validate(raw) is None
    

    

    def test_create_multiple_keys(self):
        from api_keys import APIKeyStore
        store = APIKeyStore()
        k1, _ = store.create(name="key-unique-aaa", role="user")
        k2, _ = store.create(name="key-unique-bbb", role="admin")
        assert k1 != k2
        assert len(store.list_keys()) >= 2
