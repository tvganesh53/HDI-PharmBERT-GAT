"""
test_metrics.py — Phase D test suite.
Tests metrics recording, logger, and all monitoring endpoints.

Run:
    pytest test_metrics.py -v
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics tests
# ═══════════════════════════════════════════════════════════════════════════════
from metrics import (
    record_classification, record_error,
    update_queue_depth, set_active_workers,
    classifications_total, classification_errors_total,
    queue_depth_gauge,
)


class TestMetrics:
    def test_record_classification_increments_counter(self):
        before = classifications_total.labels(status="success", top_label="billing")._value.get()
        record_classification("billing", latency_ms=500, batch_size=1)
        after = classifications_total.labels(status="success", top_label="billing")._value.get()
        assert after == before + 1

    def test_record_error_increments_counter(self):
        before = classification_errors_total.labels(error_type="test_error")._value.get()
        record_error("test_error")
        after = classification_errors_total.labels(error_type="test_error")._value.get()
        assert after == before + 1

    def test_update_queue_depth(self):
        update_queue_depth(42)
        assert queue_depth_gauge._value.get() == 42

    def test_set_active_workers(self):
        set_active_workers(4)

    def test_record_classification_with_batch(self):
        record_classification("complaint", latency_ms=1200, batch_size=8)

    def test_record_multiple_labels(self):
        for label in ["billing", "complaint", "technical_support"]:
            record_classification(label, latency_ms=300)


# ═══════════════════════════════════════════════════════════════════════════════
# Logger tests
# ═══════════════════════════════════════════════════════════════════════════════
class TestLogger:
    def test_log_classification_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib
        import logger as logger_mod
        importlib.reload(logger_mod)

        logger_mod.log_classification(
            job_id="test-job-1",
            input_text="Test input",
            top_label="billing",
            top_score=0.85,
            all_labels=[{"label": "billing", "score": 0.85}],
            latency_ms=500.0,
        )
        log_file = tmp_path / "classifications.jsonl"
        assert log_file.exists()

    def test_log_classification_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib
        import logger as logger_mod
        importlib.reload(logger_mod)

        logger_mod.log_classification(
            job_id="test-job-2",
            input_text="Another test",
            top_label="complaint",
            top_score=0.72,
            all_labels=[],
            latency_ms=800.0,
        )
        lines = (tmp_path / "classifications.jsonl").read_text().strip().splitlines()
        record = json.loads(lines[0])
        assert record["top_label"] == "complaint"
        assert record["job_id"] == "test-job-2"

    def test_log_error_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib
        import logger as logger_mod
        importlib.reload(logger_mod)

        logger_mod.log_error("test-job-3", "inference_error", "Something went wrong")
        assert (tmp_path / "errors.jsonl").exists()

    def test_get_stats_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib
        import logger as logger_mod
        importlib.reload(logger_mod)

        stats = logger_mod.get_stats()
        assert stats["total"] == 0

    def test_get_stats_counts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib
        import logger as logger_mod
        importlib.reload(logger_mod)

        for label in ["billing", "billing", "complaint"]:
            logger_mod.log_classification(
                job_id="j", input_text="t",
                top_label=label, top_score=0.8,
                all_labels=[], latency_ms=500,
            )
        stats = logger_mod.get_stats()
        assert stats["total"] == 3
        assert stats["label_counts"]["billing"] == 2
        assert stats["label_counts"]["complaint"] == 1

    def test_read_recent_returns_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import importlib
        import logger as logger_mod
        importlib.reload(logger_mod)

        result = logger_mod.read_recent_classifications()
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════════════
# API endpoint tests
# ═══════════════════════════════════════════════════════════════════════════════
from fastapi.testclient import TestClient
from app_phase_d import app

DEFAULT_LABELS = [
    "billing", "technical_support", "account_management",
    "sales_inquiry", "complaint", "feature_request",
    "general_question", "other",
]

def _make_mock_response(labels):
    n = len(labels)
    remaining = 1.0
    classifications = []
    for i, label in enumerate(labels):
        score = round(remaining * 0.5, 4) if i < n - 1 else round(remaining, 4)
        classifications.append({"label": label, "score": score, "reasoning": "test"})
        remaining -= score
    payload = json.dumps({"classifications": classifications})
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=payload)]
    return mock_msg


@pytest.fixture(scope="module")
def client():
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with TestClient(app) as c:
            yield c


def _patch_groq(client):
    from app_phase_d import predictor as p
    p._client.chat.completions.create = MagicMock(
        return_value=_make_mock_response(DEFAULT_LABELS)
    )


class TestPhaseDEndpoints:
    def test_health_200(self, client):
        assert client.get("/health").status_code == 200

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "classifications_total" in r.text

    def test_stats_endpoint(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        body = r.json()
        assert "total" in body
        assert "label_counts" in body
        assert "avg_latency_ms" in body
        assert "queue_depth" in body

    def test_dashboard_returns_html(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Phase D" in r.text

    def test_logs_classifications_endpoint(self, client):
        r = client.get("/logs/classifications")
        assert r.status_code == 200
        assert "records" in r.json()

    def test_logs_errors_endpoint(self, client):
        r = client.get("/logs/errors")
        assert r.status_code == 200
        assert "records" in r.json()

    def test_classify_records_metrics(self, client):
        _patch_groq(client)
        r = client.post("/classify", json={"inputs": "I need a refund"})
        assert r.status_code == 200

    def test_classify_response_schema(self, client):
        _patch_groq(client)
        body = client.post("/classify", json={"inputs": "test"}).json()
        assert "results" in body
        assert "latency_ms" in body

    def test_metrics_after_classify(self, client):
        _patch_groq(client)
        client.post("/classify", json={"inputs": "billing test"})
        r = client.get("/metrics")
        assert "classifications_total" in r.text

    def test_openapi_docs(self, client):
        assert client.get("/openapi.json").status_code == 200
