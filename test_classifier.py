"""
test_classifier.py — Phase C test suite.

Mocks the Anthropic API so tests run without a real API key.

Run:
    pytest test_classifier.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Mock Anthropic response factory ───────────────────────────────────────────
def _make_mock_response(labels: list[str]) -> MagicMock:
    """Build a fake anthropic.Message response with ranked classifications."""
    n = len(labels)
    scores = []
    remaining = 1.0
    for i, label in enumerate(labels):
        if i == n - 1:
            scores.append(round(remaining, 4))
        else:
            score = round(remaining * 0.5, 4)
            scores.append(score)
            remaining -= score

    classifications = [
        {"label": lbl, "score": sc, "reasoning": f"Test reasoning for {lbl}"}
        for lbl, sc in zip(labels, scores)
    ]
    payload = json.dumps({"classifications": classifications})

    mock_msg           = MagicMock()
    mock_msg.content   = [MagicMock(text=payload)]
    return mock_msg


DEFAULT_LABELS = [
    "billing", "technical_support", "account_management",
    "sales_inquiry", "complaint", "feature_request",
    "general_question", "other",
]


# ── Predictor unit tests ───────────────────────────────────────────────────────
from predictor_c import Predictor, PredictorError
import anthropic


class TestPredictor:
    @pytest.fixture
    def predictor(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            p = Predictor()
            p.load()
            return p

    def test_load_sets_is_loaded(self, predictor):
        assert predictor.is_loaded is True

    def test_device_is_api(self, predictor):
        assert predictor.device == "api"

    def test_model_params_has_labels(self, predictor):
        params = predictor.model_params
        assert "labels" in params
        assert isinstance(params["labels"], list)

    def test_predict_single_string(self, predictor):
        mock_resp = _make_mock_response(DEFAULT_LABELS)
        predictor._client.messages.create = MagicMock(return_value=mock_resp)
        result = predictor.predict("I need help with my bill")
        assert "top_label" in result
        assert "classifications" in result
        assert len(result["classifications"]) == len(DEFAULT_LABELS)

    def test_predict_list_of_texts(self, predictor):
        mock_resp = _make_mock_response(DEFAULT_LABELS)
        predictor._client.messages.create = MagicMock(return_value=mock_resp)
        result = predictor.predict(["text one", "text two"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_classifications_sorted_by_score(self, predictor):
        mock_resp = _make_mock_response(DEFAULT_LABELS)
        predictor._client.messages.create = MagicMock(return_value=mock_resp)
        result = predictor.predict("test text")
        scores = [c["score"] for c in result["classifications"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_label_matches_first_classification(self, predictor):
        mock_resp = _make_mock_response(DEFAULT_LABELS)
        predictor._client.messages.create = MagicMock(return_value=mock_resp)
        result = predictor.predict("test")
        assert result["top_label"] == result["classifications"][0]["label"]

    def test_scores_sum_to_one(self, predictor):
        mock_resp = _make_mock_response(DEFAULT_LABELS)
        predictor._client.messages.create = MagicMock(return_value=mock_resp)
        result = predictor.predict("test")
        total = sum(c["score"] for c in result["classifications"])
        assert abs(total - 1.0) < 0.01

    def test_all_labels_present_in_output(self, predictor):
        mock_resp = _make_mock_response(DEFAULT_LABELS)
        predictor._client.messages.create = MagicMock(return_value=mock_resp)
        result = predictor.predict("test")
        output_labels = {c["label"] for c in result["classifications"]}
        for label in DEFAULT_LABELS:
            assert label in output_labels

    def test_auth_error_raises_predictor_error(self, predictor):
        predictor._client.messages.create = MagicMock(
            side_effect=anthropic.AuthenticationError(
                message="Invalid key",
                response=MagicMock(status_code=401),
                body={}
            )
        )
        with pytest.raises(PredictorError, match="Invalid ANTHROPIC_API_KEY"):
            predictor.predict("test")

    def test_rate_limit_raises_predictor_error(self, predictor):
        predictor._client.messages.create = MagicMock(
            side_effect=anthropic.RateLimitError(
                message="Rate limit",
                response=MagicMock(status_code=429),
                body={}
            )
        )
        with pytest.raises(PredictorError, match="rate limit"):
            predictor.predict("test")

    def test_malformed_json_uses_fallback(self, predictor):
        mock_msg         = MagicMock()
        mock_msg.content = [MagicMock(text="this is not json at all")]
        predictor._client.messages.create = MagicMock(return_value=mock_msg)
        result = predictor.predict("test")
        # Fallback still returns valid structure
        assert "classifications" in result
        assert len(result["classifications"]) > 0

    def test_custom_labels_override(self, predictor):
        custom_labels = ["urgent", "normal", "low_priority"]
        mock_resp = _make_mock_response(custom_labels)
        predictor._client.messages.create = MagicMock(return_value=mock_resp)
        result = predictor.predict("test", labels=custom_labels)
        output_labels = {c["label"] for c in result["classifications"]}
        for label in custom_labels:
            assert label in output_labels

    def test_not_loaded_raises(self):
        p = Predictor()
        with pytest.raises(PredictorError, match="not loaded"):
            p.predict("test")


# ═══════════════════════════════════════════════════════════════════════════════
# API endpoint tests
# ═══════════════════════════════════════════════════════════════════════════════
from app_phase_c import app


@pytest.fixture(scope="module")
def client():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        with TestClient(app) as c:
            yield c


def _patch_predictor(client, labels=None):
    """Helper: patch the loaded predictor's _client for endpoint tests."""
    from app_phase_c import predictor as p
    labels = labels or DEFAULT_LABELS
    mock_resp = _make_mock_response(labels)
    p._client.messages.create = MagicMock(return_value=mock_resp)


class TestClassifyEndpoints:
    def test_health_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_has_label_count(self, client):
        body = client.get("/health").json()
        assert "label_count" in body
        assert body["label_count"] > 0

    def test_get_labels_200(self, client):
        r = client.get("/classify/labels")
        assert r.status_code == 200
        body = r.json()
        assert "labels" in body
        assert "count" in body
        assert body["count"] == len(body["labels"])

    def test_classify_sync_200(self, client):
        _patch_predictor(client)
        r = client.post("/classify", json={"inputs": "I need help with my invoice"})
        assert r.status_code == 200

    def test_classify_sync_schema(self, client):
        _patch_predictor(client)
        body = client.post("/classify", json={"inputs": "test text"}).json()
        assert "results" in body
        assert "latency_ms" in body
        assert "label_set" in body

    def test_classify_sync_has_classifications(self, client):
        _patch_predictor(client)
        body = client.post("/classify", json={"inputs": "test text"}).json()
        results = body["results"]
        if isinstance(results, list):
            result = results[0]
        else:
            result = results
        assert "top_label" in result
        assert "top_score" in result
        assert "classifications" in result

    def test_classify_sync_empty_string(self, client):
        r = client.post("/classify", json={"inputs": ""})
        assert r.status_code == 422

    def test_classify_sync_null(self, client):
        r = client.post("/classify", json={"inputs": None})
        assert r.status_code == 422

    def test_classify_sync_list_input(self, client):
        _patch_predictor(client)
        r = client.post("/classify", json={"inputs": ["text one", "text two"]})
        assert r.status_code == 200

    def test_classify_sync_custom_labels(self, client):
        custom = ["urgent", "normal", "low_priority"]
        _patch_predictor(client, labels=custom)
        r = client.post("/classify", json={
            "inputs": "this is urgent",
            "labels": custom,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["label_set"] == custom

    def test_classify_async_202(self, client):
        _patch_predictor(client)
        r = client.post("/classify/async", json={"inputs": "async test"})
        assert r.status_code == 202

    def test_classify_async_returns_job_id(self, client):
        _patch_predictor(client)
        body = client.post("/classify/async", json={"inputs": "async test"}).json()
        assert "job_id" in body
        assert body["status"] == "pending"

    def test_job_polling_reaches_done(self, client):
        _patch_predictor(client)
        body = client.post("/classify/async", json={"inputs": "poll test"}).json()
        job_id = body["job_id"]
        import time
        deadline = time.monotonic() + 10
        status = "pending"
        while time.monotonic() < deadline:
            r = client.get(f"/jobs/{job_id}")
            assert r.status_code == 200
            status = r.json()["status"]
            if status in ("done", "failed"):
                break
            time.sleep(0.1)
        assert status == "done"

    def test_job_not_found(self, client):
        assert client.get("/jobs/nonexistent-id").status_code == 404

    def test_html_input_is_cleaned(self, client):
        _patch_predictor(client)
        r = client.post("/classify", json={"inputs": "<p>I need a refund</p>"})
        assert r.status_code == 200

    def test_models_info_200(self, client):
        assert client.get("/models/info").status_code == 200

    def test_openapi_docs(self, client):
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
