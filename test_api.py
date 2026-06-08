"""
test_api.py — Integration tests for the Phase A Inference API.

Run against a live server:
    uvicorn app:app --reload &
    python test_api.py

Or as pytest tests:
    pytest test_api.py -v

The TestClient spins up the ASGI app in-process (no real server needed for
pytest runs), so all tests work offline.
"""

import pytest
from fastapi.testclient import TestClient
from app import app

# ── Shared client ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════════
# /health
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_schema(self, client):
        body = client.get("/health").json()
        assert "status" in body
        assert "model_loaded" in body
        assert "device" in body
        assert "uptime_s" in body

    def test_model_loaded(self, client):
        body = client.get("/health").json()
        assert body["model_loaded"] is True, "Model should be loaded after startup"

    def test_status_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_uptime_positive(self, client):
        body = client.get("/health").json()
        assert body["uptime_s"] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# /models/info
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelsInfo:
    def test_returns_200(self, client):
        r = client.get("/models/info")
        assert r.status_code == 200

    def test_schema(self, client):
        body = client.get("/models/info").json()
        assert "model_name" in body
        assert "model_version" in body
        assert "device" in body
        assert "parameters" in body

    def test_model_name_nonempty(self, client):
        body = client.get("/models/info").json()
        assert body["model_name"], "model_name must not be empty"

    def test_model_version_nonempty(self, client):
        body = client.get("/models/info").json()
        assert body["model_version"], "model_version must not be empty"

    def test_parameters_is_dict(self, client):
        body = client.get("/models/info").json()
        assert isinstance(body["parameters"], dict)


# ═══════════════════════════════════════════════════════════════════════════════
# /predict
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredict:
    # ── Happy path ────────────────────────────────────────────────────────────
    def test_string_input(self, client):
        r = client.post("/predict", json={"inputs": "hello world"})
        assert r.status_code == 200

    def test_response_schema(self, client):
        body = client.post("/predict", json={"inputs": "test"}).json()
        assert "outputs" in body
        assert "model_name" in body
        assert "latency_ms" in body

    def test_latency_positive(self, client):
        body = client.post("/predict", json={"inputs": "test"}).json()
        assert body["latency_ms"] >= 0

    def test_model_name_in_response(self, client):
        body = client.post("/predict", json={"inputs": "test"}).json()
        assert body["model_name"], "model_name must not be empty in response"

    def test_dict_input(self, client):
        r = client.post("/predict", json={"inputs": {"feature_a": 1.0, "feature_b": 2.0}})
        assert r.status_code == 200

    def test_list_input(self, client):
        r = client.post("/predict", json={"inputs": [1, 2, 3, 4, 5]})
        assert r.status_code == 200

    def test_with_parameters(self, client):
        r = client.post("/predict", json={"inputs": "hello", "parameters": {"temperature": 0.7}})
        assert r.status_code == 200

    def test_echo_stub_returns_inputs(self, client):
        """The stub model echoes back the inputs — verify round-trip."""
        payload = "round-trip test"
        body = client.post("/predict", json={"inputs": payload}).json()
        # The stub model wraps output in {"echo": ..., ...}
        assert body["outputs"]["echo"] == payload

    # ── Validation errors ─────────────────────────────────────────────────────
    def test_empty_string_input(self, client):
        """Empty strings should be rejected by _validate_inputs."""
        r = client.post("/predict", json={"inputs": "   "})
        assert r.status_code == 422

    def test_null_input(self, client):
        """None inputs should be rejected."""
        r = client.post("/predict", json={"inputs": None})
        assert r.status_code == 422

    def test_missing_inputs_field(self, client):
        """Missing required field 'inputs' should return 422."""
        r = client.post("/predict", json={})
        assert r.status_code == 422

    # ── Content-type ──────────────────────────────────────────────────────────
    def test_non_json_body(self, client):
        r = client.post("/predict", content=b"not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAPI / docs
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocs:
    def test_openapi_json(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema

    def test_swagger_ui(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Manual smoke-test (run as a script against a live server)
# ═══════════════════════════════════════════════════════════════════════════════

def smoke_test(base_url: str = "http://localhost:8000") -> None:
    """Quick sanity-check against a running server."""
    import httpx, json

    c = httpx.Client(base_url=base_url, timeout=30)

    def check(label, r):
        icon = "✓" if r.is_success else "✗"
        print(f"  {icon} [{r.status_code}] {label}")
        if not r.is_success:
            print(f"      {r.text[:200]}")

    print(f"\n{'─'*50}")
    print(f"  Phase A Smoke Test → {base_url}")
    print(f"{'─'*50}")

    check("GET /health",      c.get("/health"))
    check("GET /models/info", c.get("/models/info"))
    check("POST /predict (string)", c.post("/predict", json={"inputs": "smoke test"}))
    check("POST /predict (dict)",   c.post("/predict", json={"inputs": {"x": 1}}))
    check("POST /predict (empty) → expect 422",
          c.post("/predict", json={"inputs": ""}))
    check("GET /openapi.json", c.get("/openapi.json"))

    print(f"{'─'*50}\n")


if __name__ == "__main__":
    smoke_test()
