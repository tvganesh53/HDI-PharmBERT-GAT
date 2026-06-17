"""
test_auth.py — Phase E test suite.
Tests API key auth, rate limiting, and all secured endpoints.

Run:
    pytest test_auth.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api_keys import APIKeyStore
from app_phase_e import app

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
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=json.dumps({"classifications": classifications}))]
    return mock_msg


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("keys")
    with patch.dict("os.environ", {
        "GROQ_API_KEY": "test-key",
        "KEYS_FILE":    str(tmp / "keys.json"),
        "LOG_DIR":      str(tmp / "logs"),
    }):
        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="module")
def admin_key(client):
    """Get or create an admin key for testing."""
    from api_keys import key_store
    raw_key, _ = key_store.create(name="test-admin", role="admin")
    return raw_key


@pytest.fixture(scope="module")
def user_key(client, admin_key):
    """Create a user key via the admin endpoint."""
    r = client.post(
        "/admin/keys?name=test-user&role=user",
        headers={"X-API-Key": admin_key},
    )
    return r.json()["raw_key"]


def _patch_groq(client):
    from app_phase_e import predictor as p
    p._client.chat.completions.create = MagicMock(
        return_value=_make_mock_response(DEFAULT_LABELS)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public endpoints (no auth needed)
# ═══════════════════════════════════════════════════════════════════════════════
class TestPublicEndpoints:
    def test_health_no_auth(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_metrics_no_auth(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_docs_no_auth(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_no_auth(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Authentication tests
# ═══════════════════════════════════════════════════════════════════════════════
class TestAuthentication:
    def test_classify_without_key_returns_401(self, client):
        r = client.post("/classify", json={"inputs": "test"})
        assert r.status_code == 401

    def test_classify_with_invalid_key_returns_401(self, client):
        r = client.post("/classify",
                        json={"inputs": "test"},
                        headers={"X-API-Key": "invalid-key"})
        assert r.status_code == 401

    def test_classify_with_valid_key_returns_200(self, client, user_key):
        _patch_groq(client)
        r = client.post("/classify",
                        json={"inputs": "test input"},
                        headers={"X-API-Key": user_key})
        assert r.status_code == 200

    def test_stats_without_key_returns_401(self, client):
        assert client.get("/stats").status_code == 401

    def test_dashboard_without_key_returns_401(self, client):
        assert client.get("/dashboard").status_code == 401

    def test_me_endpoint_with_valid_key(self, client, user_key):
        r = client.get("/me", headers={"X-API-Key": user_key})
        assert r.status_code == 200
        body = r.json()
        assert "key_id" in body
        assert "role" in body
        assert "usage" in body

    def test_me_endpoint_without_key_returns_401(self, client):
        assert client.get("/me").status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# Role-based access tests
# ═══════════════════════════════════════════════════════════════════════════════
class TestRoleBasedAccess:
    def test_user_cannot_access_admin_endpoints(self, client, user_key):
        r = client.get("/admin/keys", headers={"X-API-Key": user_key})
        assert r.status_code == 403

    def test_admin_can_access_admin_endpoints(self, client, admin_key):
        r = client.get("/admin/keys", headers={"X-API-Key": admin_key})
        assert r.status_code == 200

    def test_admin_can_create_key(self, client, admin_key):
        r = client.post(
            "/admin/keys?name=new-test-key&role=user",
            headers={"X-API-Key": admin_key},
        )
        assert r.status_code == 200
        body = r.json()
        assert "raw_key" in body
        assert body["raw_key"].startswith("sk-")

    def test_admin_can_list_keys(self, client, admin_key):
        r = client.get("/admin/keys", headers={"X-API-Key": admin_key})
        assert r.status_code == 200
        body = r.json()
        assert "keys" in body
        assert "total" in body

    def test_admin_can_revoke_key(self, client, admin_key):
        raw, key = __import__("api_keys").key_store.create("to-revoke", "user")
        r = client.delete(
            f"/admin/keys/{key.key_id}",
            headers={"X-API-Key": admin_key},
        )
        assert r.status_code == 200

    def test_revoked_key_returns_401(self, client, admin_key):
        from api_keys import key_store
        raw, key = key_store.create("temp-key", "user")
        key_store.revoke(key.key_id)
        r = client.post("/classify",
                        json={"inputs": "test"},
                        headers={"X-API-Key": raw})
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# Rate limiting tests
# ═══════════════════════════════════════════════════════════════════════════════
class TestRateLimiting:
    def test_rate_limit_exceeded(self, client, admin_key):
        from rate_limiter import _request_log, RATE_LIMIT
        from api_keys import key_store
        raw, key = key_store.create("rate-test", "user")

        # Fill up the request log past the limit
        import time
        now = time.time()
        _request_log[key.key_id] = [now] * (RATE_LIMIT + 1)

        r = client.get("/stats", headers={"X-API-Key": raw})
        assert r.status_code == 429

    def test_rate_limit_header_present(self, client, admin_key):
        from rate_limiter import _request_log, RATE_LIMIT
        from api_keys import key_store
        raw, key = key_store.create("rate-test-2", "user")
        import time
        _request_log[key.key_id] = [time.time()] * (RATE_LIMIT + 1)
        r = client.get("/stats", headers={"X-API-Key": raw})
        assert "retry-after" in r.headers


# ═══════════════════════════════════════════════════════════════════════════════
# API key store tests
# ═══════════════════════════════════════════════════════════════════════════════
class TestAPIKeyStore:
    def test_create_key(self, tmp_path):
        store = APIKeyStore.__new__(APIKeyStore)
        store._keys = {}
        raw, key = store.create("test", "user")
        assert raw.startswith("sk-")
        assert key.role == "user"

    def test_validate_valid_key(self, tmp_path):
        store = APIKeyStore.__new__(APIKeyStore)
        store._keys = {}
        store._save = lambda: None
        raw, key = store.create("test", "user")
        store._save = lambda: None
        result = store.validate(raw)
        assert result is not None

    def test_validate_invalid_key(self, tmp_path):
        store = APIKeyStore.__new__(APIKeyStore)
        store._keys = {}
        result = store.validate("sk-invalid-key")
        assert result is None

    def test_revoke_key(self, tmp_path):
        store = APIKeyStore.__new__(APIKeyStore)
        store._keys = {}
        store._save = lambda: None
        raw, key = store.create("test", "user")
        store._save = lambda: None
        store.revoke(key.key_id)
        assert store.validate(raw) is None
