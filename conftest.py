"""
conftest.py — Shared pytest fixtures for all test files.
"""

from __future__ import annotations
import json
import os
from fastapi.testclient import TestClient
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Default labels ────────────────────────────────────────────────────────────
DEFAULT_LABELS = [
    "billing", "technical_support", "account_management",
    "sales_inquiry", "complaint", "feature_request",
    "general_question", "other",
]


# ── Mock Groq response ────────────────────────────────────────────────────────
def make_mock_groq_response():
    """Build a realistic Groq API response with correct JSON structure."""
    from unittest.mock import MagicMock

    label_json = json.dumps({
        "classifications": [
            {"label": "billing",            "score": 0.65, "reasoning": "billing issue"},
            {"label": "complaint",          "score": 0.20, "reasoning": "complaint tone"},
            {"label": "technical_support",  "score": 0.05, "reasoning": "not technical"},
            {"label": "account_management", "score": 0.04, "reasoning": "not account"},
            {"label": "sales_inquiry",      "score": 0.03, "reasoning": "not sales"},
            {"label": "feature_request",    "score": 0.02, "reasoning": "not feature"},
            {"label": "general_question",   "score": 0.01, "reasoning": "not general"},
        ]
    })

    message = MagicMock()
    message.content = label_json

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response

# ── Environment fixture ───────────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def test_environment(tmp_path_factory: pytest.TempPathFactory):
    """Set up test environment variables for the whole session."""
    tmp = tmp_path_factory.mktemp("test_data")
    env_vars = {
        "GROQ_API_KEY":            "test-key-fixture",
        "CLASSIFIER_MODEL":        "llama-3.1-8b-instant",
        "CLASSIFIER_CONTEXT":      "customer support messages",
        "CLASSIFIER_LABELS":       ",".join(DEFAULT_LABELS),
        "MAX_INPUT_LENGTH":        "512",
        "MAX_BATCH_SIZE":          "32",
        "BATCH_TIMEOUT_MS":        "50",
        "NUM_WORKERS":             "2",
        "SYNC_TIMEOUT_S":          "30",
        "QUEUE_MAX_SIZE":          "500",
        "QUEUE_BACKEND":           "memory",
        "RATE_LIMIT_PER_MINUTE":   "60",
        "KEYS_FILE":               str(tmp / "keys.json"),
        "LOG_DIR":                 str(tmp / "logs"),
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


# ── App client fixture ────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def app_client():
    """Create test client with mocked predictor loaded."""
    from unittest.mock import MagicMock, patch
    
    mock_response = make_mock_groq_response()
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_response)

    with patch("predictor_c.Groq", return_value=mock_client):
        from app_phase_e import app
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            yield client


# ── Admin key fixture ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def admin_key(app_client: TestClient):
    from api_keys import key_store
    raw_key, _ = key_store.create(name="fixture-admin", role="admin")
    return raw_key


# ── User key fixture ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def user_key(app_client: TestClient, admin_key: str):
    from api_keys import key_store
    raw_key, _ = key_store.create(name="fixture-user", role="user")
    return raw_key


# ── Groq mock fixture ─────────────────────────────────────────────────────────
@pytest.fixture
def mock_groq(app_client: TestClient):
    """Patch the Groq client for the duration of a test."""
    from unittest.mock import MagicMock
    from app_phase_e import predictor

    mock_response = make_mock_groq_response()
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_response)

    if predictor:
        predictor._client = mock_client
    yield mock_client
