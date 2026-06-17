"""
api_keys.py — API key management for Phase E.

Stores keys in keys.json (file-based, no DB needed).
Switch to Redis or Postgres in production.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("phase_e.api_keys")

KEYS_FILE = Path(os.getenv("KEYS_FILE", "keys.json"))


# ── Data model ────────────────────────────────────────────────────────────────
class APIKey:
    def __init__(
        self,
        key_id: str,
        key_hash: str,
        name: str,
        role: str = "user",       # "user" | "admin"
        created_at: float = None,
        is_active: bool = True,
        requests_today: int = 0,
        last_used: float = None,
    ):
        self.key_id        = key_id
        self.key_hash      = key_hash
        self.name          = name
        self.role          = role
        self.created_at    = created_at or time.time()
        self.is_active     = is_active
        self.requests_today = requests_today
        self.last_used     = last_used

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id":         self.key_id,
            "key_hash":       self.key_hash,
            "name":           self.name,
            "role":           self.role,
            "created_at":     self.created_at,
            "is_active":      self.is_active,
            "requests_today": self.requests_today,
            "last_used":      self.last_used,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Return safe dict — never exposes the hash."""
        return {
            "key_id":         self.key_id,
            "name":           self.name,
            "role":           self.role,
            "created_at":     self.created_at,
            "is_active":      self.is_active,
            "requests_today": self.requests_today,
            "last_used":      self.last_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "APIKey":
        return cls(**d)


# ── Key store ─────────────────────────────────────────────────────────────────
class APIKeyStore:
    """File-backed API key store."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKey] = {}
        self._load()

    def _load(self) -> None:
        if KEYS_FILE.exists():
            try:
                data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
                self._keys = {k: APIKey.from_dict(v) for k, v in data.items()}
                log.info("Loaded %d API key(s) from %s", len(self._keys), KEYS_FILE)
            except Exception as exc:
                log.error("Failed to load keys file: %s", exc)
                self._keys = {}
        else:
            self._keys = {}

    def _save(self) -> None:
        try:
            KEYS_FILE.write_text(
                json.dumps({k: v.to_dict() for k, v in self._keys.items()}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.error("Failed to save keys file: %s", exc)

    @staticmethod
    def _hash(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    # ── CRUD ──────────────────────────────────────────────────────────────────
    def create(self, name: str, role: str = "user") -> tuple[str, APIKey]:
        """Create a new API key. Returns (raw_key, APIKey)."""
        raw_key = "sk-" + secrets.token_urlsafe(32)
        key_id  = "kid-" + secrets.token_hex(8)
        api_key = APIKey(
            key_id=key_id,
            key_hash=self._hash(raw_key),
            name=name,
            role=role,
        )
        self._keys[key_id] = api_key
        self._save()
        log.info("Created API key '%s' (role=%s)", name, role)
        return raw_key, api_key

    def validate(self, raw_key: str) -> APIKey | None:
        # reload from disk to pick up external changes
        self._load()
        h = self._hash(raw_key)
        for key in self._keys.values():
            if key.key_hash == h and key.is_active:
                key.last_used = time.time()
                key.requests_today += 1
                self._save()
                return key
        return None

    def revoke(self, key_id: str) -> bool:
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            self._save()
            log.info("Revoked API key %s", key_id)
            return True
        return False

    def list_keys(self) -> list[dict]:
        return [k.to_public_dict() for k in self._keys.values()]

    def count(self) -> int:
        return len(self._keys)

    def has_any(self) -> bool:
        return any(k.is_active for k in self._keys.values())


# ── Singleton ─────────────────────────────────────────────────────────────────
key_store = APIKeyStore()


def ensure_admin_key() -> str | None:
    """
    If no keys exist, create a default admin key on first startup.
    Returns the raw key (shown once) or None if keys already exist.
    """
    if not key_store.has_any():
        raw_key, _ = key_store.create(name="default-admin", role="admin")
        log.warning("=" * 60)
        log.warning("  No API keys found — created default admin key:")
        log.warning("  %s", raw_key)
        log.warning("  Save this key — it will NOT be shown again!")
        log.warning("=" * 60)
        return raw_key
    return None
