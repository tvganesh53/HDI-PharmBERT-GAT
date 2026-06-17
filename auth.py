"""
auth.py — Authentication middleware for Phase E.

Usage in routes:
    from auth import require_user, require_admin

    @app.get("/protected")
    async def protected(api_key: APIKey = Depends(require_user)):
        ...

    @app.get("/admin-only")
    async def admin_only(api_key: APIKey = Depends(require_admin)):
        ...
"""

from __future__ import annotations
import logging
import time
from collections import defaultdict
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from api_keys import APIKey, key_store

log = logging.getLogger("phase_e.auth")

# ── API key header ────────────────────────────────────────────────────────────
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Brute-force protection ────────────────────────────────────────────────────
# Tracks failed attempts per IP: {ip: [(timestamp, ...), ...]}
_failed_attempts: dict[str, list[float]] = defaultdict(list)
MAX_FAILED      = 10      # max failed attempts
LOCKOUT_SECONDS = 300     # 5 minute lockout window


def _check_brute_force(ip: str) -> None:
    """Raise 429 if the IP has too many recent failed attempts."""
    now     = time.time()
    cutoff  = now - LOCKOUT_SECONDS
    recent  = [t for t in _failed_attempts[ip] if t > cutoff]
    _failed_attempts[ip] = recent
    if len(recent) >= MAX_FAILED:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Try again in {LOCKOUT_SECONDS // 60} minutes.",
        )


def _record_failure(ip: str) -> None:
    _failed_attempts[ip].append(time.time())


def _clear_failures(ip: str) -> None:
    _failed_attempts.pop(ip, None)


# ── Core validator ────────────────────────────────────────────────────────────
async def _get_api_key(
    request: Request,
    raw_key: str | None = Security(API_KEY_HEADER),
) -> APIKey:
    """
    Validate the X-API-Key header.
    Raises 401 if missing/invalid, 429 if brute-forced.
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_brute_force(client_ip)

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Add header: X-API-Key: <your-key>",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    api_key = key_store.validate(raw_key)
    if api_key is None:
        _record_failure(client_ip)
        log.warning("Invalid API key attempt from %s", client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    _clear_failures(client_ip)
    log.debug("Authenticated key '%s' (role=%s)", api_key.name, api_key.role)
    return api_key


# ── Role-based dependencies ───────────────────────────────────────────────────
async def require_user(api_key: APIKey = Depends(_get_api_key)) -> APIKey:
    """Allow any valid key (user or admin)."""
    return api_key


async def require_admin(api_key: APIKey = Depends(_get_api_key)) -> APIKey:
    """Allow only admin keys."""
    if api_key.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )
    return api_key
