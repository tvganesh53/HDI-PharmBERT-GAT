"""
rate_limiter.py — Per-API-key rate limiting for Phase E.

Default: 60 requests per minute per key.
Override via env: RATE_LIMIT_PER_MINUTE
"""

from __future__ import annotations
import logging
import os
import time
from collections import defaultdict

from fastapi import HTTPException, Request, status

from api_keys import APIKey

log = logging.getLogger("phase_e.rate_limiter")

RATE_LIMIT     = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
WINDOW_SECONDS = 60


# ── In-memory sliding window store ───────────────────────────────────────────
# {key_id: [timestamp, timestamp, ...]}
_request_log: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(api_key: APIKey) -> None:
    """
    Raise HTTP 429 if the key has exceeded RATE_LIMIT requests
    in the last WINDOW_SECONDS seconds.
    """
    now    = time.time()
    cutoff = now - WINDOW_SECONDS

    # Keep only recent timestamps
    recent = [t for t in _request_log[api_key.key_id] if t > cutoff]
    recent.append(now)
    _request_log[api_key.key_id] = recent

    count = len(recent)
    if count > RATE_LIMIT:
        retry_after = int(WINDOW_SECONDS - (now - recent[0]))
        log.warning(
            "Rate limit exceeded for key '%s' (%d/%d requests)",
            api_key.name, count, RATE_LIMIT,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {RATE_LIMIT} requests/minute. "
                   f"Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    log.debug("Rate check OK for '%s': %d/%d", api_key.name, count, RATE_LIMIT)


def get_usage(key_id: str) -> dict:
    """Return current usage stats for a key."""
    now    = time.time()
    cutoff = now - WINDOW_SECONDS
    recent = [t for t in _request_log.get(key_id, []) if t > cutoff]
    return {
        "requests_last_minute": len(recent),
        "limit_per_minute":     RATE_LIMIT,
        "remaining":            max(0, RATE_LIMIT - len(recent)),
    }
