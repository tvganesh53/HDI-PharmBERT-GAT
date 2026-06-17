"""
queue.py — Async job queue.

Default backend: in-memory (asyncio.Queue + dict store).
Redis backend: set QUEUE_BACKEND=redis and REDIS_URL=redis://localhost:6379/0
"""

from __future__ import annotations
import asyncio
import logging
import os
import time
from typing import Protocol

from schemas import Job, JobStatus

log = logging.getLogger("phase_b.queue")

QUEUE_BACKEND = os.getenv("QUEUE_BACKEND", "memory")   # "memory" | "redis"
QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "500"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ── Backend protocol ──────────────────────────────────────────────────────────
class QueueBackend(Protocol):
    async def enqueue(self, job: Job) -> None: ...
    async def dequeue(self) -> Job: ...
    async def get(self, job_id: str) -> Job | None: ...
    async def update(self, job: Job) -> None: ...
    async def depth(self) -> int: ...


# ── In-memory backend ─────────────────────────────────────────────────────────
class InMemoryBackend:
    def __init__(self, maxsize: int = QUEUE_MAX_SIZE) -> None:
        self._q: asyncio.Queue[Job] = asyncio.Queue(maxsize=maxsize)
        self._store: dict[str, Job] = {}

    async def enqueue(self, job: Job) -> None:
        if self._q.full():
            raise QueueFullError(f"Queue is full (max {self._q.maxsize} jobs).")
        self._store[job.job_id] = job
        await self._q.put(job)

    async def dequeue(self) -> Job:
        return await self._q.get()

    async def get(self, job_id: str) -> Job | None:
        return self._store.get(job_id)

    async def update(self, job: Job) -> None:
        self._store[job.job_id] = job

    async def depth(self) -> int:
        return self._q.qsize()

    def task_done(self) -> None:
        self._q.task_done()


# ── Redis backend (optional) ──────────────────────────────────────────────────
class RedisBackend:
    """
    Drop-in Redis backend.  Requires: pip install redis[asyncio]

    Jobs are stored as JSON strings under key  job:<job_id>
    Pending job IDs live in a Redis List       phase_b:queue
    """

    def __init__(self, url: str = REDIS_URL, maxsize: int = QUEUE_MAX_SIZE) -> None:
        import redis.asyncio as aioredis  # type: ignore
        self._r = aioredis.from_url(url, decode_responses=True)
        self._maxsize = maxsize
        self._LIST_KEY = "phase_b:queue"

    async def enqueue(self, job: Job) -> None:
        depth = await self._r.llen(self._LIST_KEY)
        if depth >= self._maxsize:
            raise QueueFullError(f"Redis queue full (max {self._maxsize}).")
        await self._r.set(f"job:{job.job_id}", job.model_dump_json(), ex=3600)
        await self._r.rpush(self._LIST_KEY, job.job_id)

    async def dequeue(self) -> Job:
        while True:
            result = await self._r.blpop(self._LIST_KEY, timeout=1)
            if result:
                _, job_id = result
                raw = await self._r.get(f"job:{job_id}")
                if raw:
                    return Job.model_validate_json(raw)

    async def get(self, job_id: str) -> Job | None:
        raw = await self._r.get(f"job:{job_id}")
        return Job.model_validate_json(raw) if raw else None

    async def update(self, job: Job) -> None:
        await self._r.set(f"job:{job.job_id}", job.model_dump_json(), ex=3600)

    async def depth(self) -> int:
        return await self._r.llen(self._LIST_KEY)


# ── Errors ────────────────────────────────────────────────────────────────────
class QueueFullError(Exception):
    pass


# ── High-level JobQueue facade ────────────────────────────────────────────────
class JobQueue:
    """
    High-level facade used by app.py and worker.py.
    Selects the backend based on QUEUE_BACKEND env var.
    """

    def __init__(self) -> None:
        if QUEUE_BACKEND == "redis":
            log.info("Using Redis queue backend (%s)", REDIS_URL)
            self._backend: QueueBackend = RedisBackend()
        else:
            log.info("Using in-memory queue backend (max=%d)", QUEUE_MAX_SIZE)
            self._backend = InMemoryBackend()

    async def submit(self, job: Job) -> Job:
        """Add a job to the queue. Raises QueueFullError if at capacity."""
        await self._backend.enqueue(job)
        log.debug("Job %s queued", job.job_id)
        return job

    async def next(self) -> Job:
        """Block until a job is available, then return it."""
        return await self._backend.dequeue()

    async def get(self, job_id: str) -> Job | None:
        return await self._backend.get(job_id)

    async def update(self, job: Job) -> None:
        await self._backend.update(job)

    async def depth(self) -> int:
        return await self._backend.depth()

    def task_done(self) -> None:
        if isinstance(self._backend, InMemoryBackend):
            self._backend.task_done()
