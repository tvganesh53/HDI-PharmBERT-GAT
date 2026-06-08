"""
batcher.py — Dynamic batching engine.

Collects individual text items and groups them into batches either when:
  • the batch reaches MAX_BATCH_SIZE items, OR
  • BATCH_TIMEOUT_MS milliseconds have elapsed since the first item arrived.
"""

from __future__ import annotations
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

log = logging.getLogger("phase_b.batcher")

MAX_BATCH_SIZE  = int(os.getenv("MAX_BATCH_SIZE",   "32"))
BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", "50"))   # milliseconds


@dataclass
class BatchItem:
    job_id: str
    texts: list[str]        # preprocessed texts for this job
    parameters: dict[str, Any] = field(default_factory=dict)
    future: asyncio.Future  = field(default_factory=asyncio.Future)  # result carrier


class DynamicBatcher:
    """
    Accumulates BatchItems and flushes them to a callback when either
    the size or time threshold is crossed.

    Usage:
        batcher = DynamicBatcher(on_batch=my_async_fn)
        await batcher.start()
        future = await batcher.add(item)
        result = await future          # blocks until the batch is processed
        await batcher.stop()
    """

    def __init__(
        self,
        on_batch: Callable[[list[BatchItem]], Coroutine],
        max_size: int   = MAX_BATCH_SIZE,
        timeout_ms: int = BATCH_TIMEOUT_MS,
    ) -> None:
        self._on_batch   = on_batch
        self._max_size   = max_size
        self._timeout_s  = timeout_ms / 1000.0
        self._queue: list[BatchItem] = []
        self._lock       = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._running    = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._running = True
        self._flush_task = asyncio.create_task(self._timeout_flusher())
        log.info(
            "DynamicBatcher started (max_size=%d, timeout=%.0fms)",
            self._max_size, self._timeout_s * 1000,
        )

    async def stop(self) -> None:
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Flush remaining items
        async with self._lock:
            if self._queue:
                await self._flush()
        log.info("DynamicBatcher stopped.")

    # ── Public API ────────────────────────────────────────────────────────────
    async def add(self, item: BatchItem) -> asyncio.Future:
        """Enqueue an item and return a Future that resolves with the result."""
        async with self._lock:
            self._queue.append(item)
            log.debug("Batcher queue depth: %d", len(self._queue))
            if len(self._queue) >= self._max_size:
                await self._flush()
        return item.future

    # ── Internal ──────────────────────────────────────────────────────────────
    async def _timeout_flusher(self) -> None:
        """Background task: flush every BATCH_TIMEOUT_MS if queue is non-empty."""
        while self._running:
            await asyncio.sleep(self._timeout_s)
            async with self._lock:
                if self._queue:
                    log.debug("Timeout flush triggered (%d items)", len(self._queue))
                    await self._flush()

    async def _flush(self) -> None:
        """Must be called with self._lock held."""
        if not self._queue:
            return
        batch, self._queue = self._queue[:], []
        log.info("Flushing batch of %d item(s)", len(batch))
        asyncio.create_task(self._dispatch(batch))

    async def _dispatch(self, batch: list[BatchItem]) -> None:
        """Run the batch callback and distribute results to waiting futures."""
        try:
            await self._on_batch(batch)
        except Exception as exc:
            log.exception("Batch dispatch error")
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)

    # ── Stats ─────────────────────────────────────────────────────────────────
    @property
    def queue_depth(self) -> int:
        return len(self._queue)
