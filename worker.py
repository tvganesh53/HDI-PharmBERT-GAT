"""
worker.py — Async batch worker.

Pulls jobs from the JobQueue, groups them via DynamicBatcher,
runs inference through the Predictor, and writes results back.
"""

from __future__ import annotations
import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

from batcher import BatchItem, DynamicBatcher
from pipeline import NLPPipeline, PipelineError
from job_queue import JobQueue, QueueFullError
from schemas import Job, JobStatus

if TYPE_CHECKING:
    from predictor import Predictor

log = logging.getLogger("phase_b.worker")

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))


class BatchWorker:
    """
    Lifecycle:
        worker = BatchWorker(predictor, queue)
        await worker.start()   # called at app startup
        ...
        await worker.stop()    # called at app shutdown
    """

    def __init__(self, predictor: "Predictor", queue: JobQueue) -> None:
        self._predictor = predictor
        self._queue = queue
        self._pipeline = NLPPipeline()
        self._batcher = DynamicBatcher(on_batch=self._run_batch)
        self._tasks: list[asyncio.Task] = []
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._running = True
        await self._batcher.start()
        for i in range(NUM_WORKERS):
            task = asyncio.create_task(self._consume_loop(worker_id=i))
            self._tasks.append(task)
        log.info("BatchWorker started (%d consumer(s))", NUM_WORKERS)

    async def stop(self) -> None:
        log.info("BatchWorker shutting down …")
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._batcher.stop()
        log.info("BatchWorker stopped.")

    # ── Consumer loop ─────────────────────────────────────────────────────────
    async def _consume_loop(self, worker_id: int) -> None:
        """Continuously pull jobs from the queue and hand to batcher."""
        log.info("Worker-%d ready", worker_id)
        while self._running:
            try:
                job = await asyncio.wait_for(self._queue.next(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Worker-%d: error dequeuing", worker_id)
                continue

            # Preprocess
            try:
                texts = self._pipeline.process(job.inputs, **job.parameters)
            except PipelineError as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.completed_at = time.time()
                await self._queue.update(job)
                self._queue.task_done()
                continue

            # Hand to batcher
            loop = asyncio.get_event_loop()
            future: asyncio.Future = loop.create_future()
            item = BatchItem(
                job_id=job.job_id,
                texts=texts,
                parameters=job.parameters,
                future=future,
            )

            # Update status
            job.status = JobStatus.PROCESSING
            await self._queue.update(job)

            await self._batcher.add(item)
            self._queue.task_done()

            # Await result (non-blocking — batcher resolves the future)
            asyncio.create_task(self._await_and_persist(job, future))

    # ── Result persistence ────────────────────────────────────────────────────
    async def _await_and_persist(self, job: Job, future: asyncio.Future) -> None:
        try:
            result = await future
            job.status = JobStatus.DONE
            job.outputs = result
        except Exception as exc:
            log.exception("Job %s failed", job.job_id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
        finally:
            job.completed_at = time.time()
            await self._queue.update(job)

    # ── Batch execution (called by DynamicBatcher) ────────────────────────────
    async def _run_batch(self, batch: list[BatchItem]) -> None:
        """Run inference on a collected batch and resolve each future."""
        # Flatten all texts, tracking which job each text belongs to
        all_texts: list[str] = []
        offsets: list[tuple[int, int]] = []   # (start, end) index per item
        for item in batch:
            start = len(all_texts)
            all_texts.extend(item.texts)
            offsets.append((start, start + len(item.texts)))

        log.info("Running inference on batch: %d job(s), %d text(s)", len(batch), len(all_texts))
        t0 = time.perf_counter()

        try:
            # Run in threadpool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                self._predictor.predict,
                all_texts,
            )
        except Exception as exc:
            # Fail all futures in this batch
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info("Batch inference done in %.1f ms", elapsed_ms)

        # Distribute results back to each job's future
        for item, (start, end) in zip(batch, offsets):
            if not item.future.done():
                result = outputs[start:end] if isinstance(outputs, list) else outputs
                item.future.set_result(result)
