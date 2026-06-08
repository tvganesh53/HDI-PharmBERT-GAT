"""
app.py (Phase B) — FastAPI server with sync + async predict modes.

New endpoints:
    POST /predict           — sync  (waits for result, up to SYNC_TIMEOUT_S)
    POST /predict/async     — async (returns job_id immediately)
    GET  /jobs/{job_id}     — poll job status / result
    GET  /health            — liveness + queue depth
    GET  /models/info       — model metadata
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predictor import Predictor, PredictorError
from pipeline import NLPPipeline, PipelineError
from job_queue import JobQueue, QueueFullError
from schemas import (
    Job, JobResponse, JobStatus,
    PredictRequest, PredictResponse,
)
from worker import BatchWorker

log = logging.getLogger("phase_b")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

SYNC_TIMEOUT_S = float(os.getenv("SYNC_TIMEOUT_S", "30"))
_start_time = time.monotonic()

# ── Globals ───────────────────────────────────────────────────────────────────
predictor: Predictor | None = None
job_queue: JobQueue | None  = None
worker:    BatchWorker | None = None
pipeline:  NLPPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, job_queue, worker, pipeline

    log.info("Phase B startup …")
    predictor = Predictor()
    predictor.load()

    job_queue = JobQueue()
    pipeline  = NLPPipeline()
    worker    = BatchWorker(predictor, job_queue)
    await worker.start()

    log.info("Phase B ready ✓")
    yield

    log.info("Phase B shutdown …")
    if worker:
        await worker.stop()
    if predictor:
        predictor.unload()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Phase B — NLP Pipeline API",
    description="Async inference with preprocessing, batching & job queue",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Operations"])
async def health():
    loaded = predictor is not None and predictor.is_loaded
    depth  = await job_queue.depth() if job_queue else -1
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "device": predictor.device if loaded else "unknown",
        "queue_depth": depth,
        "uptime_s": round(time.monotonic() - _start_time, 2),
    }


@app.get("/models/info", tags=["Operations"])
async def models_info():
    _assert_ready()
    return {
        "model_name":    predictor.model_name,
        "model_version": predictor.model_version,
        "device":        predictor.device,
        "parameters":    predictor.model_params,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_sync(request: PredictRequest):
    """
    Synchronous predict — preprocesses, enqueues, waits for result.
    Blocks up to SYNC_TIMEOUT_S seconds (default 30 s).
    """
    _assert_ready()

    # Preprocess inline (fast path — no queue round-trip for pipeline errors)
    try:
        texts = pipeline.process(request.inputs, **request.parameters)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    job = Job(inputs=texts, parameters=request.parameters)
    try:
        await job_queue.submit(job)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    # Poll until done or timeout
    t0 = time.monotonic()
    while True:
        stored = await job_queue.get(job.job_id)
        if stored and stored.status == JobStatus.DONE:
            elapsed = (time.monotonic() - t0) * 1000
            return PredictResponse(
                outputs=stored.outputs,
                model_name=predictor.model_name,
                latency_ms=round(elapsed, 2),
                batch_size=len(texts),
            )
        if stored and stored.status == JobStatus.FAILED:
            raise HTTPException(status_code=500, detail=stored.error or "Inference failed.")
        if time.monotonic() - t0 > SYNC_TIMEOUT_S:
            raise HTTPException(status_code=504, detail="Inference timed out.")
        await asyncio.sleep(0.02)  # 20 ms poll interval


@app.post("/predict/async", response_model=JobResponse, status_code=202, tags=["Inference"])
async def predict_async(request: PredictRequest):
    """
    Async predict — returns a job_id immediately.
    Poll GET /jobs/{job_id} for the result.
    """
    _assert_ready()

    try:
        texts = pipeline.process(request.inputs, **request.parameters)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    job = Job(inputs=texts, parameters=request.parameters)
    try:
        await job_queue.submit(job)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    return job.to_response()


@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Inference"])
async def get_job(job_id: str):
    """Poll the status and result of an async job."""
    _assert_ready()
    job = await job_queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_response()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _assert_ready():
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Queue not initialised.")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
