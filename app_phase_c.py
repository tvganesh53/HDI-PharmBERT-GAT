"""
app_phase_c.py — Phase C FastAPI server.

New endpoints on top of Phase B:
    POST /classify          — sync classification, all labels ranked
    POST /classify/async    — async classification, returns job_id
    GET  /classify/labels   — list configured labels
    GET  /jobs/{job_id}     — poll async job (inherited from Phase B)
    GET  /health            — liveness + queue depth
    GET  /models/info       — model + label metadata
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predictor_c import Predictor, PredictorError, CLASSIFIER_LABELS
from pipeline import NLPPipeline, PipelineError
from job_queue import JobQueue, QueueFullError
from schemas import Job, JobStatus
from schemas_c import ClassifyRequest, ClassifyResponse, ClassificationResult
from worker import BatchWorker

log = logging.getLogger("phase_c")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

SYNC_TIMEOUT_S = float(os.getenv("SYNC_TIMEOUT_S", "30"))
_start_time    = time.monotonic()

# ── Globals ───────────────────────────────────────────────────────────────────
predictor: Predictor | None    = None
job_queue: JobQueue | None     = None
worker:    BatchWorker | None  = None
pipeline:  NLPPipeline | None  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, job_queue, worker, pipeline

    log.info("Phase C startup …")
    predictor = Predictor()
    predictor.load()

    job_queue = JobQueue()
    pipeline  = NLPPipeline()
    worker    = BatchWorker(predictor, job_queue)
    await worker.start()

    log.info("Phase C ready ✓  labels=%s", CLASSIFIER_LABELS)
    yield

    log.info("Phase C shutdown …")
    if worker:    await worker.stop()
    if predictor: predictor.unload()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Phase C — NLP Classification API",
    description="Anthropic Claude-powered text classifier with custom labels, ranked scores, and async job queue.",
    version="3.0.0",
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
        "status":       "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "provider":     "anthropic",
        "queue_depth":  depth,
        "label_count":  len(CLASSIFIER_LABELS),
        "uptime_s":     round(time.monotonic() - _start_time, 2),
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


@app.get("/classify/labels", tags=["Classification"])
async def get_labels():
    """Return the currently configured label set."""
    return {
        "labels":  CLASSIFIER_LABELS,
        "count":   len(CLASSIFIER_LABELS),
        "context": os.getenv("CLASSIFIER_CONTEXT", "customer support messages"),
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify_sync(request: ClassifyRequest):
    """
    Sync classification — sends text through pipeline → Claude → returns
    all labels ranked by confidence score.
    """
    _assert_ready()

    # Preprocess
    try:
        texts = pipeline.process(request.inputs, **request.parameters)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Build kwargs for predictor
    kwargs = dict(request.parameters)
    if request.labels:
        kwargs["labels"] = request.labels
    if request.context:
        kwargs["context"] = request.context

    # Enqueue and wait
    job = Job(inputs=texts, parameters=kwargs)
    try:
        await job_queue.submit(job)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    t0 = time.monotonic()
    while True:
        stored = await job_queue.get(job.job_id)
        if stored and stored.status == JobStatus.DONE:
            elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
            raw        = stored.outputs
            results    = _normalise_results(raw)
            return ClassifyResponse(
                results=results,
                model_name=predictor.model_name,
                latency_ms=elapsed_ms,
                label_set=request.labels or CLASSIFIER_LABELS,
            )
        if stored and stored.status == JobStatus.FAILED:
            raise HTTPException(status_code=500, detail=stored.error or "Classification failed.")
        if time.monotonic() - t0 > SYNC_TIMEOUT_S:
            raise HTTPException(status_code=504, detail="Classification timed out.")
        await asyncio.sleep(0.02)


@app.post("/classify/async", tags=["Classification"], status_code=202)
async def classify_async(request: ClassifyRequest):
    """
    Async classification — returns job_id immediately.
    Poll GET /jobs/{job_id} for results.
    """
    _assert_ready()

    try:
        texts = pipeline.process(request.inputs, **request.parameters)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    kwargs = dict(request.parameters)
    if request.labels:  kwargs["labels"]  = request.labels
    if request.context: kwargs["context"] = request.context

    job = Job(inputs=texts, parameters=kwargs)
    try:
        await job_queue.submit(job)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    return job.to_response()


@app.get("/jobs/{job_id}", tags=["Classification"])
async def get_job(job_id: str):
    """Poll status and result of an async classification job."""
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


def _normalise_results(raw: Any) -> list[ClassificationResult]:
    """Convert predictor output (dict or list[dict]) to ClassificationResult list."""
    if isinstance(raw, dict):
        raw = [raw]
    results = []
    for item in raw:
        results.append(ClassificationResult(
            text=item.get("text", ""),
            top_label=item.get("top_label", ""),
            top_score=item.get("top_score", 0.0),
            classifications=item.get("classifications", []),
        ))
    return results


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
