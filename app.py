"""
Phase A — FastAPI Inference Server
Endpoints: /predict, /health, /models/info
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from predictor import Predictor, PredictorError

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("phase_a")

# ── Global predictor instance ─────────────────────────────────────────────────
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release resources on shutdown."""
    global predictor
    log.info("Loading model …")
    predictor = Predictor()
    predictor.load()
    log.info("Model ready ✓")
    yield
    log.info("Shutting down …")
    if predictor:
        predictor.unload()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Phase A Inference API",
    description="Production-ready ML inference REST API (Phase A)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    inputs: Any = Field(..., description="Raw input passed to the model")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional inference parameters (e.g. temperature, top_k)",
    )

    model_config = {"json_schema_extra": {"example": {"inputs": "Hello, world!", "parameters": {}}}}


class PredictResponse(BaseModel):
    outputs: Any
    model_name: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_s: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    device: str
    parameters: dict[str, Any]


# ── Startup timestamp ─────────────────────────────────────────────────────────
_start_time = time.monotonic()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health():
    """Liveness + readiness check."""
    loaded = predictor is not None and predictor.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        device=predictor.device if loaded else "unknown",
        uptime_s=round(time.monotonic() - _start_time, 2),
    )


@app.get("/models/info", response_model=ModelInfoResponse, tags=["Operations"])
async def models_info():
    """Return metadata about the currently loaded model."""
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet.",
        )
    return ModelInfoResponse(
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        device=predictor.device,
        parameters=predictor.model_params,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """Run inference and return model outputs."""
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet.",
        )

    t0 = time.perf_counter()
    try:
        outputs = predictor.predict(request.inputs, **request.parameters)
    except PredictorError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        log.exception("Unexpected inference error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info("Inference completed in %.1f ms", latency_ms)

    return PredictResponse(
        outputs=outputs,
        model_name=predictor.model_name,
        latency_ms=latency_ms,
    )


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
