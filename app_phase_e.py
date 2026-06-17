"""
app_phase_e.py — Phase E FastAPI server.

Adds on top of Phase D:
    POST /admin/keys            — create a new API key (admin only)
    GET  /admin/keys            — list all keys (admin only)
    DELETE /admin/keys/{key_id} — revoke a key (admin only)
    GET  /me                    — show current key info

Protected endpoints (require X-API-Key header):
    POST /classify
    POST /classify/async
    GET  /jobs/{job_id}
    GET  /stats
    GET  /dashboard
    GET  /logs/*

Public endpoints (no key needed):
    GET  /health
    GET  /docs
    GET  /metrics
"""

from __future__ import annotations
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from predictor_c import Predictor, PredictorError, CLASSIFIER_LABELS
from pipeline import NLPPipeline, PipelineError
from job_queue import JobQueue, QueueFullError
from schemas import Job, JobStatus
from schemas_c import ClassifyRequest, ClassifyResponse, ClassificationResult
from worker import BatchWorker
from metrics import record_classification, record_error, update_queue_depth, set_active_workers
from logger import log_classification, log_error, read_recent_classifications, read_recent_errors, get_stats
from api_keys import key_store, ensure_admin_key, APIKey
from auth import require_user, require_admin
from rate_limiter import check_rate_limit, get_usage

log = logging.getLogger("phase_e")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

SYNC_TIMEOUT_S = float(os.getenv("SYNC_TIMEOUT_S", "30"))
NUM_WORKERS    = int(os.getenv("NUM_WORKERS", "2"))
_start_time    = time.monotonic()

predictor: Predictor | None   = None
job_queue: JobQueue | None    = None
worker:    BatchWorker | None = None
pipeline:  NLPPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, job_queue, worker, pipeline

    log.info("Phase E startup …")

    # Create default admin key if none exist
    ensure_admin_key()

    predictor = Predictor()
    predictor.load()

    job_queue = JobQueue()
    pipeline  = NLPPipeline()
    worker    = BatchWorker(predictor, job_queue)
    await worker.start()
    set_active_workers(NUM_WORKERS)

    async def _queue_monitor():
        while True:
            try:
                depth = await job_queue.depth()
                update_queue_depth(depth)
            except Exception:
                pass
            await asyncio.sleep(5)

    monitor_task = asyncio.create_task(_queue_monitor())
    log.info("Phase E ready ✓  (keys=%d)", key_store.count())
    yield

    monitor_task.cancel()
    if worker:    await worker.stop()
    if predictor: predictor.unload()


app = FastAPI(
    title="Phase E — Secured NLP Classification API",
    description="Production-ready NLP classifier with API key auth, rate limiting, and monitoring.",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Public endpoints (no auth needed) ────────────────────────────────────────
@app.get("/health", tags=["Operations"])
async def health():
    loaded = predictor is not None and predictor.is_loaded
    depth  = await job_queue.depth() if job_queue else -1
    return {
        "status":       "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "queue_depth":  depth,
        "uptime_s":     round(time.monotonic() - _start_time, 2),
        "total_keys":   key_store.count(),
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Auth info endpoint ────────────────────────────────────────────────────────
@app.get("/me", tags=["Auth"])
async def me(api_key: APIKey = Depends(require_user)):
    """Show info about the current API key."""
    usage = get_usage(api_key.key_id)
    return {
        **api_key.to_public_dict(),
        "usage": usage,
    }


# ── Admin endpoints ───────────────────────────────────────────────────────────
@app.post("/admin/keys", tags=["Admin"])
async def create_key(
    name: str,
    role: str = "user",
    api_key: APIKey = Depends(require_admin),
):
    """Create a new API key. Admin only."""
    if role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'.")
    raw_key, new_key = key_store.create(name=name, role=role)
    return {
        "message":  "API key created. Save the raw_key — it will NOT be shown again.",
        "raw_key":  raw_key,
        "key_info": new_key.to_public_dict(),
    }


@app.get("/admin/keys", tags=["Admin"])
async def list_keys(api_key: APIKey = Depends(require_admin)):
    """List all API keys. Admin only."""
    return {"keys": key_store.list_keys(), "total": key_store.count()}


@app.delete("/admin/keys/{key_id}", tags=["Admin"])
async def revoke_key(key_id: str, api_key: APIKey = Depends(require_admin)):
    """Revoke an API key. Admin only."""
    success = key_store.revoke(key_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Key '{key_id}' not found.")
    return {"message": f"Key '{key_id}' revoked successfully."}


# ── Protected monitoring endpoints ────────────────────────────────────────────
@app.get("/stats", tags=["Monitoring"])
async def stats(api_key: APIKey = Depends(require_user)):
    check_rate_limit(api_key)
    depth = await job_queue.depth() if job_queue else 0
    s = get_stats()
    s["queue_depth"] = depth
    s["uptime_s"]    = round(time.monotonic() - _start_time, 2)
    s["model"]       = predictor.model_name if predictor else "unknown"
    return s


@app.get("/dashboard", response_class=HTMLResponse, tags=["Monitoring"])
async def dashboard():
    return HTMLResponse(content=_build_dashboard_html())

@app.get("/logs/classifications", tags=["Monitoring"])
async def get_classification_logs(
    limit: int = 50,
    api_key: APIKey = Depends(require_user),
):
    check_rate_limit(api_key)
    return {"records": read_recent_classifications(limit), "limit": limit}


@app.get("/logs/errors", tags=["Monitoring"])
async def get_error_logs(
    limit: int = 20,
    api_key: APIKey = Depends(require_user),
):
    check_rate_limit(api_key)
    return {"records": read_recent_errors(limit), "limit": limit}


@app.get("/models/info", tags=["Operations"])
async def models_info(api_key: APIKey = Depends(require_user)):
    _assert_ready()
    return {
        "model_name":    predictor.model_name,
        "model_version": predictor.model_version,
        "device":        predictor.device,
        "parameters":    predictor.model_params,
    }


@app.get("/classify/labels", tags=["Classification"])
async def get_labels(api_key: APIKey = Depends(require_user)):
    return {
        "labels":  CLASSIFIER_LABELS,
        "count":   len(CLASSIFIER_LABELS),
        "context": os.getenv("CLASSIFIER_CONTEXT", "customer support messages"),
    }


# ── Protected classify endpoints ──────────────────────────────────────────────
@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify_sync(
    request: ClassifyRequest,
    api_key: APIKey = Depends(require_user),
):
    _assert_ready()
    check_rate_limit(api_key)

    try:
        texts = pipeline.process(request.inputs, **request.parameters)
    except PipelineError as exc:
        record_error("pipeline_error")
        raise HTTPException(status_code=422, detail=str(exc))

    kwargs = dict(request.parameters)
    if request.labels:   kwargs["labels"]  = request.labels
    if request.context:  kwargs["context"] = request.context

    job = Job(inputs=texts, parameters=kwargs)
    try:
        await job_queue.submit(job)
    except QueueFullError as exc:
        record_error("queue_full")
        raise HTTPException(status_code=429, detail=str(exc))

    t0 = time.monotonic()
    while True:
        stored = await job_queue.get(job.job_id)
        if stored and stored.status == JobStatus.DONE:
            elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
            results    = _normalise_results(stored.outputs)
            for result in results:
                record_classification(top_label=result.top_label, latency_ms=elapsed_ms, batch_size=len(texts))
                log_classification(
                    job_id=job.job_id, input_text=result.text,
                    top_label=result.top_label, top_score=result.top_score,
                    all_labels=[c.model_dump() for c in result.classifications],
                    latency_ms=elapsed_ms, batch_size=len(texts),
                )
            return ClassifyResponse(
                results=results,
                model_name=predictor.model_name,
                latency_ms=elapsed_ms,
                label_set=request.labels or CLASSIFIER_LABELS,
            )
        if stored and stored.status == JobStatus.FAILED:
            record_error("inference_error")
            raise HTTPException(status_code=500, detail=stored.error or "Failed.")
        if time.monotonic() - t0 > SYNC_TIMEOUT_S:
            record_error("timeout")
            raise HTTPException(status_code=504, detail="Timed out.")
        await asyncio.sleep(0.02)


@app.post("/classify/async", tags=["Classification"], status_code=202)
async def classify_async(
    request: ClassifyRequest,
    api_key: APIKey = Depends(require_user),
):
    _assert_ready()
    check_rate_limit(api_key)

    try:
        texts = pipeline.process(request.inputs, **request.parameters)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    kwargs = dict(request.parameters)
    if request.labels:   kwargs["labels"]  = request.labels
    if request.context:  kwargs["context"] = request.context

    job = Job(inputs=texts, parameters=kwargs)
    try:
        await job_queue.submit(job)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    return job.to_response()


@app.get("/jobs/{job_id}", tags=["Classification"])
async def get_job(job_id: str, api_key: APIKey = Depends(require_user)):
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
    if isinstance(raw, dict):
        raw = [raw]
    return [
        ClassificationResult(
            text=item.get("text", ""),
            top_label=item.get("top_label", ""),
            top_score=item.get("top_score", 0.0),
            classifications=item.get("classifications", []),
        )
        for item in raw
    ]


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Dashboard HTML ────────────────────────────────────────────────────────────
def _build_dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Phase E — Secured Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f172a; color: #e2e8f0; min-height: 100vh; }
    header { background: #1e293b; padding: 20px 32px; border-bottom: 1px solid #334155;
             display: flex; align-items: center; justify-content: space-between; }
    header h1 { font-size: 1.4rem; color: #f1f5f9; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; padding: 28px 32px; }
    .card { background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; }
    .card h3 { font-size: 0.75rem; text-transform: uppercase; color: #64748b; margin-bottom: 12px; }
    .card .value { font-size: 2rem; font-weight: 700; color: #f1f5f9; }
    .card .sub { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
    .accent-green  { border-top: 3px solid #22c55e; }
    .accent-blue   { border-top: 3px solid #3b82f6; }
    .accent-amber  { border-top: 3px solid #f59e0b; }
    .accent-purple { border-top: 3px solid #a855f7; }
    .accent-red    { border-top: 3px solid #ef4444; }
    .section { padding: 0 32px 28px; }
    .section h2 { font-size: 1rem; color: #94a3b8; margin-bottom: 16px;
                  padding-bottom: 8px; border-bottom: 1px solid #334155; }
    .label-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
    .label-name { font-size: 0.85rem; color: #cbd5e1; width: 180px; flex-shrink: 0; }
    .bar-wrap { flex: 1; background: #334155; border-radius: 4px; height: 20px; overflow: hidden; }
    .bar-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6);
                border-radius: 4px; transition: width 0.5s ease; }
    .bar-count { font-size: 0.8rem; color: #94a3b8; width: 40px; text-align: right; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th { text-align: left; padding: 10px 14px; background: #0f172a; color: #64748b;
         font-weight: 600; text-transform: uppercase; font-size: 0.72rem; }
    td { padding: 10px 14px; border-bottom: 1px solid #1e293b; color: #cbd5e1;
         max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 600; }
    .badge-blue { background: #1d4ed8; color: #bfdbfe; }
    .badge-red  { background: #991b1b; color: #fecaca; }
    .key-input { background: #1e293b; border: 1px solid #334155; color: #e2e8f0;
                 padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; width: 100%; margin-bottom: 8px; }
    .btn { background: #3b82f6; color: white; border: none; padding: 8px 16px;
           border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
    .btn:hover { background: #2563eb; }
    .auth-box { background: #1e293b; border: 1px solid #f59e0b; border-radius: 12px;
                padding: 24px; margin: 28px 32px 0; }
    .auth-box h3 { color: #f59e0b; margin-bottom: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>🔐 Phase E — Secured Dashboard</h1>
    <span style="font-size:0.85rem;color:#64748b">Auto-refreshes every 5s | <span id="last-update">—</span></span>
  </header>

  <div class="auth-box">
    <h3>Enter your API Key to view live data</h3>
    <input class="key-input" id="api-key-input" type="password" placeholder="sk-..." />
    <button class="btn" onclick="saveKey()">Save Key</button>
    <span id="key-status" style="margin-left:12px;font-size:0.8rem;color:#64748b"></span>
  </div>

  <div class="grid">
    <div class="card accent-green"><h3>Total Classifications</h3><div class="value" id="total">—</div><div class="sub">all time</div></div>
    <div class="card accent-blue"><h3>Avg Latency</h3><div class="value" id="latency">—</div><div class="sub">milliseconds</div></div>
    <div class="card accent-amber"><h3>Queue Depth</h3><div class="value" id="queue">—</div><div class="sub">jobs waiting</div></div>
    <div class="card accent-purple"><h3>Avg Confidence</h3><div class="value" id="score">—</div><div class="sub">top label score</div></div>
    <div class="card accent-red"><h3>Uptime</h3><div class="value" id="uptime">—</div><div class="sub">seconds</div></div>
  </div>

  <div class="section">
    <h2>Label Distribution</h2>
    <div id="label-bars"><p style="color:#64748b">Enter your API key above to load data</p></div>
  </div>

  <div class="section">
    <h2>Recent Classifications</h2>
    <table>
      <thead><tr><th>Time</th><th>Input</th><th>Top Label</th><th>Score</th><th>Latency</th></tr></thead>
      <tbody id="recent-rows"><tr><td colspan="5" style="color:#64748b">Enter API key to load</td></tr></tbody>
    </table>
  </div>

<script>
let API_KEY = localStorage.getItem('phase_e_key') || '';
if (API_KEY) {
  document.getElementById('api-key-input').value = API_KEY;
  document.getElementById('key-status').textContent = '✓ Key loaded';
}

function saveKey() {
  API_KEY = document.getElementById('api-key-input').value.trim();
  localStorage.setItem('phase_e_key', API_KEY);
  document.getElementById('key-status').textContent = '✓ Key saved';
  refresh();
}

async function fetchJSON(url) {
  try {
    const r = await fetch(url, { headers: { 'X-API-Key': API_KEY } });
    if (!r.ok) return null;
    return r.json();
  } catch { return null; }
}

function timeAgo(iso) {
  const s = Math.floor((Date.now() - new Date(iso)) / 1000);
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  return Math.floor(s/3600) + 'h ago';
}

async function refresh() {
  if (!API_KEY) return;
  const [stats, logs] = await Promise.all([
    fetchJSON('/stats'),
    fetchJSON('/logs/classifications?limit=10'),
  ]);

  if (stats) {
    document.getElementById('total').textContent   = stats.total ?? '—';
    document.getElementById('latency').textContent = stats.avg_latency_ms ? stats.avg_latency_ms + 'ms' : '—';
    document.getElementById('queue').textContent   = stats.queue_depth ?? '—';
    document.getElementById('score').textContent   = stats.avg_score ? (stats.avg_score*100).toFixed(0)+'%' : '—';
    document.getElementById('uptime').textContent  = stats.uptime_s ?? '—';

    const counts = stats.label_counts || {};
    const maxVal = Math.max(...Object.values(counts), 1);
    document.getElementById('label-bars').innerHTML = Object.entries(counts)
      .sort((a,b) => b[1]-a[1])
      .map(([label, count]) => `
        <div class="label-bar">
          <div class="label-name">${label}</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:${(count/maxVal*100).toFixed(1)}%"></div></div>
          <div class="bar-count">${count}</div>
        </div>`).join('') || '<p style="color:#64748b">No classifications yet</p>';
  }

  if (logs && logs.records.length) {
    document.getElementById('recent-rows').innerHTML = logs.records.map(r => `
      <tr>
        <td>${timeAgo(r.timestamp)}</td>
        <td title="${r.input}">${r.input.substring(0,60)}${r.input.length>60?'…':''}</td>
        <td><span class="badge badge-blue">${r.top_label}</span></td>
        <td>${(r.top_score*100).toFixed(0)}%</td>
        <td>${r.latency_ms}ms</td>
      </tr>`).join('');
  }

  document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""
