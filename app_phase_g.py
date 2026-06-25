"""
app_phase_g.py — Phase G NLP Classifier API
Full stack: MySQL persistence, analytics, auth, HDI pipeline (PharmBERT + PharmFusion)
"""
from __future__ import annotations
 
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse, FileResponse
 
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
 
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
 
from auth import require_admin, require_user
from database import close_db, create_tables, get_db
 
try:
    from metrics import (
        CLASSIFICATIONS_TOTAL,
        CLASSIFICATION_ERRORS,
        CLASSIFICATION_LATENCY,
        QUEUE_DEPTH,
        generate_metrics,
    )
except ImportError:
    from unittest.mock import MagicMock
    CLASSIFICATIONS_TOTAL = MagicMock()
    CLASSIFICATION_ERRORS = MagicMock()
    CLASSIFICATION_LATENCY = MagicMock()
    QUEUE_DEPTH = MagicMock()
    def generate_metrics(): return ""
 
from repository import (
    get_daily_trends,
    get_label_distribution,
    get_latency_percentiles,
    get_recent_classifications,
    get_summary,
    get_top_inputs,
    refresh_analytics_daily,
    save_classification,
    upsert_api_key_log,
)
 
log = logging.getLogger("phase_g")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
 
# ── Predictor ─────────────────────────────────────────────────────────────────
predictor: Any = None
 
def _load_predictor() -> Any:
    try:
        from pipeline_hdi import hdi_pipeline
        hdi_pipeline.load()
        log.info("HDIPipeline loaded (PharmBERT P9 + PharmFusion P8)")
        return hdi_pipeline
    except Exception as exc:
        log.warning("HDIPipeline failed (%s), falling back to PharmBERT only.", exc)
        try:
            from predictor_pharmbert import pharmbert_predictor
            pharmbert_predictor.load()
            return pharmbert_predictor
        except Exception as exc2:
            log.warning("PharmBERT also failed (%s) — stub mode.", exc2)
            return None
 
# ── Analytics background refresh ─────────────────────────────────────────────
async def _analytics_refresh_loop() -> None:
    while True:
        await asyncio.sleep(3600)
        try:
            from database import AsyncSessionLocal
            async with AsyncSessionLocal() as db:
                await refresh_analytics_daily(db)
                await db.commit()
        except Exception as exc:
            log.warning("Analytics refresh error: %s", exc)
 
# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, _analytics_task
    log.info("Phase G startup …")
    await create_tables()
    predictor = _load_predictor()
    _analytics_task = asyncio.create_task(_analytics_refresh_loop())
    # Load permanent admin key — write hash into keys.json so
    # validate() (which re-reads disk on every request) always finds it.
    permanent_key = os.getenv("PERMANENT_ADMIN_KEY")
    if permanent_key:
        try:
            import hashlib as _hl, time as _time
            from api_keys import key_store, APIKey
            perm_hash = _hl.sha256(permanent_key.encode()).hexdigest()
            already = any(k.key_hash == perm_hash for k in key_store._keys.values())
            if not already:
                perm_obj = APIKey(
                    key_id="kid-permanent-admin",
                    key_hash=perm_hash,
                    name="hf-admin",
                    role="admin",
                    created_at=_time.time(),
                    is_active=True,
                )
                key_store._keys["kid-permanent-admin"] = perm_obj
                key_store._save()
                log.info("Permanent admin key written to keys.json.")
            else:
                log.info("Permanent admin key already present.")
        except Exception as exc:
            log.warning("Could not load permanent key: %s", exc)
    yield
    _analytics_task.cancel()
    await close_db()
    log.info("Phase G shutdown.")
 
_analytics_task: asyncio.Task | None = None
 
# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NLP Classifier API",
    description="Herb-Drug Interaction classifier — PharmBERT P9 + PharmFusion P8",
    version="9.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Schemas ───────────────────────────────────────────────────────────────────
class LabelScore(BaseModel):
    label: str
    score: float
    reasoning: str | None = None
 
class InteractionType(BaseModel):
    top_label: str
    top_score: float
    all_scores: list[LabelScore] = []
    node_id_used: str | None = None
    gat_lookup: bool = False
 
class ClassifyResult(BaseModel):
    job_id: str
    text: str
    top_label: str
    top_score: float
    classifications: list[LabelScore] = []
    interaction_type: InteractionType | None = None
    summary: str | None = None
 
class ClassifyResponse(BaseModel):
    results: list[ClassifyResult]
    model_name: str
    latency_ms: float
 
class ClassifyRequest(BaseModel):
    inputs: str | list[str] = Field(..., description="One or more texts to classify.")
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_key_name(api_key) -> str:
    for attr in ("key_id", "label", "api_key_name"):
        val = getattr(api_key, attr, None)
        if val and not callable(val):
            return str(val)
    return "unknown"
 
# ── Classify ──────────────────────────────────────────────────────────────────
@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify(
    req: ClassifyRequest,
    request: Request,
    api_key=Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    if predictor is None:
        raise HTTPException(503, "Model not loaded.")
 
    texts = req.inputs if isinstance(req.inputs, list) else [req.inputs]
    if not texts or any(not t.strip() for t in texts):
        raise HTTPException(422, "Input text must not be empty.")
 
    t0 = time.perf_counter()
    try:
        raw = predictor.predict(texts)
    except Exception as exc:
        CLASSIFICATION_ERRORS.labels(error_type=type(exc).__name__).inc()
        log.error("Predictor error: %s", exc)
        raise HTTPException(500, f"Classifier error: {exc}") from exc
 
    latency_ms = (time.perf_counter() - t0) * 1000
    outputs = raw.get("outputs", [{}] * len(texts))
    results: list[ClassifyResult] = []
 
    for idx, text_input in enumerate(texts):
        out = outputs[idx] if idx < len(outputs) else {}
 
        # HDIPipeline returns severity + interaction_type; PharmBERT returns classifications
        itype_obj = None
        summary_str = None
        if "severity" in out:
            # HDIPipeline format
            sev         = out["severity"]
            ranked      = sev.get("all_scores", [])
            top         = {"label": sev["top_label"], "score": sev["top_score"], "reasoning": out.get("summary", "")}
            summary_str = out.get("summary")
            # Build interaction_type object
            itype = out.get("interaction_type")
            if itype:
                itype_obj = InteractionType(
                    top_label    = itype.get("top_label", "unknown"),
                    top_score    = float(itype.get("top_score", 0.0)),
                    all_scores   = [
                        LabelScore(label=s.get("label",""), score=float(s.get("score",0.0)))
                        for s in itype.get("all_scores", [])
                    ],
                    node_id_used = itype.get("node_id_used"),
                    gat_lookup   = itype.get("gat_lookup", False),
                )
        else:
            # PharmBERT-only format
            ranked = out.get("classifications", [])
            top    = ranked[0] if ranked else {"label": "unknown", "score": 0.0, "reasoning": ""}
 
        job_id = str(uuid.uuid4())
 
        await save_classification(
            db,
            job_id=job_id,
            input_text=text_input,
            top_label=top.get("label", "unknown"),
            top_score=float(top.get("score", 0.0)),
            all_labels=ranked,
            latency_ms=round(latency_ms, 2),
            batch_size=len(texts),
            model_name=getattr(predictor, "model_name", "hdi-pipeline"),
            api_key_name=_safe_key_name(api_key),
        )
        await upsert_api_key_log(db, key_name=_safe_key_name(api_key),
                                  role=getattr(api_key, "role", "user"))
 
        CLASSIFICATIONS_TOTAL.labels(
            status="success", top_label=top.get("label", "unknown")
        ).inc()
        CLASSIFICATION_LATENCY.observe(latency_ms)
 
        results.append(ClassifyResult(
            job_id=job_id,
            text=text_input,
            top_label=top.get("label", "unknown"),
            top_score=float(top.get("score", 0.0)),
            classifications=[
                LabelScore(
                    label=r.get("label", ""),
                    score=float(r.get("score", 0.0)),
                    reasoning=r.get("reasoning"),
                )
                for r in ranked
            ],
            interaction_type=itype_obj,
            summary=summary_str,
        ))
 
    return ClassifyResponse(
        results=results,
        model_name=getattr(predictor, "model_name", "hdi-pipeline"),
        latency_ms=round(latency_ms, 2),
    )
 
# ── History ───────────────────────────────────────────────────────────────────
@app.get("/history", tags=["History"])
async def list_history(
    limit: int = Query(50, ge=1, le=500),
    api_key=Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_recent_classifications(db, limit=limit)
    return {
        "total": len(rows),
        "results": [
            {
                "job_id": r.job_id,
                "input_text": r.input_text,
                "top_label": r.top_label,
                "top_score": r.top_score,
                "latency_ms": r.latency_ms,
                "api_key_name": r.api_key_name,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }
 
@app.get("/history/{job_id}", tags=["History"])
async def get_history_item(
    job_id: str,
    api_key=Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    from repository import get_classification
    row = await get_classification(db, job_id)
    if row is None:
        raise HTTPException(404, f"job_id '{job_id}' not found.")
    return {
        "job_id": row.job_id,
        "input_text": row.input_text,
        "top_label": row.top_label,
        "top_score": row.top_score,
        "all_labels": row.all_labels,
        "latency_ms": row.latency_ms,
        "batch_size": row.batch_size,
        "model_name": row.model_name,
        "api_key_name": row.api_key_name,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
 
# ── Analytics ─────────────────────────────────────────────────────────────────
@app.get("/analytics/summary", tags=["Analytics"])
async def analytics_summary(
    api_key=Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    return await get_summary(db)
 
@app.get("/analytics/labels", tags=["Analytics"])
async def analytics_labels(
    days: int = Query(30, ge=1, le=365),
    api_key=Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    data = await get_label_distribution(db, days=days)
    return {"days": days, "distribution": data}
 
@app.get("/analytics/trends", tags=["Analytics"])
async def analytics_trends(
    days: int = Query(30, ge=1, le=365),
    api_key=Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    data = await get_daily_trends(db, days=days)
    return {"days": days, "trends": data}
 
@app.get("/analytics/top-inputs", tags=["Analytics"])
async def analytics_top_inputs(
    limit: int = Query(10, ge=1, le=100),
    api_key=Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    data = await get_top_inputs(db, limit=limit)
    return {"top_inputs": data}
 
@app.get("/analytics/latency", tags=["Analytics"])
async def analytics_latency(
    api_key=Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    return await get_latency_percentiles(db)
 
@app.post("/admin/analytics/refresh", tags=["Admin"])
async def admin_refresh_analytics(
    api_key=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    await refresh_analytics_daily(db)
    return {"message": "analytics_daily refreshed for today."}
 
# ── Health & Metrics ──────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None and getattr(predictor, "is_loaded", True),
        "phase": "G",
    }
 
@app.get("/metrics", tags=["System"], include_in_schema=False)
async def metrics():
    return PlainTextResponse(generate_metrics(), media_type="text/plain")
 
# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {
        "name": "NLP Classifier API",
        "phase": "G",
        "models": "PharmBERT P9 + PharmFusion P8",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "health": "/health",
    }
# ── Setup (generate API key) ──────────────────────────────────────────────────
@app.get("/setup", tags=["System"])
async def setup():
    from api_keys import key_store
    raw_key, info = key_store.create("hf-admin", "admin")
    return {"admin_key": raw_key, "name": info.name, "role": info.role}
 
# ── Dashboard ─────────────────────────────────────────────────────────────────
def _build_dashboard_html() -> str:
    return """<!DOCTYPE html>
<html>
<head>
    <title>HDI Classifier Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { font-size: 2em; font-weight: bold; color: #4CAF50; }
        input[type=text] { padding: 8px; width: 400px; border: 1px solid #ddd;
                           border-radius: 4px; }
        button { padding: 8px 16px; background: #4CAF50; color: white;
                 border: none; border-radius: 4px; cursor: pointer; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f2f2f2; }
        .badge { display:inline-block; padding:2px 8px; border-radius:12px;
                 font-size:0.85em; font-weight:bold; }
        .Harmful  { background:#ffcccc; color:#c00; }
        .Possible { background:#fff3cc; color:#a60; }
        .Positive { background:#ccffcc; color:#060; }
        .Negative { background:#e0e0ff; color:#33c; }
        .NoEffect { background:#eee;    color:#666; }
    </style>
</head>
<body>
    <h1>🌿 HDI Classifier Dashboard</h1>
    <p style="color:#666">PharmBERT P9 (severity) + PharmFusion P8 (type)</p>
    <div class="card">
        <label>API Key: <input type="text" id="apiKey" placeholder="sk-..." /></label>
        <button onclick="loadData()">Load Data</button>
    </div>
    <div class="card">
        <h2>Analytics Summary</h2>
        <div id="summary">Enter your API key above and click Load Data.</div>
    </div>
    <div class="card">
        <h2>Recent Classifications</h2>
        <div id="history">—</div>
    </div>
    <script>
    async function loadData() {
        const key = document.getElementById('apiKey').value.trim();
        if (!key) { alert('Enter your API key first'); return; }
        const h = { 'X-API-Key': key };
 
        // Summary
        try {
            const r = await fetch('/analytics/summary', { headers: h });
            const d = await r.json();
            if (r.status === 401) {
                document.getElementById('summary').innerHTML =
                    '<span style="color:red">Invalid API key</span>';
                return;
            }
            document.getElementById('summary').innerHTML = `
                <p>Total Requests: <span class="metric">${d.total_requests}</span></p>
                <p>Avg Latency:    <span class="metric">${(d.avg_latency_ms||0).toFixed(1)} ms</span></p>
                <p>Max Latency:    <span class="metric">${(d.max_latency_ms||0).toFixed(1)} ms</span></p>
                <p>Top Label:      <span class="metric">${d.top_label ?? 'N/A'}</span></p>`;
        } catch(e) {
            document.getElementById('summary').innerHTML = 'Error loading summary: ' + e;
        }
 
        // History
        try {
            const r = await fetch('/history?limit=10', { headers: h });
            const d = await r.json();
            const rows = (d.results || d);
            if (!rows.length) {
                document.getElementById('history').innerHTML = 'No classifications yet.';
                return;
            }
            const trs = rows.map(x => {
                const cls = (x.top_label||'').replace(' ','');
                return `<tr>
                    <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis">${x.input_text||x.text||''}</td>
                    <td><span class="badge ${cls}">${x.top_label}</span></td>
                    <td>${(x.latency_ms||0).toFixed(1)} ms</td>
                </tr>`;
            }).join('');
            document.getElementById('history').innerHTML =
                `<table><thead><tr><th>Text</th><th>Label</th><th>Latency</th></tr></thead>
                 <tbody>${trs}</tbody></table>`;
        } catch(e) {
            document.getElementById('history').innerHTML = 'Error loading history: ' + e;
        }
    }
    </script>
</body>
</html>"""
 
@app.get("/dashboard", response_class=HTMLResponse, tags=["Monitoring"])
async def dashboard():
    return HTMLResponse(content=_build_dashboard_html())
 
# ── Debug endpoints (keep for diagnostics) ───────────────────────────────────
@app.get("/debug/predictor", tags=["System"])
async def debug_predictor():
    try:
        info = {
            "status":    "ok",
            "model":     getattr(predictor, "model_name", "unknown"),
            "is_loaded": getattr(predictor, "is_loaded", predictor is not None),
            "type":      type(predictor).__name__,
        }
        # If HDIPipeline, show sub-model status
        if hasattr(predictor, "_bert_predictor"):
            info["bert_loaded"]   = predictor._bert_predictor is not None
            info["fusion_loaded"] = predictor._fusion_predictor is not None
            if predictor._fusion_predictor:
                info["fusion_labels"] = predictor._fusion_predictor.label_names
                info["gat_nodes"]     = len(predictor._fusion_predictor._node_to_idx or {})
        return info
    except Exception as e:
        return {"status": "error", "error": str(e)}
 
@app.get("/debug/model-file", tags=["System"])
async def debug_model_file():
    import os
    return {
        "pharmbert_path":  os.getenv("PHARMBERT_MODEL_PATH", "pharmbert_p9_best.pt"),
        "fusion_path":     os.getenv("PHARMFUSION_MODEL_PATH", "pharmfusion_p8_best.pt"),
        "gat_path":        os.getenv("PHARMGAT_EMB_PATH", "pharmgat_node_embeddings.pt"),
        "pharmbert_exists": Path("pharmbert_p9_best.pt").exists(),
        "fusion_exists":    Path("pharmfusion_p8_best.pt").exists(),
        "gat_exists":       Path("pharmgat_node_embeddings.pt").exists(),
    }
 
@app.get("/debug/weights", tags=["System"])
async def debug_weights():
    import torch
    try:
        ckpt = torch.load("pharmbert_p9_best.pt", map_location="cpu", weights_only=False)
        cw   = ckpt["model_state_dict"]["classifier.weight"]
        return {
            "classifier_weight_sum": round(cw.sum().item(), 6),
            "classifier_weight_std": round(cw.std().item(), 6),
            "label_names": ckpt.get("label_names"),
            "macro_f1":    ckpt.get("metrics", {}).get("macro_f1"),
        }
    except Exception as e:
        return {"error": str(e)}
 

@app.get("/debug/auth-env", tags=["System"])
async def debug_auth_env():
    import os
    perm = os.getenv("PERMANENT_ADMIN_KEY", "")
    return {
        "permanent_key_set": bool(perm),
        "permanent_key_length": len(perm),
        "permanent_key_prefix": perm[:6] if perm else None
    }
