"""
logger.py — Structured JSON logger for Phase D.
Every classification is saved to logs/classifications.jsonl
"""

from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(exist_ok=True)

CLASSIFICATION_LOG = LOG_DIR / "classifications.jsonl"
ERROR_LOG          = LOG_DIR / "errors.jsonl"

log = logging.getLogger("phase_d.logger")


def _write(path: Path, record: dict) -> None:
    """Append a JSON record to a .jsonl file."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.error("Failed to write log record: %s", exc)


def log_classification(
    job_id: str,
    input_text: str,
    top_label: str,
    top_score: float,
    all_labels: list[dict],
    latency_ms: float,
    batch_size: int = 1,
    model: str = "groq-classifier",
) -> None:
    """Log a successful classification to classifications.jsonl"""
    record = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "job_id":     job_id,
        "event":      "classification",
        "model":      model,
        "input":      input_text[:500],   # truncate for storage
        "top_label":  top_label,
        "top_score":  top_score,
        "all_labels": all_labels,
        "latency_ms": latency_ms,
        "batch_size": batch_size,
    }
    _write(CLASSIFICATION_LOG, record)


def log_error(
    job_id: str,
    error_type: str,
    error_message: str,
    input_text: str = "",
) -> None:
    """Log a classification error to errors.jsonl"""
    record = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "job_id":        job_id,
        "event":         "error",
        "error_type":    error_type,
        "error_message": error_message,
        "input":         input_text[:500],
    }
    _write(ERROR_LOG, record)


def read_recent_classifications(limit: int = 50) -> list[dict]:
    """Read the most recent N classifications from the log file."""
    if not CLASSIFICATION_LOG.exists():
        return []
    try:
        lines = CLASSIFICATION_LOG.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-limit:]
        return [json.loads(line) for line in reversed(recent)]
    except Exception as exc:
        log.error("Failed to read classification log: %s", exc)
        return []


def read_recent_errors(limit: int = 20) -> list[dict]:
    """Read the most recent N errors from the error log."""
    if not ERROR_LOG.exists():
        return []
    try:
        lines = ERROR_LOG.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-limit:]
        return [json.loads(line) for line in reversed(recent)]
    except Exception as exc:
        log.error("Failed to read error log: %s", exc)
        return []


def get_stats() -> dict[str, Any]:
    """Compute summary stats from the classification log."""
    if not CLASSIFICATION_LOG.exists():
        return {"total": 0, "label_counts": {}, "avg_latency_ms": 0, "avg_score": 0}

    try:
        lines = CLASSIFICATION_LOG.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(l) for l in lines if l.strip()]

        if not records:
            return {"total": 0, "label_counts": {}, "avg_latency_ms": 0, "avg_score": 0}

        label_counts: dict[str, int] = {}
        total_latency = 0.0
        total_score   = 0.0

        for r in records:
            label = r.get("top_label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
            total_latency += r.get("latency_ms", 0)
            total_score   += r.get("top_score", 0)

        n = len(records)
        return {
            "total":          n,
            "label_counts":   label_counts,
            "avg_latency_ms": round(total_latency / n, 2),
            "avg_score":      round(total_score / n, 4),
        }
    except Exception as exc:
        log.error("Failed to compute stats: %s", exc)
        return {"total": 0, "label_counts": {}, "avg_latency_ms": 0, "avg_score": 0}
