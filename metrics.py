"""
metrics.py — Prometheus metrics for Phase D.
Tracks classifications, latency, errors, queue depth, and batch size.
"""

from __future__ import annotations
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY

# ── Counters ──────────────────────────────────────────────────────────────────
classifications_total = Counter(
    "classifications_total",
    "Total number of classification requests",
    ["status", "top_label"],
)

classification_errors_total = Counter(
    "classification_errors_total",
    "Total number of classification errors",
    ["error_type"],
)

label_distribution = Counter(
    "label_distribution_total",
    "How many times each label was the top prediction",
    ["label"],
)

# ── Histograms ────────────────────────────────────────────────────────────────
classification_latency = Histogram(
    "classification_latency_ms",
    "Classification latency in milliseconds",
    buckets=[50, 100, 250, 500, 1000, 2000, 5000, 10000],
)

batch_size_histogram = Histogram(
    "classification_batch_size",
    "Number of texts per batch",
    buckets=[1, 2, 4, 8, 16, 32],
)

# ── Gauges ────────────────────────────────────────────────────────────────────
queue_depth_gauge = Gauge(
    "queue_depth",
    "Current number of jobs waiting in the queue",
)

active_workers_gauge = Gauge(
    "active_workers",
    "Number of active batch workers",
)


def record_classification(
    top_label: str,
    latency_ms: float,
    batch_size: int = 1,
    status: str = "success",
) -> None:
    """Call this after every successful classification."""
    classifications_total.labels(status=status, top_label=top_label).inc()
    label_distribution.labels(label=top_label).inc()
    classification_latency.observe(latency_ms)
    batch_size_histogram.observe(batch_size)


def record_error(error_type: str) -> None:
    """Call this when a classification fails."""
    classification_errors_total.labels(error_type=error_type).inc()
    classifications_total.labels(status="error", top_label="none").inc()


def update_queue_depth(depth: int) -> None:
    """Call this to update the live queue depth gauge."""
    queue_depth_gauge.set(depth)


def set_active_workers(count: int) -> None:
    """Call this on startup to record worker count."""
    active_workers_gauge.set(count)
