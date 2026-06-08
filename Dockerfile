# ─────────────────────────────────────────────────────────────────────────────
# Phase A — Inference API  •  Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build:
#   1. builder — install Python deps into a venv
#   2. runtime — copy only the venv + app code into a slim final image
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels (numpy, tokenizers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Bring in the pre-built venv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY app.py predictor.py ./

# Optional: copy model weights (uncomment and adjust path as needed)
# COPY models/ ./models/

# Ensure the non-root user owns the app directory
RUN chown -R appuser:appuser /app
USER appuser

# ── Runtime config ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# Health-check so container orchestrators know when the API is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Entry-point ───────────────────────────────────────────────────────────────
CMD ["sh", "-c", \
     "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info"]
