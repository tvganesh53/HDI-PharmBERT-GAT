FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# Pre-cache BioBERT tokenizer so startup is faster
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')"

# Copy application code (no .pt files — they download at runtime)
COPY app_phase_g.py .
COPY predictor.py .
COPY predictor_c.py .
COPY predictor_pharmbert.py .
COPY predictor_fusion.py .
COPY pipeline_hdi.py .
COPY pipeline.py .
COPY batcher.py .
COPY worker.py .
COPY schemas.py .
COPY job_queue.py .
COPY auth.py .
COPY api_keys.py .
COPY rate_limiter.py .
COPY database.py .
COPY models.py .
COPY repository.py .
COPY db_adapter.py .
COPY keys.json .

RUN mkdir -p /app/model_cache /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# Allow extra time for model downloads on first boot (up to 10 min)
HEALTHCHECK --interval=60s --timeout=30s --retries=10 --start-period=600s \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app_phase_g:app", "--host", "0.0.0.0", "--port", "7860"]
