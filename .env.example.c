# ─────────────────────────────────────────────────────────────────────────────
# Phase C — .env configuration
# Copy to .env and fill in your values
# ─────────────────────────────────────────────────────────────────────────────

# ── Server ────────────────────────────────────────────────────────────────────
PORT=8000
LOG_LEVEL=info
ALLOWED_ORIGINS=*

# ── ANTHROPIC API (REQUIRED for Phase C) ──────────────────────────────────────
ANTHROPIC_API_KEY=your-anthropic-api-key-here
CLASSIFIER_MODEL=llama-3.1-8b-instant

# One-line description of what you're classifying (used in the prompt)
CLASSIFIER_CONTEXT=customer support messages

# Your custom labels — comma separated, no spaces around commas
CLASSIFIER_LABELS=billing,technical_support,account_management,sales_inquiry,complaint,feature_request,general_question,other

# ── Pipeline (from Phase B) ───────────────────────────────────────────────────
MAX_INPUT_LENGTH=512
MAX_BATCH_SIZE=32
BATCH_TIMEOUT_MS=50
NUM_WORKERS=2
SYNC_TIMEOUT_S=30
QUEUE_MAX_SIZE=500
QUEUE_BACKEND=memory
