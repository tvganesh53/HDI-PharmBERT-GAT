# HF Spaces Deployment Guide

## Step-by-step deployment for tvganesh538/nlp-classifier-api

---

## Step 1 — Install huggingface_hub

```powershell
pip install huggingface_hub
```

---

## Step 2 — Copy deployment files to your project folder

Copy these files from this package into your `nodejs/` project folder:
- `README.md`        ← replaces existing README if any
- `Dockerfile`       ← replaces your existing Dockerfile
- `requirements_hf.txt`
- `db_adapter.py`
- `deploy_to_hf.py`

---

## Step 3 — Run the deploy script

```powershell
cd C:\Users\tvgan\OneDrive\Desktop\nodejs
python deploy_to_hf.py
```

This will:
- Create the Space `tvganesh538/nlp-classifier-api` on HF
- Set your GROQ_API_KEY as a secret
- Upload all your app files
- Trigger a Docker build

---

## Step 4 — Watch the build logs

Go to:
https://huggingface.co/spaces/tvganesh538/nlp-classifier-api/logs

Wait 2-5 minutes for the build to complete.

---

## Step 5 — Access your live API

Once built:
- Dashboard : https://tvganesh538-nlp-classifier-api.hf.space/dashboard
- Swagger UI: https://tvganesh538-nlp-classifier-api.hf.space/docs
- Health    : https://tvganesh538-nlp-classifier-api.hf.space/health

---

## Step 6 — Generate your first API key on HF Space

```powershell
# The admin key is printed in the Space logs on first startup
# Go to: https://huggingface.co/spaces/tvganesh538/nlp-classifier-api/logs
# Look for: "created default admin key: sk-..."
# Copy that key
```

---

## Auto-deploy with GitHub Actions (optional)

1. Push your project to GitHub
2. Go to Settings → Secrets → New secret
3. Name: `HF_TOKEN`  Value: `YOUR_HF_TOKEN`
4. Copy `.github/workflows/deploy.yml` to your repo
5. Every push to `main` auto-deploys to HF Spaces

---

## Differences from local Docker setup

| Feature        | Local Docker      | HF Spaces         |
|----------------|-------------------|-------------------|
| Database       | MySQL             | SQLite            |
| Port           | 8001              | 7860              |
| API URL        | localhost:8001    | *.hf.space        |
| keys.json      | volume mounted    | written to /app/data |
| GROQ_API_KEY   | docker-compose.yml| HF Space secret   |
