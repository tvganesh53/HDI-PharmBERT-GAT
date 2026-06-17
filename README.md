---
title: NLP Classifier API
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# NLP Herb-Drug Interaction Classifier API

A FastAPI-based REST API for classifying herb-drug interactions using Groq LLM.

## Endpoints

- `GET /health` — public health check
- `POST /classify` — classify text (requires API key)
- `GET /analytics/summary` — analytics (requires API key)
- `GET /history` — classification history (requires API key)
- `GET /dashboard` — live dashboard (public page, key entered in UI)
- `GET /docs` — Swagger UI

## Usage

All protected endpoints require `X-API-Key` header.

Generate a key via the `/admin/keys` endpoint using the admin key printed in container logs on first startup.
