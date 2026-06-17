"""
predictor_c.py (Phase C) — Groq-powered text classifier.
Free API — no credit card needed.
"""
 
from __future__ import annotations
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
 
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
 
from groq import Groq
 
log = logging.getLogger("phase_c.predictor")
 
# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
CLASSIFIER_MODEL   = os.getenv("CLASSIFIER_MODEL", "llama-3.1-8b-instant")
CLASSIFIER_CONTEXT = os.getenv("CLASSIFIER_CONTEXT", "customer support messages")
 
DEFAULT_LABELS = [
    "billing",
    "technical_support",
    "account_management",
    "sales_inquiry",
    "complaint",
    "feature_request",
    "general_question",
    "other",
]
 
CLASSIFIER_LABELS: list[str] = [
    label.strip()
    for label in os.getenv("CLASSIFIER_LABELS", ",".join(DEFAULT_LABELS)).split(",")
    if label.strip()
]
 
 
class PredictorError(Exception):
    """Raised for known, user-facing inference errors."""
 
 
class Predictor:
    """Groq-powered text classifier."""
 
    def __init__(self) -> None:
        self.model_name    = "groq-classifier"
        self.model_version = "1.0.0"
        self._client: Groq | None = None
        self.is_loaded     = False
 
    @property
    def device(self) -> str:
        return "api"
 
    @property
    def model_params(self) -> dict[str, Any]:
        return {
            "provider":      "groq",
            "groq_model":    CLASSIFIER_MODEL,
            "labels":        CLASSIFIER_LABELS,
            "label_count":   len(CLASSIFIER_LABELS),
            "context":       CLASSIFIER_CONTEXT,
        }
 
    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def load(self) -> None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        self._client = Groq(api_key=GROQ_API_KEY)
        self.is_loaded = True
        log.info("Groq client initialised (model=%s, labels=%s)",
                 CLASSIFIER_MODEL, CLASSIFIER_LABELS)
 
    def unload(self) -> None:
        self._client = None
        self.is_loaded = False
        log.info("Predictor unloaded.")
 
    # ── Public inference ──────────────────────────────────────────────────────
    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        if not self.is_loaded or self._client is None:
            raise PredictorError("Predictor not loaded.")
 
        texts   = [inputs] if isinstance(inputs, str) else inputs
        labels  = kwargs.get("labels",  CLASSIFIER_LABELS)
        context = kwargs.get("context", CLASSIFIER_CONTEXT)
        model   = kwargs.get("model",   CLASSIFIER_MODEL)
 
        results = [
            self._classify_single(text, labels=labels, context=context, model=model)
            for text in texts
        ]
        return results if len(results) > 1 else results[0]
 
    # ── Internal ──────────────────────────────────────────────────────────────
    def _classify_single(
        self,
        text: str,
        labels: list[str],
        context: str,
        model: str,
    ) -> dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self._system_prompt(labels)},
                    {"role": "user",   "content": self._build_prompt(text, labels, context)},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            raw = response.choices[0].message.content
        except Exception as exc:
            raise PredictorError(f"Groq API error: {exc}") from exc
 
        return self._parse_response(raw, text, labels)
 
    # ── Prompt ────────────────────────────────────────────────────────────────
    @staticmethod
    def _system_prompt(labels: list[str]) -> str:
        label_list = "\n".join(f"  - {lbl}" for lbl in labels)
        return f"""You are a text classification engine.
Classify input text into one or more of these labels:
 
{label_list}
 
RULES:
1. Respond ONLY with valid JSON - no explanation, no markdown, no code fences.
2. Return every label with a confidence score between 0.00 and 1.00.
3. Scores across all labels must sum to exactly 1.00.
4. Sort labels from highest to lowest score.
5. Use exactly this JSON format:
{{
  "classifications": [
    {{"label": "<label>", "score": <float>, "reasoning": "<one short sentence>"}},
    ...
  ]
}}"""
 
    @staticmethod
    def _build_prompt(text: str, labels: list[str], context: str) -> str:
        return (
            f"Classify the following {context}:\n\n"
            f'"""\n{text}\n"""\n\n'
            f"Available labels: {', '.join(labels)}\n"
            "Return the JSON classification now."
        )
 
    # ── Response parsing ──────────────────────────────────────────────────────
    def _parse_response(
        self,
        raw: str,
        original_text: str,
        labels: list[str],
    ) -> dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
 
        try:
            data = json.loads(cleaned)
            classifications = data["classifications"]
        except (json.JSONDecodeError, KeyError):
            log.warning("Failed to parse response, using fallback. Raw: %s", raw)
            classifications = self._fallback_classifications(labels)
 
        # Normalise scores to sum to 1.0
        total = sum(c.get("score", 0) for c in classifications)
        if total > 0:
            for c in classifications:
                c["score"] = round(c["score"] / total, 4)
 
        # Ensure all labels are present
        present = {c["label"] for c in classifications}
        for label in labels:
            if label not in present:
                classifications.append({
                    "label": label,
                    "score": 0.0,
                    "reasoning": "not applicable"
                })
 
        # Sort highest to lowest
        classifications.sort(key=lambda x: x["score"], reverse=True)
 
        return {
            "text":            original_text,
            "top_label":       classifications[0]["label"],
            "top_score":       classifications[0]["score"],
            "classifications": classifications,
        }
 
    @staticmethod
    def _fallback_classifications(labels: list[str]) -> list[dict]:
        uniform = round(1.0 / len(labels), 4)
        return [
            {"label": lbl, "score": uniform, "reasoning": "parse error - uniform fallback"}
            for lbl in labels
        ]
 

# deploy 1781060726.704266