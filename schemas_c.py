"""
schemas_c.py — Phase C Pydantic models for classification output.
Extends Phase B schemas.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from schemas import Job, JobStatus, JobResponse, PredictRequest   # re-export Phase B


# ── Classification output ─────────────────────────────────────────────────────
class LabelScore(BaseModel):
    label:     str
    score:     float = Field(..., ge=0.0, le=1.0)
    reasoning: str   = ""


class ClassificationResult(BaseModel):
    text:            str
    top_label:       str
    top_score:       float
    classifications: list[LabelScore]


class ClassifyRequest(BaseModel):
    inputs:     Any                  = Field(..., description="Text or list of texts to classify")
    labels:     list[str] | None     = Field(None,  description="Override default label set")
    context:    str | None           = Field(None,  description="Override classifier context description")
    parameters: dict[str, Any]       = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "inputs": "I can't log into my account and need help urgently.",
                "labels": None,
                "context": None,
            }
        }
    }


class ClassifyResponse(BaseModel):
    results:    list[ClassificationResult] | ClassificationResult
    model_name: str
    latency_ms: float
    label_set:  list[str]
class ClassifyResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # ← add this line
    results:    list[ClassificationResult] | ClassificationResult
    model_name: str
    latency_ms: float
    label_set:  list[str]