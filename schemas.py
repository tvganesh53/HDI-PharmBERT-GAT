"""
schemas.py — Shared Pydantic models for Phase B pipeline.
"""

from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid
import time


class JobStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


class PredictRequest(BaseModel):
    inputs: Any = Field(..., description="Raw text input (str or list[str])")
    parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "inputs": "Classify this sentence.",
                "parameters": {"max_length": 512}
            }
        }
    }


class PredictResponse(BaseModel):
    outputs: Any
    model_name: str
    latency_ms: float
    batch_size: int = 1


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    outputs: Any | None = None
    error: str | None = None
    queued_at: float
    completed_at: float | None = None


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    inputs: Any
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    outputs: Any | None = None
    error: str | None = None
    queued_at: float = Field(default_factory=time.time)
    completed_at: float | None = None

    def to_response(self) -> JobResponse:
        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            outputs=self.outputs,
            error=self.error,
            queued_at=self.queued_at,
            completed_at=self.completed_at,
        )
class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # ← add this line
    outputs: Any
    model_name: str
    latency_ms: float
    batch_size: int = 1