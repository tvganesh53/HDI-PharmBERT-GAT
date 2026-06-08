"""
pipeline.py — NLP text preprocessing pipeline.

Steps: validate → clean → normalise → truncate → batch
"""

from __future__ import annotations
import html
import logging
import os
import re
import unicodedata
from typing import Any

log = logging.getLogger("phase_b.pipeline")

# ── Config (override via .env) ────────────────────────────────────────────────
MAX_INPUT_LENGTH  = int(os.getenv("MAX_INPUT_LENGTH",  "512"))   # chars
MAX_BATCH_SIZE    = int(os.getenv("MAX_BATCH_SIZE",    "32"))


class PipelineError(ValueError):
    """Raised for recoverable, user-visible preprocessing errors."""


class NLPPipeline:
    """
    Stateless preprocessing pipeline.  Call process() on raw API input;
    get back a clean list of strings ready for the model.
    """

    def __init__(
        self,
        max_length: int = MAX_INPUT_LENGTH,
        max_batch: int  = MAX_BATCH_SIZE,
    ) -> None:
        self.max_length = max_length
        self.max_batch  = max_batch

    # ── Public entry-point ────────────────────────────────────────────────────
    def process(self, inputs: Any, **params: Any) -> list[str]:
        """
        Accept str, list[str], or dict with a 'text' key.
        Returns a clean, truncated list[str] ready for batching.
        """
        texts = self._coerce_to_list(inputs)
        if len(texts) > self.max_batch:
            raise PipelineError(
                f"Batch too large: got {len(texts)}, max is {self.max_batch}."
            )

        cleaned = [self._clean(t) for t in texts]
        truncated = [self._truncate(t, params.get("max_length", self.max_length)) for t in cleaned]

        log.debug("Pipeline processed %d text(s)", len(truncated))
        return truncated

    # ── Coercion ──────────────────────────────────────────────────────────────
    def _coerce_to_list(self, inputs: Any) -> list[str]:
        if isinstance(inputs, str):
            self._require_nonempty(inputs)
            return [inputs]
        if isinstance(inputs, list):
            if not inputs:
                raise PipelineError("'inputs' list must not be empty.")
            for i, item in enumerate(inputs):
                if not isinstance(item, str):
                    raise PipelineError(f"inputs[{i}] must be a string, got {type(item).__name__}.")
                self._require_nonempty(item, index=i)
            return inputs
        if isinstance(inputs, dict):
            text = inputs.get("text") or inputs.get("content") or inputs.get("input")
            if not text or not isinstance(text, str):
                raise PipelineError("Dict input must have a 'text', 'content', or 'input' key.")
            self._require_nonempty(text)
            return [text]
        raise PipelineError(
            f"Unsupported input type '{type(inputs).__name__}'. "
            "Use str, list[str], or dict with a 'text' key."
        )

    @staticmethod
    def _require_nonempty(text: str, index: int | None = None) -> None:
        label = f"inputs[{index}]" if index is not None else "'inputs'"
        if not text or not text.strip():
            raise PipelineError(f"{label} must not be empty or whitespace-only.")

    # ── Cleaning steps ────────────────────────────────────────────────────────
    def _clean(self, text: str) -> str:
        text = html.unescape(text)                        # &amp; → &
        text = self._strip_html_tags(text)                # <b>…</b> → …
        text = unicodedata.normalize("NFKC", text)        # unicode normalise
        text = self._fix_whitespace(text)                 # collapse spaces
        return text

    @staticmethod
    def _strip_html_tags(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    @staticmethod
    def _fix_whitespace(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)   # collapse spaces/tabs
        text = re.sub(r"\n{3,}", "\n\n", text) # collapse excessive newlines
        return text.strip()

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        log.debug("Truncating text from %d to %d chars", len(text), max_length)
        return text[:max_length]
