"""
Predictor — model loading, device selection, and inference engine.

Adapt the `_run_inference()` method to your actual model (PyTorch, ONNX,
HuggingFace, scikit-learn, etc.).  Everything else is wiring.
"""

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("phase_a.predictor")


class PredictorError(Exception):
    """Raised for known, user-facing inference errors."""


class Predictor:
    """
    Thin wrapper around any ML model.

    Environment variables (see .env.example):
        MODEL_NAME          Human-readable model name          (default: "phase-a-model")
        MODEL_VERSION       Semver string                       (default: "1.0.0")
        MODEL_PATH          Path to weights / checkpoint        (default: "models/model.pt")
        DEVICE              "cpu" | "cuda" | "auto"             (default: "auto")
    """

    def __init__(self) -> None:
        self.model_name: str = os.getenv("MODEL_NAME", "phase-a-model")
        self.model_version: str = os.getenv("MODEL_VERSION", "1.0.0")
        self.model_path: Path = Path(os.getenv("MODEL_PATH", "models/model.pt"))
        self._device_pref: str = os.getenv("DEVICE", "auto")

        self._model: Any = None
        self._tokenizer: Any = None  # optional — remove if not needed
        self.is_loaded: bool = False

    # ── Device ────────────────────────────────────────────────────────────────
    @property
    def device(self) -> str:
        """Resolved device string, e.g. 'cuda' or 'cpu'."""
        if self._device_pref == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self._device_pref

    # ── Model metadata ────────────────────────────────────────────────────────
    @property
    def model_params(self) -> dict[str, Any]:
        """Return useful metadata about the loaded model."""
        info: dict[str, Any] = {
            "model_path": str(self.model_path),
            "device": self.device,
        }
        # If it's a PyTorch model, report parameter count
        try:
            import torch.nn as nn
            if isinstance(self._model, nn.Module):
                n_params = sum(p.numel() for p in self._model.parameters())
                info["num_parameters"] = n_params
        except Exception:
            pass
        return info

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def load(self) -> None:
        """
        Load model weights into memory.

        ┌─────────────────────────────────────────────────────────┐
        │  REPLACE the body of this method with your own loader.  │
        │  Examples for common frameworks are shown below.        │
        └─────────────────────────────────────────────────────────┘
        """
        log.info("Loading model '%s' v%s onto %s …", self.model_name, self.model_version, self.device)

        # ── Example A: PyTorch checkpoint ────────────────────────────────────
        # import torch
        # from my_model import MyModel
        # self._model = MyModel()
        # state = torch.load(self.model_path, map_location=self.device)
        # self._model.load_state_dict(state)
        # self._model.to(self.device)
        # self._model.eval()

        # ── Example B: HuggingFace Transformers ──────────────────────────────
        # from transformers import AutoModelForSequenceClassification, AutoTokenizer
        # self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        # self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        # self._model.to(self.device)
        # self._model.eval()

        # ── Example C: ONNX Runtime ───────────────────────────────────────────
        # import onnxruntime as ort
        # providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        # self._model = ort.InferenceSession(str(self.model_path), providers=providers)

        # ── Example D: scikit-learn / joblib ─────────────────────────────────
        # import joblib
        # self._model = joblib.load(self.model_path)

        # ── STUB (remove when using a real model) ─────────────────────────────
        self._model = _StubModel()
        # ─────────────────────────────────────────────────────────────────────

        self.is_loaded = True
        log.info("Model loaded successfully ✓")

    def unload(self) -> None:
        """Release model resources."""
        self._model = None
        self._tokenizer = None
        self.is_loaded = False
        log.info("Model unloaded.")

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        """
        Public inference entry-point.

        Args:
            inputs:   Raw input (string, list, dict, ndarray — whatever your model expects).
            **kwargs: Optional inference parameters forwarded from the request.

        Returns:
            Model output (JSON-serialisable).
        """
        if not self.is_loaded or self._model is None:
            raise PredictorError("Model is not loaded.")

        self._validate_inputs(inputs)
        preprocessed = self._preprocess(inputs, **kwargs)
        raw_output = self._run_inference(preprocessed, **kwargs)
        return self._postprocess(raw_output)

    # ── Internal pipeline steps ───────────────────────────────────────────────
    def _validate_inputs(self, inputs: Any) -> None:
        """Raise PredictorError for obviously bad inputs."""
        if inputs is None:
            raise PredictorError("'inputs' must not be None.")
        if isinstance(inputs, str) and len(inputs.strip()) == 0:
            raise PredictorError("'inputs' string must not be empty.")

    def _preprocess(self, inputs: Any, **kwargs: Any) -> Any:
        """
        Convert raw API inputs into the tensor / array your model expects.

        ┌──────────────────────────────────────────────────────────┐
        │  REPLACE with your preprocessing logic.                  │
        └──────────────────────────────────────────────────────────┘
        """
        # Example: tokenise text for a HuggingFace model
        # if self._tokenizer:
        #     return self._tokenizer(inputs, return_tensors="pt", truncation=True, padding=True).to(self.device)
        return inputs  # pass-through for stub

    def _run_inference(self, preprocessed: Any, **kwargs: Any) -> Any:
        """
        Call the model.

        ┌──────────────────────────────────────────────────────────┐
        │  REPLACE with your model's forward / run call.           │
        └──────────────────────────────────────────────────────────┘
        """
        # PyTorch example:
        # import torch
        # with torch.no_grad():
        #     return self._model(**preprocessed)

        # ONNX example:
        # input_name = self._model.get_inputs()[0].name
        # return self._model.run(None, {input_name: preprocessed})

        return self._model(preprocessed, **kwargs)  # stub call

    def _postprocess(self, raw_output: Any) -> Any:
        """
        Convert model output into a JSON-serialisable Python object.

        ┌──────────────────────────────────────────────────────────┐
        │  REPLACE with your postprocessing / decoding logic.      │
        └──────────────────────────────────────────────────────────┘
        """
        # PyTorch softmax → list example:
        # import torch
        # probs = torch.softmax(raw_output.logits, dim=-1)
        # return probs.tolist()

        return raw_output  # stub pass-through


# ── Stub model (used until you plug in a real one) ────────────────────────────
class _StubModel:
    """Echo model — returns a structured echo of whatever it receives."""

    def __call__(self, inputs: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "echo": inputs,
            "parameters_received": kwargs,
            "note": "Replace _StubModel with your real model in predictor.py",
        }
