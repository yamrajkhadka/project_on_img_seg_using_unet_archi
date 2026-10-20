"""Inference with the paper's U-Net (exported losslessly from Keras to ONNX).

The network is fully convolutional, so instead of squashing a whole image into 224x224 (which
changes the apparent size of every field and building) we slide overlapping windows over imagery
that has already been resampled to the training ground sample distance, and blend the softmax
outputs with a smooth weight ramp so no tile seams appear.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import MODEL_GSD_M, NUM_CLASSES, ONNX_MODEL_PATH

ProgressFn = Callable[[float, str], None]


def _ramp(n: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones(n, np.float32)
    i = np.arange(n, dtype=np.float32) + 0.5
    return np.clip(np.minimum(i, n - i) / overlap, 1e-3, 1.0)


def _positions(total: int, window: int, stride: int) -> list[int]:
    if total <= window:
        return [0]
    pos = list(range(0, total - window, stride))
    pos.append(total - window)
    return pos


# Test-time augmentation: (forward, inverse) pairs on HWC arrays.
_TTA = [
    (lambda a: a, lambda a: a),
    (lambda a: a[:, ::-1], lambda a: a[:, ::-1]),
    (lambda a: a[::-1, :], lambda a: a[::-1, :]),
    (lambda a: np.rot90(a, 1), lambda a: np.rot90(a, -1)),
]


class LandCoverModel:
    """Thread-safe wrapper around the ONNX Runtime session."""

    def __init__(self, path: str | os.PathLike = ONNX_MODEL_PATH, threads: int | None = None):
        import onnxruntime as ort  # imported lazily so the rest of the package works without it

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = threads or int(os.environ.get("MONITOR_NEPAL_THREADS", "0")) or None  # cap CPU use (heat)
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self._lock = threading.Lock()
        self.path = str(path)

    def _run(self, batch: np.ndarray) -> np.ndarray:
        with self._lock:  # ORT sessions are thread-safe, but serialising keeps peak memory bounded
            return self.session.run(None, {self.input_name: batch})[0]

    def predict_window(self, x: np.ndarray, tta: int = 1) -> np.ndarray:
        """Softmax for one float32 HWC window (H, W multiples of 16)."""
        acc = np.zeros(x.shape[:2] + (NUM_CLASSES,), np.float32)
        for fwd, inv in _TTA[:max(1, tta)]:
            out = self._run(np.ascontiguousarray(fwd(x))[None])[0]
            acc += inv(out)
        return acc / max(1, tta)

    def predict_proba(self, rgb: np.ndarray, *, window: int = 512, overlap: int = 64, tta: int = 1,
                      progress: ProgressFn | None = None) -> np.ndarray:
        """Per-pixel class probabilities (H, W, 7) for an RGB uint8 image of any size."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image of shape (H, W, 3).")
        h, w = rgb.shape[:2]
        window = max(64, (window // 16) * 16)
        win_h = min(window, int(np.ceil(h / 16) * 16))
        win_w = min(window, int(np.ceil(w / 16) * 16))
        ph, pw = max(h, win_h), max(w, win_w)
        img = rgb.astype(np.float32) / 255.0
        if (ph, pw) != (h, w):
            img = np.pad(img, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect")

        ov_h, ov_w = min(overlap, win_h // 4), min(overlap, win_w // 4)
        rows = _positions(ph, win_h, win_h - ov_h)
        cols = _positions(pw, win_w, win_w - ov_w)
        if len(rows) == 1 and len(cols) == 1:
            out = self.predict_window(img, tta)
            if progress:
                progress(1.0, "Segmented 1/1 window")
            return out[:h, :w]

        weight = (_ramp(win_h, ov_h)[:, None] * _ramp(win_w, ov_w)[None, :])[..., None]
        acc = np.zeros((ph, pw, NUM_CLASSES), np.float32)
        wsum = np.zeros((ph, pw, 1), np.float32)
        total, k = len(rows) * len(cols), 0
        for r in rows:
            for c in cols:
                acc[r:r + win_h, c:c + win_w] += self.predict_window(img[r:r + win_h, c:c + win_w], tta) * weight
                wsum[r:r + win_h, c:c + win_w] += weight
                k += 1
                if progress:
                    progress(k / total, f"Segmented {k}/{total} windows")
        return (acc / np.maximum(wsum, 1e-6))[:h, :w]

    def predict_paper_style(self, rgb: np.ndarray) -> np.ndarray:
        """Exactly what the original demo did: squash the whole image to 224x224 and predict."""
        from PIL import Image

        small = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR))
        return self.predict_window(small.astype(np.float32) / 255.0, tta=1)


@dataclass
class Prediction:
    classes: np.ndarray  # (H, W) uint8
    confidence: np.ndarray  # (H, W) float16, max softmax probability
    mean_proba: np.ndarray  # (7,) mean probability per class

    @classmethod
    def from_proba(cls, proba: np.ndarray, valid: np.ndarray | None = None) -> "Prediction":
        classes = proba.argmax(-1).astype(np.uint8)
        conf = proba.max(-1).astype(np.float16)
        if valid is not None:
            classes = np.where(valid, classes, 6).astype(np.uint8)
            conf = np.where(valid, conf, 0).astype(np.float16)
            mp = proba[valid].mean(0) if valid.any() else np.zeros(NUM_CLASSES)
        else:
            mp = proba.reshape(-1, NUM_CLASSES).mean(0)
        return cls(classes, conf, mp.astype(np.float32))


@dataclass(frozen=True)
class ModelInfo:
    key: str
    label: str
    short: str
    filename: str
    description: str

    @property
    def path(self):
        from .config import MODELS_DIR

        return MODELS_DIR / self.filename

    @property
    def available(self) -> bool:
        return self.path.exists()


MODELS: dict[str, ModelInfo] = {
    "nepal": ModelInfo("nepal", "Nepal-adapted U-Net (fine-tuned from the paper's weights)", "Nepal-adapted",
                       "unet_nepal.onnx",
                       "The paper's network and loss, fine-tuned on weakly-labelled Nepal patches from Esri, "
                       "Sentinel-2 and Landsat imagery. Recommended for monitoring."),
    "paper": ModelInfo("paper", "Original U-Net (as published, DeepGlobe-trained)", "Original (paper)",
                       "unet_deepglobe.onnx",
                       "Exactly the weights in the paper (deepglobe_unet_jay.keras), exported losslessly to ONNX."),
}

_MODELS: dict[str, LandCoverModel] = {}
_MODEL_LOCK = threading.Lock()


def available_models() -> list[ModelInfo]:
    return [m for m in MODELS.values() if m.available]


def default_model_key() -> str:
    return "nepal" if MODELS["nepal"].available else "paper"


def get_model(key: str | None = None) -> LandCoverModel:
    """Process-wide cache of ONNX sessions (the app additionally wraps this in st.cache_resource)."""
    key = key or default_model_key()
    with _MODEL_LOCK:
        if key not in _MODELS:
            _MODELS[key] = LandCoverModel(MODELS[key].path)
        return _MODELS[key]


__all__ = ["LandCoverModel", "Prediction", "ModelInfo", "MODELS", "available_models", "default_model_key",
           "get_model", "MODEL_GSD_M"]
