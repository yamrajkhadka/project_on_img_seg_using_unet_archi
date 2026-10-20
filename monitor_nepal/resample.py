"""Resampling that matches how the paper's network was trained.

The training notebook shrank 2448 px DeepGlobe tiles to 224 px with ``cv2.resize`` (bilinear,
*no antialiasing*).  At an ~11x reduction that samples a sparse lattice of the 0.5 m pixels and
keeps their high-frequency texture - and the network relies on it: forest IoU on DeepGlobe drops
from 0.58 to 0.00 if the same tiles are shrunk with an antialiasing filter instead.  This module
re-implements OpenCV's ``INTER_LINEAR`` sampling (centre-aligned, edge-clamped) in NumPy so the app
can feed the model exactly what it learned from without depending on OpenCV.
"""
from __future__ import annotations

import numpy as np


def _axis(n_out: int, start: float, length: float, n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = length / n_out
    f = start + (np.arange(n_out) + 0.5) * scale - 0.5
    s0 = np.floor(f).astype(np.int64)
    w = (f - s0).astype(np.float32)
    low = s0 < 0
    w[low], s0[low] = 0.0, 0
    high = s0 >= n_in - 1
    w[high], s0[high] = 0.0, n_in - 1
    return s0, np.minimum(s0 + 1, n_in - 1), w


def cv2_linear(img: np.ndarray, out_w: int, out_h: int,
               box: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """``cv2.resize(img, (out_w, out_h), interpolation=INTER_LINEAR)`` - optionally of a sub-box.

    ``box`` = (left, top, right, bottom) in source pixel units (floats allowed).
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box if box is not None else (0.0, 0.0, float(w), float(h))
    ya, yb, fy = _axis(out_h, y0, y1 - y0, h)
    xa, xb, fx = _axis(out_w, x0, x1 - x0, w)
    src = img.astype(np.float32)
    if src.ndim == 2:
        src = src[..., None]
    fx = fx[None, :, None]
    top = src[ya][:, xa] * (1 - fx) + src[ya][:, xb] * fx
    bot = src[yb][:, xa] * (1 - fx) + src[yb][:, xb] * fx
    out = top * (1 - fy)[:, None, None] + bot * fy[:, None, None]
    if img.dtype == np.uint8:
        out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    else:
        out = out.astype(img.dtype)
    return out[..., 0] if img.ndim == 2 else out
