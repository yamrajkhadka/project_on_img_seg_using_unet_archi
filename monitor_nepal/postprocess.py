"""Mask post-processing.

``paper_morphology`` / ``paper_median`` reproduce the published demo (``legacy/app_alpha.py``)
bit-for-bit, using SciPy instead of scikit-image.  They have one quirk worth knowing: pixels that
are removed as noise are never re-assigned, so they fall back to class 0 - *Urban land* - and show
up as false urban speckle.  ``clean`` fixes that by giving each removed pixel the class of the
nearest surviving region, and ``majority_filter`` is a proper multi-class mode filter.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import NUM_CLASSES

_EIGHT = np.ones((3, 3), bool)
_FOUR = ndimage.generate_binary_structure(2, 1)


def paper_morphology(mask: np.ndarray, min_size: int = 64, hole_area: int = 64) -> np.ndarray:
    """As published: per class, drop components < ``min_size`` px and fill holes < ``hole_area`` px."""
    processed = np.zeros_like(mask)
    for cls in np.unique(mask):
        labels, _ = ndimage.label(mask == cls, structure=_EIGHT)  # skimage.measure.label default
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_size
        keep[0] = False
        kept = keep[labels]
        holes, _ = ndimage.label(~kept, structure=_FOUR)  # remove_small_holes default connectivity=1
        hsizes = np.bincount(holes.ravel())
        small = hsizes < hole_area
        small[0] = False
        processed[kept | small[holes]] = cls
    return processed


def paper_median(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """As published: class-wise binary median filter, later classes overwrite earlier ones."""
    smoothed = np.zeros_like(mask)
    for cls in np.unique(mask):
        filtered = ndimage.median_filter((mask == cls).astype(np.uint8), size=size)
        smoothed[filtered > 0] = cls
    return smoothed


def clean(mask: np.ndarray, min_size: int = 64, valid: np.ndarray | None = None) -> np.ndarray:
    """Remove regions smaller than ``min_size`` px and give their pixels the nearest surviving class."""
    if min_size <= 1:
        return mask.copy()
    removed = np.zeros(mask.shape, bool)
    for cls in np.unique(mask):
        labels, _ = ndimage.label(mask == cls, structure=_EIGHT)
        sizes = np.bincount(labels.ravel())
        small = sizes < min_size
        small[0] = False
        removed |= small[labels]
    if valid is not None:
        removed &= valid
    if not removed.any() or removed.all():
        return mask.copy()
    idx = ndimage.distance_transform_edt(removed, return_distances=False, return_indices=True)
    return mask[tuple(idx)]


def majority_filter(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Multi-class mode filter: each pixel takes the most frequent class in its neighbourhood."""
    counts = np.empty(mask.shape + (NUM_CLASSES,), np.float32)
    for c in range(NUM_CLASSES):
        counts[..., c] = ndimage.uniform_filter((mask == c).astype(np.float32), size=size, mode="nearest")
    # ties resolve towards the pixel's own class
    counts[np.arange(mask.shape[0])[:, None], np.arange(mask.shape[1])[None, :], mask] += 1e-3
    return counts.argmax(-1).astype(mask.dtype)


def min_size_pixels(min_area_m2: float, ground_res_m: float) -> int:
    return max(1, int(round(min_area_m2 / (ground_res_m ** 2))))
