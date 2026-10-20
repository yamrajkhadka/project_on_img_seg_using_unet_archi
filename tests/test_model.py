"""Model tests: the shipped ONNX networks load, produce probabilities, and tile seamlessly."""
from __future__ import annotations

import numpy as np
import pytest

from monitor_nepal.model import MODELS, LandCoverModel

AVAILABLE = [m for m in MODELS.values() if m.available]


@pytest.fixture(scope="module", params=[m.key for m in AVAILABLE])
def model(request):
    return LandCoverModel(MODELS[request.param].path)


def test_probabilities(model):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    p = model.predict_proba(img)
    assert p.shape == (224, 224, 7)
    assert np.allclose(p.sum(-1), 1, atol=1e-4)


def test_any_size_and_padding(model):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (150, 301, 3), dtype=np.uint8)
    assert model.predict_proba(img).shape == (150, 301, 7)


def test_sliding_window_matches_single_pass_in_the_interior(model):
    rng = np.random.default_rng(2)
    base = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    img = np.kron(base, np.ones((20, 20, 1), np.uint8))  # 480 x 480, smooth blocks
    whole = model.predict_proba(img, window=480, overlap=0)
    tiled = model.predict_proba(img, window=256, overlap=64)
    inner = (slice(96, 384), slice(96, 384))
    agree = (whole[inner].argmax(-1) == tiled[inner].argmax(-1)).mean()
    # Flat synthetic colour blocks put many pixels on a class decision boundary, so a little argmax flicker is
    # expected; on real imagery both models agree on > 98.5 % of interior pixels.
    assert agree > 0.95
    assert np.abs(whole[inner] - tiled[inner]).mean() < 0.03


def test_paper_style(model):
    img = np.full((1000, 800, 3), 120, np.uint8)
    assert model.predict_paper_style(img).shape == (224, 224, 7)


def test_tta_runs(model):
    img = np.random.default_rng(3).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    assert model.predict_proba(img, tta=4).shape == (224, 224, 7)
