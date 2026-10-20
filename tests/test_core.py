"""Offline tests for the core library (run: pytest -q).  Network tests live in test_network.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from monitor_nepal import boundaries
from monitor_nepal.analysis import stats
from monitor_nepal.config import CLASSES, MODEL_GSD_M, NUM_CLASSES, PALETTES
from monitor_nepal.geo import Grid, lonlat_to_merc, merc_to_lonlat, tile_bounds, zoom_resolution
from monitor_nepal.postprocess import clean, majority_filter, paper_median, paper_morphology
from monitor_nepal.resample import cv2_linear


# ---- configuration -------------------------------------------------------------------------------------------
def test_classes_match_paper_table():
    assert [c.paper_rgb for c in CLASSES] == [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255),
                                             (255, 255, 255), (0, 0, 0)]
    assert NUM_CLASSES == 7 and len(PALETTES["cartographic"]) == 7
    assert MODEL_GSD_M == pytest.approx(5.464, abs=1e-3)


# ---- geometry ------------------------------------------------------------------------------------------------
def test_mercator_roundtrip():
    for lon, lat in [(85.324, 27.7172), (80.06, 26.35), (88.2, 30.44)]:
        x, y = lonlat_to_merc(lon, lat)
        lo, la = merc_to_lonlat(x, y)
        assert lo == pytest.approx(lon, abs=1e-9) and la == pytest.approx(lat, abs=1e-9)


def test_tiles():
    assert zoom_resolution(0) == pytest.approx(156543.03392804097)
    x0, y0, x1, y1 = tile_bounds(0, 0, 0)
    assert x1 - x0 == pytest.approx(2 * 20037508.342789244)


def test_grid_ground_resolution_and_area():
    g = Grid.around(85.324, 27.7172, 1224, MODEL_GSD_M)
    assert g.width == g.height == 224
    assert g.ground_res == pytest.approx(MODEL_GSD_M, rel=1e-3)
    assert g.area_km2 == pytest.approx(1.224 ** 2, rel=0.01)
    # a 1x1 degree box near 28N is ~ 111.32 * cos(28) * 110.6 km^2
    g2 = Grid.from_lonlat_bbox((84.0, 27.5, 85.0, 28.5), 100.0)
    assert g2.area_km2 == pytest.approx(111.32 * math.cos(math.radians(28)) * 110.6, rel=0.02)


# ---- resampling ------------------------------------------------------------------------------------------------
def test_cv2_linear_identity_and_formula():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    assert np.array_equal(cv2_linear(img, 53, 37), img)
    # 2x2 -> 4x4 upsampling follows OpenCV's centre-aligned, edge-clamped formula
    small = np.array([[0, 100], [200, 255]], np.uint8)
    up = cv2_linear(small, 4, 4)
    assert up[0, 0] == 0 and up[3, 3] == 255 and up[0, 1] == 25 and up[0, 2] == 75


# ---- post-processing ----------------------------------------------------------------------------------------------
def _toy():
    m = np.full((40, 40), 3, np.uint8)  # forest
    m[5:7, 5:7] = 1  # a 4 px speck of agriculture
    m[20:40, 20:40] = 1  # a large agriculture block
    return m


def test_paper_morphology_quirk_turns_specks_urban():
    out = paper_morphology(_toy(), min_size=64, hole_area=0)
    # the removed speck is not re-assigned, so it becomes class 0 (urban) - the published behaviour
    assert (out[5:7, 5:7] == 0).all()


def test_clean_fills_specks_with_surrounding_class():
    out = clean(_toy(), 64)
    assert (out[5:7, 5:7] == 3).all()
    assert (out[20:40, 20:40] == 1).all()


def test_paper_median_and_majority_shapes():
    m = _toy()
    assert paper_median(m).shape == m.shape
    maj = majority_filter(m, 3)
    assert maj.dtype == m.dtype and (maj[25:35, 25:35] == 1).all()


# ---- statistics -------------------------------------------------------------------------------------------------
def test_survey_estimate_and_ci():
    rng = np.random.default_rng(1)
    frac = rng.dirichlet(np.ones(7), size=200)
    est = stats.survey_estimate(frac, 1000.0)
    assert est["share"].sum() == pytest.approx(100, abs=1e-6)
    assert (est["low"] <= est["share"]).all() and (est["share"] <= est["high"]).all()
    assert est["km2"].sum() == pytest.approx(1000.0)


def test_stratified_equals_mean_under_proportional_allocation():
    rng = np.random.default_rng(2)
    a = rng.dirichlet(np.ones(7), size=30)
    b = rng.dirichlet(np.ones(7), size=60)
    frac = np.vstack([a, b])
    strata = np.array(["A"] * 30 + ["B"] * 60)
    est = stats.stratified_estimate(frac, strata, {"A": 100.0, "B": 200.0})
    assert est["share"].to_numpy() / 100 == pytest.approx(frac.mean(0), abs=1e-9)


def test_transitions_and_agreement():
    g = Grid.around(85.3, 27.7, 500, 5.0)
    a = np.zeros(g.shape, np.uint8)
    b = a.copy()
    b[: g.height // 2] = 3
    m = stats.transitions(a, b, g)
    assert m.values.sum() == pytest.approx(g.area_km2, rel=1e-6)
    agr = stats.agreement(b, b)
    assert agr["overall"] == 1.0 and agr["kappa"] == pytest.approx(1.0)


# ---- boundaries ---------------------------------------------------------------------------------------------------
def test_boundaries_counts_and_lookup():
    assert len(boundaries.regions(1)) == 7
    assert len(boundaries.regions(2)) == 77
    assert len(boundaries.regions(3)) >= 753
    ktm = boundaries.locate(np.array([85.324]), np.array([27.7172]), level=2)[0]
    assert ktm is not None and ktm.name == "Kathmandu" and ktm.province == "Bagmati"
    assert boundaries.locate(np.array([77.2]), np.array([28.6]), level=1)[0] is None  # Delhi


def test_random_points_inside_region():
    d = next(r for r in boundaries.regions(2) if r.name == "Bhaktapur")
    pts = boundaries.random_points(d, 25, seed=3)
    assert len(pts) == 25
    assert all(boundaries.contains(d, lon, lat) for lon, lat in pts)
