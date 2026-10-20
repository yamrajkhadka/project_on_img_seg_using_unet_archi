"""EOX Sentinel-2 cloudless: one cloud-free, colour-balanced mosaic of the planet per year.

Over Nepal the 2017 layer is blank (pure white at every zoom), so it is left out.  The mosaics of different years
were processed differently, so before classification each year's colours are mapped onto a common reference
(``normalize``): a per-channel gain and offset that matches the year's national mean and spread, measured at
random sites across Nepal, to the 2018-2025 average.  This removes year-wide processing differences that the
network would otherwise read as land-cover change, while keeping every local contrast.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from ..config import HISTORY_DIR
from .base import XYZSource

YEARS = (2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
ATTRIBUTION = ("Sentinel-2 cloudless by EOX IT Services GmbH (contains modified Copernicus Sentinel data); "
               "2016: CC BY 4.0, 2017+: CC BY-NC-SA 4.0")


def layer_name(year: int) -> str:
    return "s2cloudless_3857" if year == 2016 else f"s2cloudless-{year}_3857"


def annual_source(year: int) -> XYZSource:
    if year not in YEARS:
        raise ValueError(f"Sentinel-2 cloudless over Nepal exists for {', '.join(map(str, YEARS))}, not {year}.")
    url = f"https://tiles.maps.eox.at/wmts/1.0.0/{layer_name(year)}/default/g/{{z}}/{{y}}/{{x}}.jpg"
    return XYZSource(key=f"eox-{year}", name="Sentinel-2 cloudless (EOX)", url_template=url,
                     attribution=ATTRIBUTION, max_zoom=15, native_gsd=10, acquired=str(year),
                     label=f"Sentinel-2 cloudless · {year} annual mosaic", meta={"year": year})


@lru_cache(maxsize=1)
def _radiometry() -> dict:
    try:
        return json.loads((HISTORY_DIR / "eox_radiometry.json").read_text())
    except FileNotFoundError:
        return {}


def normalize(rgb: np.ndarray, year: int) -> np.ndarray:
    """Map a year's colours onto the 2018-2025 reference (per-channel gain and offset)."""
    stats = _radiometry()
    y = stats.get("years", {}).get(str(year))
    if not y:
        return rgb
    mean, std = np.asarray(y["mean"], np.float32), np.asarray(y["std"], np.float32)
    ref_mean = np.asarray(stats["reference"]["mean"], np.float32)
    ref_std = np.asarray(stats["reference"]["std"], np.float32)
    out = (rgb.astype(np.float32) - mean) / np.maximum(std, 1e-3) * ref_std + ref_mean
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
