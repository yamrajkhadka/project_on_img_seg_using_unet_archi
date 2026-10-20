"""End-to-end analyses: classify an area wall-to-wall, or estimate a whole region by sampling."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .. import boundaries, imagery
from ..boundaries import Region
from ..config import CLASS_KEYS, MODEL_GSD_M, MODEL_TILE, NUM_CLASSES
from ..geo import Grid
from ..http import FetchError
from ..imagery import ImagerySpec, Scene
from ..model import get_model
from ..postprocess import clean
from . import stats

ProgressFn = Callable[[float, str], None]
HIGH_RES = {"esri", "wayback"}
PAPER_MIN_OBJECT_PX = 64  # the paper's morphological threshold (64 px = 0.19 ha at 5.46 m)


def _noop(_f: float, _t: str) -> None:
    pass


@dataclass
class AreaResult:
    scene: Scene
    classes: np.ndarray  # post-processed
    raw: np.ndarray
    confidence: np.ndarray
    composition: pd.DataFrame
    model_key: str
    spec: ImagerySpec
    meta: dict = field(default_factory=dict)

    @property
    def grid(self) -> Grid:
        return self.scene.grid


def model_input(scene: Scene, spec: ImagerySpec | None) -> np.ndarray:
    """The pixels the network sees: annual mosaics get their year's radiometric normalisation."""
    if spec is not None and spec.kind == "eox" and spec.year:
        from ..imagery import eox

        return eox.normalize(scene.rgb, int(spec.year))
    return scene.rgb


def classify_scene(scene: Scene, model_key: str, *, tta: int = 1, min_object_px: int = PAPER_MIN_OBJECT_PX,
                   progress: ProgressFn | None = None,
                   spec: ImagerySpec | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proba = get_model(model_key).predict_proba(model_input(scene, spec), tta=tta, progress=progress)
    raw = proba.argmax(-1).astype(np.uint8)
    conf = proba.max(-1).astype(np.float16)
    raw[~scene.valid] = 6
    out = clean(raw, min_object_px, valid=scene.valid) if min_object_px > 1 else raw.copy()
    out[~scene.valid] = 6
    return out, raw, conf


def classify_area(grid: Grid, spec: ImagerySpec, model_key: str, *, tta: int = 1,
                  min_object_px: int = PAPER_MIN_OBJECT_PX, progress: ProgressFn | None = None) -> AreaResult:
    progress = progress or _noop
    progress(0.02, "Fetching imagery…")
    scene = imagery.fetch(spec, grid, progress=lambda f, t: progress(0.02 + 0.48 * f, t),
                          for_model=spec.kind in HIGH_RES)
    if scene.valid_fraction < 0.05:
        raise FetchError("The imagery source has no data for this area and date. Try another date or source.")
    classes, raw, conf = classify_scene(scene, model_key, tta=tta, min_object_px=min_object_px,
                                        progress=lambda f, t: progress(0.5 + 0.48 * f, t), spec=spec)
    comp = stats.composition(classes, grid, scene.valid)
    progress(1.0, "Done")
    return AreaResult(scene, classes, raw, conf, comp, model_key, spec)


@dataclass
class SurveyResult:
    region: Region
    spec: ImagerySpec
    model_key: str
    samples: pd.DataFrame  # one row per sample window, class fractions in columns CLASS_KEYS
    estimate: pd.DataFrame
    thumbnails: dict = field(default_factory=dict)  # sample id -> (rgb, classes) small arrays

    @property
    def fractions(self) -> np.ndarray:
        return self.samples[CLASS_KEYS].to_numpy(dtype=float)


def sample_points(region: Region, n: int, seed: int = 0) -> np.ndarray:
    half = MODEL_TILE * MODEL_GSD_M / 2
    return boundaries.random_points(region, n, seed=seed, margin_m=half if region.area_km2 > 50 else 0)


def survey(region: Region, spec: ImagerySpec, model_key: str, n: int = 40, seed: int = 0, *, tta: int = 1,
           points: np.ndarray | None = None, keep_thumbnails: int = 12, workers: int = 4,
           progress: ProgressFn | None = None) -> SurveyResult:
    """Classify ``n`` random 1.2 km windows in the region and estimate its composition with 95 % intervals."""
    progress = progress or _noop
    pts = points if points is not None else sample_points(region, n, seed)
    model = get_model(model_key)
    size = MODEL_TILE * MODEL_GSD_M

    def one(i_pt):
        i, (lon, lat) = i_pt
        g = Grid.around(float(lon), float(lat), size, MODEL_GSD_M)
        try:
            sc = imagery.fetch(spec, g, for_model=spec.kind in HIGH_RES)
        except FetchError as exc:
            return i, lon, lat, None, None, str(exc)
        if sc.valid_fraction < 0.5:
            return i, lon, lat, sc, None, "no imagery"
        proba = model.predict_proba(model_input(sc, spec), tta=tta)
        cls = clean(proba.argmax(-1).astype(np.uint8), PAPER_MIN_OBJECT_PX, valid=sc.valid)
        return i, lon, lat, sc, (cls, float(proba.max(-1)[sc.valid].mean())), None

    rows, thumbs, done = [], {}, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, lon, lat, sc, out, err in ex.map(one, enumerate(pts)):
            done += 1
            progress(done / len(pts), f"Classified {done}/{len(pts)} sample windows")
            row = {"id": i, "lon": float(lon), "lat": float(lat), "error": err}
            if out is None:
                row.update({k: np.nan for k in CLASS_KEYS})
            else:
                cls, conf = out
                fr = stats.fractions(cls, sc.valid)
                row.update(dict(zip(CLASS_KEYS, fr)))
                row.update(confidence=conf, dominant=CLASS_KEYS[int(np.nanargmax(fr))], acquired=sc.acquired)
                if len(thumbs) < keep_thumbnails:
                    thumbs[i] = (sc.rgb[::2, ::2].copy(), cls[::2, ::2].copy())
            rows.append(row)
    samples = pd.DataFrame(rows)
    est = stats.survey_estimate(samples[CLASS_KEYS].to_numpy(dtype=float), region.area_km2)
    return SurveyResult(region, spec, model_key, samples, est, thumbs)


def reference_survey(points: np.ndarray, year: int) -> np.ndarray:
    """Reference (Impact Observatory) class fractions in the same windows, for validation."""
    from ..imagery import reference

    size = MODEL_TILE * MODEL_GSD_M

    def one(pt):
        g = Grid.around(float(pt[0]), float(pt[1]), size, 10.0)
        try:
            _, dg = reference.fetch_io(g, year)
        except FetchError:
            return np.full(NUM_CLASSES, np.nan)
        ok = dg != 255
        return stats.fractions(dg, ok) if ok.any() else np.full(NUM_CLASSES, np.nan)

    with ThreadPoolExecutor(max_workers=6) as ex:
        return np.array(list(ex.map(one, points)))
