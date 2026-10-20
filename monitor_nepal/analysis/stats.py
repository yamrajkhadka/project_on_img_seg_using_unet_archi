"""Areas, compositions, transitions and agreement metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

from ..config import CLASS_KEYS, CLASS_NAMES, NUM_CLASSES
from ..geo import Grid


def composition(classes: np.ndarray, grid: Grid, valid: np.ndarray | None = None) -> pd.DataFrame:
    """Area (km^2) and share (%) of each class, using the true ground area of every pixel row."""
    area = np.broadcast_to(grid.pixel_area_km2(), classes.shape)
    ok = np.ones(classes.shape, bool) if valid is None else valid
    km2 = np.bincount(classes[ok].ravel(), weights=area[ok].ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    total = km2.sum()
    return pd.DataFrame({"key": CLASS_KEYS, "class": CLASS_NAMES, "km2": km2,
                         "share": km2 / total * 100 if total else np.zeros(NUM_CLASSES)})


def fractions(classes: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    c = classes[valid] if valid is not None else classes.ravel()
    if c.size == 0:
        return np.full(NUM_CLASSES, np.nan)
    return np.bincount(c, minlength=NUM_CLASSES)[:NUM_CLASSES] / c.size


def survey_estimate(frac: np.ndarray, region_km2: float, confidence: float = 0.95) -> pd.DataFrame:
    """Design-based estimate from equal-size cluster samples.

    ``frac`` is (n_samples, 7): the class fractions inside each sample window.  The class share is the
    mean fraction, its standard error ``sd / sqrt(n)``, and the interval uses Student's t.
    """
    frac = frac[~np.isnan(frac).any(1)]
    n = len(frac)
    mean = frac.mean(0) if n else np.full(NUM_CLASSES, np.nan)
    se = frac.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full(NUM_CLASSES, np.nan)
    t = sstats.t.ppf(0.5 + confidence / 2, max(1, n - 1))
    lo, hi = np.clip(mean - t * se, 0, 1), np.clip(mean + t * se, 0, 1)
    return pd.DataFrame({"key": CLASS_KEYS, "class": CLASS_NAMES, "share": mean * 100, "se": se * 100,
                         "low": lo * 100, "high": hi * 100, "km2": mean * region_km2,
                         "km2_low": lo * region_km2, "km2_high": hi * region_km2, "n": n})


def stratified_estimate(frac: np.ndarray, strata: np.ndarray, stratum_km2: dict, region_km2: float | None = None,
                        confidence: float = 0.95, clip: bool = True) -> pd.DataFrame:
    """Stratified estimator: p = sum_h W_h p_h, Var = sum_h W_h^2 s_h^2 / n_h (strata = districts).

    Used for the national atlas, where every district gets a minimum number of sites (so districts can be
    reported) and the rest are allocated in proportion to area.
    """
    ok = ~np.isnan(frac).any(1)
    frac, strata = frac[ok], np.asarray(strata)[ok]
    hs = [h for h in np.unique(strata) if h in stratum_km2]
    total = sum(stratum_km2[h] for h in hs)
    mean = np.zeros(NUM_CLASSES)
    var = np.zeros(NUM_CLASSES)
    n_tot = 0
    for h in hs:
        f = frac[strata == h]
        w = stratum_km2[h] / total
        mean += w * f.mean(0)
        if len(f) > 1:
            var += w ** 2 * f.var(0, ddof=1) / len(f)
        n_tot += len(f)
    se = np.sqrt(var)
    t = sstats.t.ppf(0.5 + confidence / 2, max(1, n_tot - len(hs)))
    lo, hi = mean - t * se, mean + t * se
    if clip:
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    area = region_km2 if region_km2 is not None else total
    return pd.DataFrame({"key": CLASS_KEYS, "class": CLASS_NAMES, "share": mean * 100, "se": se * 100,
                         "low": lo * 100, "high": hi * 100, "km2": mean * area, "km2_low": lo * area,
                         "km2_high": hi * area, "n": n_tot, "strata": len(hs)})


def stratified_change(frac_a: np.ndarray, frac_b: np.ndarray, strata: np.ndarray, stratum_km2: dict,
                      region_km2: float, confidence: float = 0.95) -> pd.DataFrame:
    """Stratified estimate of the paired difference between two epochs observed at the same sites."""
    est = stratified_estimate(frac_b - frac_a, strata, stratum_km2, region_km2, confidence, clip=False)
    dof = max(1, int(est["n"].iloc[0]) - int(est["strata"].iloc[0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * (1 - sstats.t.cdf(np.abs(est["share"] / est["se"]), dof))
    return pd.DataFrame({"key": CLASS_KEYS, "class": CLASS_NAMES, "change_pp": est["share"], "low": est["low"],
                         "high": est["high"], "km2": est["km2"], "p_value": p, "n": est["n"]})


def paired_change(frac_a: np.ndarray, frac_b: np.ndarray, confidence: float = 0.95) -> pd.DataFrame:
    """Change in class share between two epochs observed at the same sample sites (panel design)."""
    ok = ~(np.isnan(frac_a).any(1) | np.isnan(frac_b).any(1))
    d = (frac_b - frac_a)[ok]
    n = len(d)
    mean = d.mean(0) if n else np.full(NUM_CLASSES, np.nan)
    se = d.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full(NUM_CLASSES, np.nan)
    t = sstats.t.ppf(0.5 + confidence / 2, max(1, n - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * (1 - sstats.t.cdf(np.abs(mean / se), max(1, n - 1)))
    return pd.DataFrame({"key": CLASS_KEYS, "class": CLASS_NAMES, "change_pp": mean * 100, "low": (mean - t * se) * 100,
                         "high": (mean + t * se) * 100, "p_value": p, "n": n})


def transitions(a: np.ndarray, b: np.ndarray, grid: Grid, valid: np.ndarray | None = None) -> pd.DataFrame:
    """From-to matrix in km^2 (rows: earlier class, columns: later class)."""
    area = np.broadcast_to(grid.pixel_area_km2(), a.shape)
    ok = np.ones(a.shape, bool) if valid is None else valid
    m = np.bincount((a[ok].astype(np.int64) * NUM_CLASSES + b[ok]).ravel(), weights=area[ok].ravel(),
                    minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    return pd.DataFrame(m, index=CLASS_NAMES, columns=CLASS_NAMES)


def agreement(pred: np.ndarray, ref: np.ndarray, ignore: tuple[int, ...] = (6, 255)) -> dict:
    """Overall agreement, Cohen's kappa and per-class scores of a prediction against a reference map."""
    ok = ~np.isin(ref, ignore)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    np.add.at(cm, (ref[ok].astype(int), pred[ok].astype(int)), 1)
    n = cm.sum()
    if n == 0:
        return {"n": 0}
    oa = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    kappa = (oa - pe) / (1 - pe) if pe < 1 else np.nan
    rows = []
    for i in range(NUM_CLASSES - 1):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        rows.append({"class": CLASS_NAMES[i], "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                     "f1": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn),
                     "ref_share": cm[i, :].sum() / n * 100, "pred_share": cm[:, i].sum() / n * 100})
    return {"n": int(n), "overall": float(oa), "kappa": float(kappa), "confusion": cm,
            "per_class": pd.DataFrame(rows)}
