"""The national land-cover atlas: a fixed panel of sample sites classified for every epoch.

Built offline by ``scripts/build_atlas.py`` (and refreshed by the scheduled workflow).  Because the same
sites are observed in every epoch, change between epochs is estimated from paired differences, which is
far more precise than comparing two independent snapshots.

Files in ``data/atlas``:
    sites.csv        id, lon, lat, province, district, district_code, local, local_code, stratum_km2
    results.csv.gz   site, epoch, model, <7 class fractions>, confidence
    reference.csv.gz site, year, <7 class fractions>   (Impact Observatory 10 m, for validation)
    meta.json        build information and epoch labels
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import boundaries
from ..boundaries import Region
from ..config import ATLAS_DIR, CLASS_KEYS
from . import stats


@lru_cache(maxsize=1)
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict] | None:
    try:
        sites = pd.read_csv(ATLAS_DIR / "sites.csv")
        results = pd.read_csv(ATLAS_DIR / "results.csv.gz")
        ref_path = ATLAS_DIR / "reference.csv.gz"
        reference = pd.read_csv(ref_path) if ref_path.exists() else pd.DataFrame()
        meta = json.loads((ATLAS_DIR / "meta.json").read_text())
    except FileNotFoundError:
        return None
    return sites, results, reference, meta


def available() -> bool:
    return load() is not None


def _site_ids(region: Region, sites: pd.DataFrame) -> np.ndarray:
    if region.level == 0:
        return sites.id.to_numpy()
    if region.level == 1:
        return sites[sites.province == region.name].id.to_numpy()
    if region.level == 2:
        return sites[sites.district_code == region.code].id.to_numpy()
    return sites[sites.local_code == region.code].id.to_numpy()


def epochs(model: str) -> list[str]:
    loaded = load()
    if loaded is None:
        return []
    _, results, _, meta = loaded
    have = set(results[results.model == model].epoch)
    return [e for e in meta["epochs"] if e in have]


def esri_capture_span() -> tuple[str, str, str] | None:
    """(earliest, median, latest) capture date of the Esri imagery at the atlas sites."""
    loaded = load()
    if loaded is None or "esri_captured" not in loaded[0]:
        return None
    dates = sorted(loaded[0].esri_captured.dropna().astype(str))
    if not dates:
        return None
    return dates[0], dates[len(dates) // 2], dates[-1]


def epoch_label(epoch: str) -> str:
    loaded = load()
    label = loaded[3]["epoch_labels"].get(epoch, epoch) if loaded else epoch
    if epoch == "esri":
        span = esri_capture_span()
        if span:
            label = f"Latest high-resolution imagery (captured {span[0][:4]}–{span[2][:4]}, median {span[1][:7]})"
    return label


def _estimate_rows(region: Region, rows: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame | None:
    """Stratified (by district) estimate for country/province/district; simple mean for a local unit."""
    if len(rows) < 3:
        return None
    frac = rows[CLASS_KEYS].to_numpy(float)
    if region.level == 3:
        return stats.survey_estimate(frac, region.area_km2)
    strata = rows.site.map(sites.set_index("id").district_code).to_numpy()
    areas = sites.groupby("district_code").stratum_km2.first().to_dict()
    return stats.stratified_estimate(frac, strata, areas, region.area_km2)


def estimate(region: Region, epoch: str, model: str) -> pd.DataFrame | None:
    loaded = load()
    if loaded is None:
        return None
    sites, results, _, _ = loaded
    ids = _site_ids(region, sites)
    r = results[(results.model == model) & (results.epoch == epoch) & results.site.isin(ids)]
    return _estimate_rows(region, r, sites)


def trend(region: Region, model: str) -> pd.DataFrame:
    """Long table: epoch, label, key, share, low, high, n."""
    rows = []
    for ep in epochs(model):
        est = estimate(region, ep, model)
        if est is None:
            continue
        est = est.assign(epoch=ep, label=epoch_label(ep))
        rows.append(est)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def reference_trend(region: Region) -> pd.DataFrame:
    loaded = load()
    if loaded is None or loaded[2].empty:
        return pd.DataFrame()
    sites, _, reference, _ = loaded
    ids = _site_ids(region, sites)
    rows = []
    for year, g in reference[reference.site.isin(ids)].groupby("year"):
        est = _estimate_rows(region, g, sites)
        if est is not None:
            rows.append(est.assign(year=int(year)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def change(region: Region, model: str, epoch_a: str, epoch_b: str) -> pd.DataFrame | None:
    loaded = load()
    if loaded is None:
        return None
    sites, results, _, _ = loaded
    ids = _site_ids(region, sites)
    r = results[(results.model == model) & results.site.isin(ids)]
    a = r[r.epoch == epoch_a].set_index("site")[CLASS_KEYS]
    b = r[r.epoch == epoch_b].set_index("site")[CLASS_KEYS]
    common_ids = a.index.intersection(b.index)
    if len(common_ids) < 3:
        return None
    fa, fb = a.loc[common_ids].to_numpy(float), b.loc[common_ids].to_numpy(float)
    if region.level == 3:
        out = stats.paired_change(fa, fb)
        out["km2"] = out["change_pp"] / 100 * region.area_km2
        return out
    strata = pd.Series(common_ids).map(sites.set_index("id").district_code).to_numpy()
    areas = sites.groupby("district_code").stratum_km2.first().to_dict()
    return stats.stratified_change(fa, fb, strata, areas, region.area_km2)


def headline_epochs(model: str) -> list[str]:
    """Epochs for headline figures, best first: the sub-metre Esri imagery the network was designed for (it agrees
    best with the reference maps nationally), then the annual Sentinel-2 mosaics, newest first."""
    eps = epochs(model)
    annual = [e for e in eps if e.startswith("eox-")]
    return [e for e in eps if e not in annual] + list(reversed(annual))


def national_latest(model: str) -> tuple[pd.DataFrame, str, int] | None:
    """Headline national estimate (see ``headline_epochs``)."""
    nepal = boundaries.country()
    for ep in headline_epochs(model):
        est = estimate(nepal, ep, model)
        if est is not None:
            return est, epoch_label(ep), int(est["n"].iloc[0])
    return None


def district_table(model: str, epoch: str) -> pd.DataFrame:
    """Per-district class shares (%) with standard errors, from the sites in each district."""
    loaded = load()
    if loaded is None:
        return pd.DataFrame()
    sites, results, _, _ = loaded
    r = results[(results.model == model) & (results.epoch == epoch)].merge(
        sites[["id", "district", "district_code", "province"]], left_on="site", right_on="id")
    rows = []
    for (code, name, prov), g in r.groupby(["district_code", "district", "province"]):
        f = g[CLASS_KEYS].to_numpy(float)
        row = {"district_code": code, "district": name, "province": prov, "n": len(g)}
        for i, k in enumerate(CLASS_KEYS):
            row[k] = f[:, i].mean() * 100
            row[f"{k}_se"] = f[:, i].std(ddof=1) / np.sqrt(len(g)) * 100 if len(g) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def site_table(model: str, epoch: str) -> pd.DataFrame:
    loaded = load()
    if loaded is None:
        return pd.DataFrame()
    sites, results, _, _ = loaded
    r = results[(results.model == model) & (results.epoch == epoch)]
    df = sites.merge(r, left_on="id", right_on="site")
    df["dominant"] = df[CLASS_KEYS].to_numpy(float).argmax(1)
    return df
