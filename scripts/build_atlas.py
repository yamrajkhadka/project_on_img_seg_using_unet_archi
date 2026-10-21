"""Build (or extend) the national land-cover atlas in data/atlas.

Design: a fixed panel of random sites, stratified by district (at least --floor sites per district, the rest in
proportion to area).  Every site is a 1.22 x 1.22 km window (one 224 x 224 model tile at 5.46 m) classified for
every epoch:

  * eox-YYYY   Sentinel-2 cloudless annual mosaic of that year (2016-2025)
  * esri       Esri World Imagery, latest (sub-metre, sampled like the training data)

with every installed model, plus the Impact Observatory 10 m reference fractions (2017-2025) for validation.
Because the panel is fixed, re-running only adds what is missing (new epochs, a new model).

Usage:
    MONITOR_NEPAL_CACHE=/tmp/tiles python scripts/build_atlas.py --total 1000 --floor 8
    python scripts/build_atlas.py --epochs eox-2026            # add a new year later
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor_nepal import boundaries  # noqa: E402
from monitor_nepal.analysis import stats  # noqa: E402
from monitor_nepal.config import ATLAS_DIR, CLASS_KEYS, MODEL_GSD_M, MODEL_TILE  # noqa: E402
from monitor_nepal.geo import Grid  # noqa: E402
from monitor_nepal.http import FetchError  # noqa: E402
from monitor_nepal.imagery import eox, esri, reference  # noqa: E402
from monitor_nepal.model import MODELS, get_model  # noqa: E402
from monitor_nepal.postprocess import clean  # noqa: E402

DEFAULT_EPOCHS = [f"eox-{y}" for y in eox.years()] + ["esri"]


def epoch_label(epoch: str) -> str:
    if epoch.startswith("eox-"):
        return f"{epoch[4:]} (Sentinel-2 cloudless)"
    return "Latest high-resolution (Esri World Imagery)"


def make_sites(total: int, floor: int, seed: int) -> pd.DataFrame:
    districts = boundaries.regions(2)
    area = sum(d.area_km2 for d in districts)
    rows = []
    for i, d in enumerate(districts):
        n = max(floor, int(round(total * d.area_km2 / area)))
        for lon, lat in boundaries.random_points(d, n, seed=seed + i, margin_m=650):
            rows.append({"lon": round(float(lon), 6), "lat": round(float(lat), 6), "province": d.province,
                         "district": d.name, "district_code": d.code, "stratum_km2": d.area_km2})
    df = pd.DataFrame(rows)
    local = boundaries.locate(df.lon.to_numpy(), df.lat.to_numpy(), level=3)
    df["local"] = [r.name if r else None for r in local]
    df["local_code"] = [r.code if r else None for r in local]
    df.insert(0, "id", [f"a{i:05d}" for i in range(len(df))])
    return df


def source_for(epoch: str):
    if epoch.startswith("eox-"):
        return eox.annual_source(int(epoch[4:])), False
    if epoch == "esri":
        return esri.current_source(), True
    raise ValueError(epoch)


def classify_epoch(sites: pd.DataFrame, epoch: str, models: list[str], workers: int) -> pd.DataFrame:
    src, for_model = source_for(epoch)
    nets = {m: get_model(m) for m in models}
    size = MODEL_TILE * MODEL_GSD_M

    def one(row):
        g = Grid.around(row.lon, row.lat, size, MODEL_GSD_M)
        try:
            sc = src.fetch(g, for_model=for_model, workers=4)
        except FetchError as exc:
            return [], str(exc)
        if sc.valid_fraction < 0.5:
            return [], "no data"
        rgb = eox.normalize(sc.rgb, int(epoch[4:])) if epoch.startswith("eox-") else sc.rgb
        out = []
        for m, net in nets.items():
            proba = net.predict_proba(rgb)
            cls = clean(proba.argmax(-1).astype(np.uint8), 64, valid=sc.valid)
            fr = stats.fractions(cls, sc.valid)
            out.append({"site": row.id, "epoch": epoch, "model": m, **dict(zip(CLASS_KEYS, np.round(fr, 5))),
                        "confidence": round(float(proba.max(-1)[sc.valid].mean()), 4)})
        return out, None

    rows, errors, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (res, err) in enumerate(ex.map(one, sites.itertuples())):
            rows += res
            errors += err is not None
            if (i + 1) % 100 == 0:
                print(f"  {epoch}: {i + 1}/{len(sites)} sites ({(i + 1) / (time.time() - t0):.1f}/s, {errors} failed)",
                      flush=True)
    return pd.DataFrame(rows)


def reference_fractions(sites: pd.DataFrame, years: list[int], workers: int) -> pd.DataFrame:
    size = MODEL_TILE * MODEL_GSD_M

    def one(args):
        row, year = args
        g = Grid.around(row.lon, row.lat, size, 10.0)
        try:
            _, dg = reference.fetch_io(g, year)
        except FetchError:
            return None
        ok = dg != 255
        if ok.mean() < 0.5:
            return None
        return {"site": row.id, "year": year, **dict(zip(CLASS_KEYS, np.round(stats.fractions(dg, ok), 5)))}

    jobs = [(r, y) for y in years for r in sites.itertuples()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = [r for r in ex.map(one, jobs) if r]
    return pd.DataFrame(out)


def add_capture_dates(sites: pd.DataFrame, workers: int = 2) -> pd.DataFrame:
    """Real capture date of the Esri imagery at every site (the 'esri' epoch mixes acquisition years)."""
    latest = esri.wayback_releases()[0]

    def one(row):
        try:
            info = esri.capture_info(round(row.lon, 5), round(row.lat, 5), latest.metadata_url)
            return info.date.isoformat() if info.date else None, info.sensor
        except FetchError:
            return None, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(one, sites.itertuples()))
    sites = sites.copy()
    sites["esri_captured"] = [g[0] for g in got]
    sites["esri_sensor"] = [g[1] for g in got]
    return sites


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--floor", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1768)
    ap.add_argument("--epochs", default=",".join(DEFAULT_EPOCHS))
    ap.add_argument("--models", default=",".join(m.key for m in MODELS.values() if m.available))
    ap.add_argument("--ref-years", default=",".join(str(y) for y in reference.years()))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh", action="store_true", help="recompute epochs that already exist")
    ap.add_argument("--capture-dates", action="store_true", help="record the Esri capture date at every site")
    ap.add_argument("--prefetch", action="store_true",
                    help="only download (cache) the imagery for --epochs, without classifying; needs MONITOR_NEPAL_CACHE")
    args = ap.parse_args()
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)

    sites_path, results_path = ATLAS_DIR / "sites.csv", ATLAS_DIR / "results.csv.gz"
    ref_path, meta_path = ATLAS_DIR / "reference.csv.gz", ATLAS_DIR / "meta.json"
    if sites_path.exists():
        sites = pd.read_csv(sites_path)
    else:
        sites = make_sites(args.total, args.floor, args.seed)
        sites.to_csv(sites_path, index=False)
    print(f"{len(sites)} sites in {sites.district_code.nunique()} districts", flush=True)
    if args.capture_dates or ("esri" in args.epochs.split(",") and "esri_captured" not in sites):
        sites = add_capture_dates(sites)
        sites.to_csv(sites_path, index=False)
        dates = sorted(sites.esri_captured.dropna())
        if dates:
            print(f"capture dates for {len(dates)} sites: {dates[0]} … {dates[-1]}, median {dates[len(dates) // 2]}",
                  flush=True)

    if args.prefetch:
        size = MODEL_TILE * MODEL_GSD_M
        for epoch in [e for e in args.epochs.split(",") if e]:
            src, for_model = source_for(epoch)

            def warm(row, src=src, for_model=for_model):
                try:
                    src.fetch(Grid.around(row.lon, row.lat, size, MODEL_GSD_M), for_model=for_model, workers=4)
                except FetchError:
                    pass

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(warm, sites.itertuples()))
            print(f"prefetched {epoch} in {time.time() - t0:.0f}s", flush=True)
        return

    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame(columns=["site", "epoch", "model"])
    models = [m for m in args.models.split(",") if m]
    for epoch in [e for e in args.epochs.split(",") if e]:
        # The original DeepGlobe network only works on sub-metre imagery; never pair it with Sentinel-2 epochs.
        eligible = [m for m in models if not (m == "paper" and epoch.startswith("eox-"))]
        todo = [m for m in eligible if args.refresh or results[(results.epoch == epoch) & (results.model == m)].empty]
        if not todo:
            continue
        print(f"epoch {epoch} · models {todo}", flush=True)
        new = classify_epoch(sites, epoch, todo, args.workers)
        results = pd.concat([results[~((results.epoch == epoch) & results.model.isin(todo))], new], ignore_index=True)
        results.to_csv(results_path, index=False, compression="gzip")

    refs = pd.read_csv(ref_path) if ref_path.exists() else pd.DataFrame(columns=["site", "year"])
    years = [int(y) for y in args.ref_years.split(",") if y and int(y) not in set(refs.year.astype(int))]
    if years:
        print(f"reference years {years}", flush=True)
        refs = pd.concat([refs, reference_fractions(sites, years, args.workers)], ignore_index=True)
        refs.to_csv(ref_path, index=False, compression="gzip")

    results = results[~((results.model == "paper") & results.epoch.astype(str).str.startswith("eox-"))]
    results.to_csv(results_path, index=False, compression="gzip")
    epochs_present = [e for e in DEFAULT_EPOCHS + sorted(set(results.epoch) - set(DEFAULT_EPOCHS))
                      if e in set(results.epoch)]
    meta = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sites": int(len(sites)),
            "design": {"strata": "district", "floor_per_district": args.floor, "target_total": args.total,
                       "window_m": round(MODEL_TILE * MODEL_GSD_M, 1)},
            "epochs": epochs_present, "epoch_labels": {e: epoch_label(e) for e in epochs_present},
            "models": sorted(set(results.model)), "reference_years": sorted(int(y) for y in set(refs.year))}
    meta_path.write_text(json.dumps(meta, indent=1))
    print("done:", json.dumps(meta)[:400], flush=True)


if __name__ == "__main__":
    main()
