"""Add newly published years so the app's timeline keeps growing (run monthly by .github/workflows/refresh-atlas.yml).

* Sentinel-2 cloudless (EOX): a year is added once its annual mosaic exists and is not blank over Nepal. Its colour
  statistics are then measured at 300 atlas sites and stored in data/history/eox_radiometry.json, which is where the
  app reads its list of years from (see monitor_nepal/imagery/eox.py).
* Impact Observatory 10 m reference map: a year is added once the service returns land classes over Nepal
  (data/history/reference_years.json).

``scripts/build_atlas.py`` then classifies every year that is not in the atlas yet.

Usage:
    python scripts/extend_years.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitor_nepal.config import ATLAS_DIR, MODEL_GSD_M, MODEL_TILE  # noqa: E402
from monitor_nepal.geo import Grid  # noqa: E402
from monitor_nepal.http import FetchError  # noqa: E402
from monitor_nepal.imagery import eox, reference  # noqa: E402

KATHMANDU = (85.32, 27.70)
WINDOW_M = MODEL_TILE * MODEL_GSD_M


def eox_available(year: int) -> bool:
    """The mosaic exists and shows land (the blank 2017 layer is pure white)."""
    try:
        scene = eox.annual_source(year, check=False).fetch(Grid.around(*KATHMANDU, WINDOW_M, MODEL_GSD_M))
    except (FetchError, ValueError):
        return False
    return scene.valid_fraction > 0.5 and float(scene.rgb[scene.valid].std()) > 5


def eox_radiometry(year: int, n: int = 300) -> dict:
    """National per-channel RGB mean and standard deviation of the year's mosaic at ``n`` atlas sites."""
    sites = pd.read_csv(ATLAS_DIR / "sites.csv").sample(n, random_state=0)
    src = eox.annual_source(year, check=False)

    def pixels(row):
        try:
            scene = src.fetch(Grid.around(row.lon, row.lat, WINDOW_M, MODEL_GSD_M))
        except FetchError:
            return None
        return scene.rgb[scene.valid].astype(np.float64) if scene.valid.any() else None

    with ThreadPoolExecutor(max_workers=6) as ex:
        parts = [p for p in ex.map(pixels, sites.itertuples()) if p is not None]
    if len(parts) < n // 2:
        raise RuntimeError(f"only {len(parts)} of {n} sites returned imagery for {year}")
    px = np.concatenate(parts)
    return {"mean": [round(float(v), 2) for v in px.mean(0)], "std": [round(float(v), 2) for v in px.std(0)]}


def reference_available(year: int) -> bool:
    try:
        _, dg = reference.fetch_io(Grid.around(*KATHMANDU, WINDOW_M, 10.0), year, check=False)
    except FetchError:
        return False
    return float((dg != reference.NODATA).mean()) > 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report new years without writing anything")
    args = ap.parse_args()
    this_year = datetime.now(timezone.utc).year
    added = []

    radiometry = json.loads(eox.RADIOMETRY.read_text())
    for year in range(max(eox.years()) + 1, this_year + 1):
        if not eox_available(year):
            print(f"Sentinel-2 cloudless {year}: not published yet", flush=True)
            break
        stats = eox_radiometry(year)
        print(f"Sentinel-2 cloudless {year}: available, colour statistics {stats}", flush=True)
        radiometry["years"][str(year)] = stats
        added.append(f"eox-{year}")
    if added and not args.dry_run:
        eox.RADIOMETRY.write_text(json.dumps(radiometry, indent=1))

    ref_years = list(reference.years())
    new_ref = []
    for year in range(max(ref_years) + 1, this_year + 1):
        if not reference_available(year):
            print(f"Reference map {year}: not published yet", flush=True)
            break
        print(f"Reference map {year}: available", flush=True)
        new_ref.append(year)
    if new_ref and not args.dry_run:
        doc = json.loads(reference.YEARS_FILE.read_text()) if reference.YEARS_FILE.exists() else {}
        doc["years"] = ref_years + new_ref
        reference.YEARS_FILE.write_text(json.dumps(doc, indent=1))

    print("added:", ", ".join(added + [f"reference-{y}" for y in new_ref]) or "nothing new", flush=True)


if __name__ == "__main__":
    main()
