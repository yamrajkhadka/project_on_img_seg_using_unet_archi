"""Keep every timeline growing: add newly published years (run monthly by .github/workflows/refresh-atlas.yml).

* Sentinel-2 cloudless (EOX): a year is added once its annual mosaic exists and is not blank over Nepal. Its colour
  statistics are then measured at 300 atlas sites and stored in data/history/eox_radiometry.json, which is where the
  app reads its list of years from (see monitor_nepal/imagery/eox.py).
* Impact Observatory 10 m reference map: a year is added once the service returns land classes over Nepal
  (data/history/reference_years.json).

* Rivers: the peak and mean flow of every finished year (data/history/rivers_annual.csv); the "normal" range stays
  the fixed 2004-2023 base period, like climate normals.
* Climate: national temperature and precipitation for the years after the World Bank CCKP record, from ERA5
  (Open-Meteo) at points across Nepal, calibrated to the CCKP series over the ten years they share
  (data/history/climate_recent.csv).
* Events: strong earthquakes (M6+, USGS) and floods above the 10-year level after the curated timeline ends
  (data/history/events_auto.csv).

``scripts/build_atlas.py`` then classifies every year that is not in the atlas yet.

Usage:
    python scripts/extend_years.py [--dry-run]
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
from monitor_nepal.config import ATLAS_DIR, HISTORY_DIR, MODEL_GSD_M, MODEL_TILE  # noqa: E402
from monitor_nepal.geo import Grid  # noqa: E402
from monitor_nepal.history import events, sources  # noqa: E402
from monitor_nepal.http import FetchError, get_json  # noqa: E402
from monitor_nepal.imagery import eox, reference  # noqa: E402
from monitor_nepal.live import rivers  # noqa: E402

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


def extend_rivers(last_full: int) -> list[str]:
    """Append the annual peak and mean flow of every finished year not yet in the river record."""
    path = HISTORY_DIR / "rivers_annual.csv"
    ann = pd.read_csv(path)
    rows, added = [], []
    for r in rivers.RIVERS:
        have = int(ann.loc[ann.river == r.key, "year"].max())
        if have >= last_full:
            continue
        df = rivers.history(r, f"{have + 1}-01-01", f"{last_full}-12-31").dropna()
        for year, g in df.groupby(df.date.dt.year):
            if len(g) < 360:  # only complete years
                continue
            peak = g.loc[g.discharge.idxmax()]
            rows.append({"river": r.key, "year": int(year), "peak": round(float(peak.discharge), 1),
                         "mean": round(float(g.discharge.mean()), 1), "peak_date": peak.date.date().isoformat()})
            added.append(f"{r.key}-{year}")
        time.sleep(10)  # Open-Meteo fair use
    if rows:
        pd.concat([ann, pd.DataFrame(rows)], ignore_index=True).sort_values(["river", "year"]).to_csv(path, index=False)
    return added


ERA5 = "https://archive-api.open-meteo.com/v1/archive"


def era5_national(y0: int, y1: int, n: int = 24) -> pd.DataFrame:
    """Annual mean temperature (degC) and precipitation (mm/year) averaged over ``n`` fixed points across Nepal."""
    pts = boundaries.random_points(boundaries.country(), n, seed=11)
    data = get_json(ERA5, params={
        "latitude": ",".join(f"{lat:.3f}" for _, lat in pts), "longitude": ",".join(f"{lon:.3f}" for lon, _ in pts),
        "start_date": f"{y0}-01-01", "end_date": f"{y1}-12-31", "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Asia/Kathmandu"}, timeout=180)
    per_point = []
    for loc in data if isinstance(data, list) else [data]:
        d = pd.DataFrame({"date": pd.to_datetime(loc["daily"]["time"]), "tas": loc["daily"]["temperature_2m_mean"],
                          "pr": loc["daily"]["precipitation_sum"]})
        per_point.append(d.groupby(d.date.dt.year).agg(tas=("tas", "mean"), pr=("pr", "sum")))
    return pd.concat(per_point).groupby(level=0).mean().rename_axis("year").reset_index()


def extend_climate(last_full: int) -> list[str]:
    """ERA5 years after each CCKP series, calibrated to it over the ten years they share."""
    rows, added = [], []
    base = {key: sources.climate(key, extended=False) for key in sources.CLIMATE}
    last = min(int(df.year.max()) for df in base.values())
    if last >= last_full:
        return added
    era = era5_national(last - 9, last_full)
    overlap = era[era.year.between(last - 9, last)]
    for key, df in base.items():
        var = "tas" if key.endswith("_tas") else "pr"
        ref = df[df.year.between(last - 9, last)].value.mean()
        new = era[era.year > int(df.year.max())]
        values = (new[var] + (ref - overlap[var].mean()) if var == "tas"
                  else new[var] * (ref / overlap[var].mean()))
        rows += [{"key": key, "year": int(y), "value": round(float(v), 2)} for y, v in zip(new.year, values)]
        added += [f"{key}-{int(y)}" for y in new.year]
    if rows:
        pd.DataFrame(rows).to_csv(HISTORY_DIR / "climate_recent.csv", index=False)
    return added


def record_events() -> list[str]:
    """Strong earthquakes and major floods after the curated timeline, as timeline events."""
    since = max(e.year for e in events.EVENTS) + 1
    found = []
    quakes = get_json("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
        "format": "geojson", "starttime": f"{since}-01-01", "minmagnitude": 6, "minlatitude": 26.3,
        "maxlatitude": 30.5, "minlongitude": 80.0, "maxlongitude": 88.3, "orderby": "time-asc"}, timeout=60)
    for f in quakes.get("features", []):
        p = f["properties"]
        when = datetime.fromtimestamp(p["time"] / 1000, timezone.utc)
        found.append({"year": when.year, "title": f"M{p['mag']:.1f} earthquake", "category": "disaster",
                      "detail": f"{p['place']}, {when:%d %b %Y} (USGS)."})
    ann = pd.read_csv(HISTORY_DIR / "rivers_annual.csv")
    thr = rivers.thresholds().set_index("river")
    for r in rivers.RIVERS:
        big = ann[(ann.river == r.key) & (ann.year >= since) & (ann.peak >= thr.loc[r.key, "q10"])]
        for row in big.itertuples():
            found.append({"year": int(row.year), "title": f"Major flood on the {r.name}", "category": "disaster",
                          "detail": f"Peak {row.peak:,.0f} m³/s on {row.peak_date}, above the 10-year flood level "
                                    "(GloFAS)."})
    pd.DataFrame(found, columns=["year", "title", "category", "detail"]).sort_values("year").to_csv(
        events.AUTO_EVENTS, index=False)
    return [f"{e['year']} {e['title']}" for e in found]


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

    last_full = this_year - 1
    for name, step in (("rivers", lambda: extend_rivers(last_full)), ("climate", lambda: extend_climate(last_full)),
                       ("events", record_events)):
        if args.dry_run:
            continue
        try:
            got = step()
            print(f"{name}: {', '.join(got) if got else 'up to date'}", flush=True)
        except (FetchError, KeyError, ValueError) as exc:  # one source down must not block the others
            print(f"{name}: skipped ({exc})", flush=True)

    print("added:", ", ".join(added + [f"reference-{y}" for y in new_ref]) or "nothing new", flush=True)


if __name__ == "__main__":
    main()
