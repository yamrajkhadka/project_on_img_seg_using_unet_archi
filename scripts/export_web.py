"""Export the project's data for the website in web/ (nepalmonitor.yamrajkhadka.com.np).

Writes web/data/*.json: regions with SVG outlines, the U-Net's land-cover estimates for Nepal, its provinces and
districts (with the year-by-year trend and the reference map), the main town of each district (for weather and air
quality), rivers, climate, events, national statistics and the model card. Live data (weather, air quality, fires,
earthquakes, river flow) is fetched by the website itself. Run after the atlas or the model changes; the monthly
workflow runs it too.

Usage:
    python scripts/export_web.py [--towns]   # --towns re-geocodes the district towns (otherwise reused)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitor_nepal import boundaries  # noqa: E402
from monitor_nepal.analysis import atlas  # noqa: E402
from monitor_nepal.config import CLASSES, HISTORY_DIR, MODELS_DIR  # noqa: E402
from monitor_nepal.history import events, sources  # noqa: E402
from monitor_nepal.http import FetchError, get_json  # noqa: E402
from monitor_nepal.imagery import eox, reference  # noqa: E402
from monitor_nepal.live import rivers  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "web" / "data"
LON0, LAT1, COS = 80.0, 30.5, math.cos(math.radians(28.3))  # equirectangular SVG projection, 100 units per degree
TOLERANCE = {0: 0.01, 1: 0.008, 2: 0.004, 3: 0.0015}  # outline simplification in degrees
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def xy(lon: float, lat: float) -> tuple[float, float]:
    return round((lon - LON0) * COS * 100, 1), round((LAT1 - lat) * 100, 1)


def svg_path(geom) -> str:
    parts = []
    for poly in getattr(geom, "geoms", [geom]):
        for ring in (poly.exterior, *poly.interiors):
            pts = [xy(lon, lat) for lon, lat in ring.coords]
            if len(pts) >= 3:
                parts.append("M" + "L".join(f"{x:g} {y:g}" for x, y in pts) + "Z")
    return "".join(parts)


def bbox(geom) -> list[float]:
    lon0, lat0, lon1, lat1 = geom.bounds
    (x0, y1), (x1, y0) = xy(lon0, lat0), xy(lon1, lat1)
    return [x0, y0, x1, y1]


def dump(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {name} ({(OUT / name).stat().st_size / 1024:,.0f} KB)", flush=True)


def export_regions() -> dict[str, str]:
    """regions.json (metadata) and paths-<level>.json (SVG outlines); returns code -> slug."""
    country = boundaries.country()
    meta, slugs = [], {}
    paths = {0: {country.code: svg_path(boundaries.geometry(country).simplify(TOLERANCE[0]))}}
    for level in (1, 2, 3):
        regs = boundaries.regions(level)
        names = pd.Series([slugify(r.name) for r in regs])
        paths[level] = {}
        for r, base in zip(regs, names):
            # local-unit names repeat across districts: add the district to keep every address unique
            slug = base if (names == base).sum() == 1 else f"{base}-{slugify(r.parent_name or '')}"
            slugs[r.code] = slug
            geom = boundaries.geometry(r)
            paths[level][r.code] = svg_path(geom.simplify(TOLERANCE[level], preserve_topology=True))
            meta.append({"code": r.code, "level": level, "name": r.name, "slug": slug, "parent": r.parent_code,
                         "province": r.province, "area": round(r.area_km2, 1), "lon": round(r.lon, 4),
                         "lat": round(r.lat, 4), "bbox": bbox(geom)})
    dump("regions.json", {"country": {"code": country.code, "name": "Nepal", "area": round(country.area_km2, 1),
                                      "bbox": bbox(boundaries.geometry(country))}, "regions": meta})
    for level, p in paths.items():
        dump(f"paths-{level}.json", p)
    return slugs


def export_landcover() -> None:
    """U-Net estimates (latest high-resolution imagery), the yearly trend and the reference map, per region."""
    out = {}
    for region in [boundaries.country(), *boundaries.regions(1), *boundaries.regions(2)]:
        latest = atlas.national_latest("nepal") if region.level == 0 else None
        est = latest[0] if latest else None
        epoch = atlas.headline_epochs("nepal")[0]
        if est is None:
            est = atlas.estimate(region, epoch, "nepal")
        if est is None:
            continue
        trend = atlas.trend(region, "nepal")
        trend = trend[trend.epoch.str.startswith("eox-")]
        ref = atlas.reference_trend(region)
        out[region.code] = {
            "n": int(est["n"].iloc[0]),
            "imagery": atlas.epoch_label(epoch),
            "latest": {r.key: [round(r.share, 2), round(r.low, 2), round(r.high, 2), round(r.km2, 1)]
                       for r in est.itertuples() if r.key != "unknown"},
            "trend": {str(int(ep.split("-")[1])): {r.key: [round(r.share, 2), round(r.low, 2), round(r.high, 2)]
                                                    for r in g.itertuples() if r.key != "unknown"}
                      for ep, g in trend.groupby("epoch")},
            "reference": {str(int(y)): {r.key: round(r.share, 2) for r in g.itertuples() if r.key != "unknown"}
                          for y, g in ref.groupby("year")} if not ref.empty else {},
        }
    dump("landcover.json", {"classes": [{"key": c.key, "name": c.name, "definition": c.definition}
                                        for c in CLASSES if c.key != "unknown"],
                            "years": list(eox.years()), "reference_years": list(reference.years()), "regions": out})


SEAT_RANK = {"PPLC": 0, "PPLA": 1, "PPLA2": 2, "PPLA3": 3, "PPLA4": 4}  # capital, then administrative seats


def ascii_name(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def export_towns(refresh: bool) -> None:
    """The main town of every district (weather and air-quality pages): populated places inside the district found by
    geocoding the district's and its local units' names, preferring administrative seats, then population."""
    path = OUT / "towns.json"
    if path.exists() and not refresh:
        return
    towns = {}
    for d in boundaries.regions(2):
        names = [d.name] + [re.sub(r"\s+(Municipality|Rural Municipality|Sub-Metropolitan City|Metropolitan City)$",
                                   "", u.name) for u in boundaries.children(d)]
        hits = {}
        for name in names:
            try:
                res = get_json(GEOCODE, params={"name": name, "count": 10, "countryCode": "NP"}, timeout=30)
            except FetchError:
                continue
            for h in res.get("results", []):
                if str(h.get("feature_code", "")).startswith("PPL") and boundaries.contains(d, h["longitude"], h["latitude"]):
                    hits[h["id"]] = h
            time.sleep(0.15)
        if hits:
            h = min(hits.values(), key=lambda h: (SEAT_RANK.get(h.get("feature_code"), 9), -(h.get("population") or 0)))
            best = (ascii_name(h["name"]), h["latitude"], h["longitude"], h.get("population") or 0, h.get("feature_code"))
        else:  # no populated place found: the district centre
            best = (d.name, d.lat, d.lon, 0, "centre")
        towns[d.code] = {"town": best[0], "lat": round(best[1], 4), "lon": round(best[2], 4), "population": best[3],
                         "kind": best[4]}
        print(f"{d.name}: {best[0]} ({best[4]}, {best[3]:,})", flush=True)
    dump("towns.json", towns)


def export_static() -> None:
    dump("rivers.json", {
        "rivers": [{"key": r.key, "name": r.name, "gauge": r.gauge, "lat": r.lat, "lon": r.lon, "province": r.province}
                   for r in rivers.RIVERS],
        "climatology": {k: g[["doy", "p10", "p50", "p90"]].round(1).values.tolist()
                        for k, g in rivers.climatology().groupby("river")},
        "thresholds": rivers.thresholds().set_index("river")[["q2", "q10", "mean"]].round(1).to_dict("index"),
        "annual": {k: g[["year", "peak", "mean", "peak_date"]].values.tolist() for k, g in rivers.annual().groupby("river")},
    })
    dump("climate.json", {key: {"name": spec[1], "unit": spec[2],
                                "base_end": int(sources.climate(key, extended=False).year.max()),
                                "series": sources.climate(key)[["year", "value"]].round(2).values.tolist()}
                          for key, spec in sources.CLIMATE.items()})
    dump("events.json", [{"year": e.year, "end": e.end_year, "title": e.title, "category": e.category,
                          "detail": e.detail} for e in events.all_events()])
    stats = {}
    for key, ind in sources.WB_INDICATORS.items():
        snap = HISTORY_DIR / f"wb_{key}.csv"
        if snap.exists():
            df = pd.read_csv(snap).dropna()
            stats[key] = {"name": ind.name, "unit": ind.unit, "series": df.iloc[:, :2].values.tolist()}
    dump("national.json", stats)
    log = json.loads((MODELS_DIR / "finetune_log.json").read_text())
    ev = json.loads((MODELS_DIR / "evaluation.json").read_text())
    dump("model.json", {
        "val_miou": round(log["epochs"][log["best_epoch"] - 1]["val"]["all"]["miou"], 3),
        "baseline_miou": round(log["baseline"]["all"]["miou"], 3), "summary": log.get("summary"),
        "updated": log.get("updated"), "evaluation": ev["description"],
        "results": [{"model": r["model_key"], "source": r["source"], "reference": r["reference"],
                     "overall": round(r["overall"], 3), "kappa": round(r["kappa"], 3), "miou": round(r["miou"], 3)}
                    for r in ev["results"]],
        "inter_reference": ev.get("inter_reference"),
    })


def export_hazards() -> None:
    """Fire detections per year, month and district (NASA FIRMS archive) and earthquakes per year (USGS), adding
    only years not already exported, so the monthly run just appends each newly published year."""
    from datetime import datetime, timezone

    from monitor_nepal.live import fires

    path = OUT / "hazards.json"
    doc = json.loads(path.read_text()) if path.exists() else {"fires": {}, "quakes": {}}
    last = fires.latest_archive_year()
    for product, (_, first) in fires.ARCHIVE_PRODUCTS.items():
        have = doc["fires"].setdefault(product, {})
        for year in range(max(first, 2001), last + 1):
            if str(year) in have:
                continue
            try:
                df = fires.archive(year, product)
            except (FetchError, ValueError) as exc:
                print(f"fires {product} {year}: skipped ({exc})", flush=True)
                continue
            local = df.time_utc.dt.tz_convert("Asia/Kathmandu")
            have[str(year)] = {
                "total": int(len(df)),
                "months": [int((local.dt.month == m).sum()) for m in range(1, 13)],
                "districts": {k: int(v) for k, v in df.district.value_counts().items()},
            }
            print(f"fires {product} {year}: {len(df):,}", flush=True)
            time.sleep(1)
    this_year = datetime.now(timezone.utc).year
    for year in range(1950, this_year):
        if str(year) in doc["quakes"]:
            continue
        counts = {}
        for mag in (4, 5, 6):
            try:
                n = get_json("https://earthquake.usgs.gov/fdsnws/event/1/count", params={
                    "format": "geojson", "starttime": f"{year}-01-01", "endtime": f"{year + 1}-01-01", "minmagnitude": mag,
                    "minlatitude": 25.4, "maxlatitude": 31.3, "minlongitude": 79.2, "maxlongitude": 89.2}, timeout=60)
                counts[f"m{mag}"] = int(n["count"])
            except (FetchError, KeyError, TypeError):
                counts = {}
                break
        if counts:
            doc["quakes"][str(year)] = counts
    dump("hazards.json", doc)


def export_tools() -> None:
    """What the in-browser AI tools need: both models (served from web/public) and the yearly colour statistics."""
    import shutil

    public = OUT.parent / "public" / "models"
    public.mkdir(parents=True, exist_ok=True)
    for name in ("unet_nepal.onnx", "unet_deepglobe.onnx"):
        src, dst = MODELS_DIR / name, public / name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dst)
            print(f"copied {name}", flush=True)
    shutil.copyfile(eox.RADIOMETRY, OUT / "eox_radiometry.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--towns", action="store_true", help="re-geocode the main town of every district")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    export_regions()
    export_landcover()
    export_static()
    export_tools()
    export_hazards()
    export_towns(args.towns)


if __name__ == "__main__":
    main()
