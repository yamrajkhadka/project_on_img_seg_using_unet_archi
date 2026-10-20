"""Build a weakly-labelled Nepal land-cover dataset for fine-tuning the paper's U-Net.

The paper (Sec. 3.7/4.4) names the missing piece for Nepal: a labelled set of Nepalese patches.  This
script creates one automatically.  For sites spread over every district it pairs 224x224 image
patches at the model's 5.46 m scale with labels from the Impact Observatory / Esri 10 m annual land
cover of the *same year*, mapped to the seven DeepGlobe classes:

  * Esri World Imagery (sub-metre, sampled exactly like the training data)  + IO 2024 labels
  * Sentinel-2 cloudless annual mosaic (EOX)                                + IO labels, same year
  * Sentinel-2 L2A least-cloudy seasonal mosaic (Planetary Computer)        + IO labels, same year
  * Landsat 8/9 least-cloudy seasonal mosaic (Planetary Computer)           + IO labels, same year

Eleven whole districts are held out for validation so the evaluation is spatially independent.

Labels are *weak*: IO is itself a machine-learned product (roughly 75-85 % accurate), so a model
fine-tuned on it inherits its biases.  Evaluate against an independent map (ESA WorldCover) - see
scripts/evaluate_models.py.

Licences: imagery and labels remain under their providers' terms (Esri World Imagery; EOX
CC BY-NC-SA 4.0; Copernicus Sentinel-2; USGS Landsat public domain; IO LULC CC BY 4.0).  Use the
resulting dataset for research and non-commercial purposes.

Usage:
    MONITOR_NEPAL_CACHE=/tmp/tiles python scripts/build_nepal_dataset.py --out /data/nepal_lc
    python scripts/build_nepal_dataset.py --out /data/nepal_lc --pack
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor_nepal import boundaries, imagery  # noqa: E402
from monitor_nepal.config import MODEL_GSD_M, NEPAL_BBOX  # noqa: E402
from monitor_nepal.geo import Grid, lonlat_to_merc  # noqa: E402
from monitor_nepal.http import get  # noqa: E402
from monitor_nepal.imagery import planetary, reference  # noqa: E402

HOLDOUT_DISTRICTS = ["Ilam", "Sunsari", "Dhanusa", "Lalitpur", "Rasuwa", "Kaski", "Mustang", "Rupandehi",
                     "Jumla", "Surkhet", "Kailali"]
YEARS = list(range(2017, 2026))
SEASONS = {"post-monsoon": ((10, 1), (12, 31)), "dry": ((1, 1), (4, 30))}
PATCH = 224


def national_reference(year: int = 2024, res_m: float = 250.0) -> tuple[np.ndarray, Grid]:
    """IO land cover for all of Nepal at coarse resolution (one request), used to find rare classes."""
    x0, y0 = lonlat_to_merc(NEPAL_BBOX[0], NEPAL_BBOX[1])
    x1, y1 = lonlat_to_merc(NEPAL_BBOX[2], NEPAL_BBOX[3])
    g = Grid(x0, y1, res_m, int((x1 - x0) / res_m), int((y1 - y0) / res_m))
    t0 = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t1 = int(datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc).timestamp() * 1000)
    r = get(f"{reference.SERVICE}/exportImage", params={
        "bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": 3857, "imageSR": 3857, "size": f"{g.width},{g.height}",
        "format": "tiff", "pixelType": "U8", "time": f"{t0},{t1}", "renderingRule": '{"rasterFunction":"None"}',
        "interpolation": "RSP_NearestNeighbor", "f": "image"}, timeout=180)
    return np.asarray(Image.open(io.BytesIO(r.content))), g


def sample_sites(per_district: int, urban: int, water: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    sites = []
    for i, d in enumerate(boundaries.regions(2)):
        for lon, lat in boundaries.random_points(d, per_district, seed=seed + i, margin_m=800):
            sites.append({"lon": float(lon), "lat": float(lat), "kind": "random"})
    ref, g = national_reference()
    for cls_io, n, kind in ((7, urban, "urban"), (1, water, "water")):
        rows, cols = np.nonzero(ref == cls_io)
        pick = rng.choice(len(rows), size=min(n * 3, len(rows)), replace=False)
        lon, lat = g.pixel_lonlat(rows[pick], cols[pick])
        inside = boundaries.locate(lon, lat, 2)
        k = 0
        for lo, la, reg in zip(lon, lat, inside):
            if reg is not None and k < n:
                sites.append({"lon": float(lo + rng.uniform(-0.002, 0.002)),
                              "lat": float(la + rng.uniform(-0.002, 0.002)), "kind": kind})
                k += 1
    lon = np.array([s["lon"] for s in sites])
    lat = np.array([s["lat"] for s in sites])
    dist = boundaries.locate(lon, lat, 2)
    out = []
    for idx, (s, d) in enumerate(zip(sites, dist)):
        if d is None:
            continue
        s.update(id=f"s{idx:05d}", district=d.name, province=d.province,
                 split="val" if d.name in HOLDOUT_DISTRICTS else "train",
                 year=int(rng.choice(YEARS)), season=str(rng.choice(list(SEASONS))))
        out.append(s)
    return out


def looks_cloudy(rgb: np.ndarray, label: np.ndarray) -> bool:
    """Bright, colourless pixels where the label says it is not barren/snow -> probably cloud."""
    mx, mn = rgb.max(-1).astype(int), rgb.min(-1).astype(int)
    white = (mn > 190) & ((mx - mn) < 25)
    return float((white & (label != 5) & (label != 255)).mean()) > 0.08


def _eox_normalized(year: int, g: Grid):
    scene = imagery.eox.annual_source(year).fetch(g, workers=4)
    scene.rgb = imagery.eox.normalize(scene.rgb, year)
    return scene


def build_site(site: dict, out_dir: Path) -> str:
    path = out_dir / "sites" / f"{site['id']}.npz"
    if path.exists():
        return "cached"
    g = Grid.around(site["lon"], site["lat"], PATCH * MODEL_GSD_M, MODEL_GSD_M)
    y = site["year"]
    (m0, d0), (m1, d1) = SEASONS[site["season"]]
    start, end = date(y, m0, d0), date(y, m1, d1)
    labels = {}

    def label(year: int) -> np.ndarray:
        if year not in labels:
            labels[year] = reference.fetch_io(g, year)[1]
        return labels[year]

    jobs = [
        ("esri", 2024, lambda: imagery.esri.current_source().fetch(g, for_model=True, workers=4)),
        ("eox", y, lambda: _eox_normalized(y, g)),
        ("s2", y, lambda: planetary.mosaic_source("s2", NEPAL_BBOX, start, end, 20).fetch(g, workers=4)),
        ("landsat", y, lambda: planetary.mosaic_source("landsat", NEPAL_BBOX, start, end, 30).fetch(g, workers=4)),
    ]
    imgs, labs, srcs, yrs = [], [], [], []
    for name, lyear, fn in jobs:
        if name == "eox" and y not in imagery.eox.YEARS:  # the 2017 mosaic is blank over Nepal
            continue
        if name == "landsat" and y < 2018 and site["season"] == "dry":
            continue  # Landsat 8 coverage is fine, but keep the set balanced towards recent years
        try:
            sc = fn()
            lab = label(lyear)
        except Exception as exc:  # one source failing must not kill the site
            print(f"  {site['id']} {name}: {exc}", flush=True)
            continue
        if sc.valid_fraction < 0.98 or (lab != 255).mean() < 0.9 or looks_cloudy(sc.rgb, lab):
            continue
        imgs.append(sc.rgb)
        labs.append(lab)
        srcs.append(name)
        yrs.append(lyear)
    if not imgs:
        np.savez_compressed(path, images=np.zeros((0, PATCH, PATCH, 3), np.uint8),
                            labels=np.zeros((0, PATCH, PATCH), np.uint8), sources=np.array([]), years=np.array([]))
        return "empty"
    np.savez_compressed(path, images=np.stack(imgs), labels=np.stack(labs), sources=np.array(srcs),
                        years=np.array(yrs))
    return f"{len(imgs)} patches"


def pack(out_dir: Path) -> None:
    meta = {s["id"]: s for s in csv.DictReader(open(out_dir / "sites.csv"))}
    buckets: dict[str, dict[str, list]] = {}
    for p in sorted((out_dir / "sites").glob("*.npz")):
        z = np.load(p)
        if len(z["images"]) == 0:
            continue
        s = meta[p.stem]
        b = buckets.setdefault(s["split"], {"images": [], "labels": [], "rows": []})
        for i in range(len(z["images"])):
            b["images"].append(z["images"][i])
            b["labels"].append(z["labels"][i])
            b["rows"].append({**s, "source": str(z["sources"][i]), "label_year": int(z["years"][i])})
    for split, b in buckets.items():
        np.savez_compressed(out_dir / f"{split}.npz", images=np.stack(b["images"]), labels=np.stack(b["labels"]))
        with open(out_dir / f"{split}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(b["rows"][0].keys()))
            w.writeheader()
            w.writerows(b["rows"])
        lab = np.stack(b["labels"])
        share = np.bincount(lab[lab != 255].ravel(), minlength=7) / max(1, (lab != 255).sum())
        print(f"{split}: {len(b['images'])} patches; class share {np.round(share * 100, 1)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-district", type=int, default=24)
    ap.add_argument("--urban", type=int, default=260)
    ap.add_argument("--water", type=int, default=160)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pack", action="store_true", help="only pack per-site files into train/val shards")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N sites (for testing)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "sites").mkdir(parents=True, exist_ok=True)
    if args.pack:
        pack(out)
        return
    sites_csv = out / "sites.csv"
    if sites_csv.exists():
        sites = list(csv.DictReader(open(sites_csv)))
        for s in sites:
            s["lon"], s["lat"], s["year"] = float(s["lon"]), float(s["lat"]), int(s["year"])
    else:
        sites = sample_sites(args.per_district, args.urban, args.water, args.seed)
        with open(sites_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sites[0].keys()))
            w.writeheader()
            w.writerows(sites)
    if args.limit:
        sites = sites[:: max(1, len(sites) // args.limit)][: args.limit]
    print(f"{len(sites)} sites ({sum(s['split'] == 'val' for s in sites)} validation)", flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_site, s, out): s["id"] for s in sites}
        for fut in as_completed(futs):
            done += 1
            try:
                status = fut.result()
            except Exception as exc:
                status = f"failed: {exc}"
            if done % 25 == 0 or status.startswith("failed"):
                rate = done / (time.time() - t0)
                print(f"[{done}/{len(sites)}] {futs[fut]}: {status} · {rate:.2f} sites/s", flush=True)
    (out / "build_info.json").write_text(json.dumps({"finished": datetime.now().isoformat(), "sites": len(sites)}))
    pack(out)


if __name__ == "__main__":
    main()
