"""Evaluate the original and the Nepal-adapted U-Net on districts never used for training.

For random sites in the eleven held-out districts we fetch four kinds of imagery from 2021 and compare each
model's map with two independent references of the same year: Impact Observatory 10 m LULC (the source of the
fine-tuning labels) and ESA WorldCover 2021 (produced independently - the fairer test).  We also report how well
the two references agree with each other, the practical ceiling for any model scored against them.

Usage:
    MONITOR_NEPAL_CACHE=/tmp/tiles python scripts/evaluate_models.py --per-district 20
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_nepal_dataset import HOLDOUT_DISTRICTS  # noqa: E402

from monitor_nepal import boundaries  # noqa: E402
from monitor_nepal.config import CLASS_NAMES, MODEL_GSD_M, MODEL_TILE, MODELS_DIR, NEPAL_BBOX  # noqa: E402
from monitor_nepal.geo import Grid  # noqa: E402
from monitor_nepal.http import FetchError  # noqa: E402
from monitor_nepal.imagery import eox, esri, planetary, reference  # noqa: E402
from monitor_nepal.model import MODELS, get_model  # noqa: E402
from monitor_nepal.postprocess import clean  # noqa: E402

YEAR = 2021
SOURCES = {
    "Esri World Imagery (latest)": lambda g: esri.current_source().fetch(g, for_model=True, workers=4),
    "Sentinel-2 cloudless 2021": lambda g: _normalized(eox.annual_source(YEAR).fetch(g, workers=4), YEAR),
    "Sentinel-2 L2A, Oct-Dec 2021": lambda g: planetary.mosaic_source("s2", NEPAL_BBOX, date(YEAR, 10, 1),
                                                                      date(YEAR, 12, 31), 20).fetch(g, workers=4),
    "Landsat 8, Oct-Dec 2021": lambda g: planetary.mosaic_source("landsat", NEPAL_BBOX, date(YEAR, 10, 1),
                                                                 date(YEAR, 12, 31), 30).fetch(g, workers=4),
}


def _normalized(scene, year: int):
    scene.rgb = eox.normalize(scene.rgb, year)
    return scene


def confusion(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ok = ref < 6
    cm = np.zeros((7, 7), np.int64)
    np.add.at(cm, (ref[ok].astype(int), pred[ok].astype(int)), 1)
    return cm


def summarise(cm: np.ndarray) -> dict:
    n = int(cm.sum())
    if n == 0:
        return {"n": 0}
    oa = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    inter = np.diag(cm)[:6]
    union = cm.sum(0)[:6] + cm.sum(1)[:6] - inter
    iou = inter / np.maximum(union, 1)
    present = cm.sum(1)[:6] > 0
    f1 = 2 * inter / np.maximum(cm.sum(0)[:6] + cm.sum(1)[:6], 1)
    return {"n": n, "overall": round(float(oa), 4), "kappa": round(float((oa - pe) / (1 - pe)), 4),
            "miou": round(float(iou[present].mean()), 4),
            "iou": {CLASS_NAMES[i]: round(float(iou[i]), 4) for i in range(6)},
            "f1": {CLASS_NAMES[i]: round(float(f1[i]), 4) for i in range(6)},
            "ref_share": {CLASS_NAMES[i]: round(float(cm[i].sum() / n * 100), 2) for i in range(6)},
            "pred_share": {CLASS_NAMES[i]: round(float(cm[:, i].sum() / n * 100), 2) for i in range(7)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-district", type=int, default=20)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(MODELS_DIR / "evaluation.json"))
    args = ap.parse_args()
    models = [m.key for m in MODELS.values() if m.available]
    nets = {m: get_model(m) for m in models}
    districts = [d for d in boundaries.regions(2) if d.name in HOLDOUT_DISTRICTS]
    sites = [(d.name, float(lon), float(lat)) for i, d in enumerate(districts)
             for lon, lat in boundaries.random_points(d, args.per_district, seed=args.seed + i, margin_m=700)]
    print(f"{len(sites)} sites in {len(districts)} held-out districts; models {models}", flush=True)

    cms: dict[tuple[str, str, str], np.ndarray] = {}
    inter = np.zeros((7, 7), np.int64)

    def one(site):
        _, lon, lat = site
        g = Grid.around(lon, lat, MODEL_TILE * MODEL_GSD_M, MODEL_GSD_M)
        try:
            refs = {"Impact Observatory 2021": reference.fetch_io(g, YEAR)[1],
                    "ESA WorldCover 2021": reference.fetch_worldcover(g, YEAR)}
        except FetchError as exc:
            return None, str(exc)
        imgs = {}
        for name, fn in SOURCES.items():
            try:
                sc = fn(g)
                if sc.valid_fraction > 0.98:
                    imgs[name] = sc.rgb
            except FetchError:
                pass
        return (refs, imgs), None

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for out, _err in ex.map(one, sites):
            done += 1
            if out is None:
                continue
            refs, imgs = out
            a, b = refs["Impact Observatory 2021"], refs["ESA WorldCover 2021"]
            ok = (a < 6) & (b < 6)
            np.add.at(inter, (b[ok].astype(int), a[ok].astype(int)), 1)
            for src, rgb in imgs.items():
                for m, net in nets.items():
                    pred = clean(net.predict_proba(rgb).argmax(-1).astype(np.uint8), 64)
                    for rname, ref in refs.items():
                        key = (m, src, rname)
                        cms[key] = cms.get(key, np.zeros((7, 7), np.int64)) + confusion(pred, ref)
            if done % 25 == 0:
                print(f"  {done}/{len(sites)}", flush=True)

    results = []
    for (m, src, rname), cm in sorted(cms.items()):
        s = summarise(cm)
        results.append({"model": MODELS[m].short, "model_key": m, "source": src, "reference": rname, **s})
        print(f"{MODELS[m].short:16s} | {src:30s} | {rname:24s} | agreement {s['overall']:.3f} "
              f"kappa {s['kappa']:.3f} mIoU {s['miou']:.3f}", flush=True)
    ir = summarise(inter)
    print(f"Impact Observatory vs WorldCover agreement: {ir['overall']:.3f} (kappa {ir['kappa']:.3f})", flush=True)
    payload = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "description": f"{len(sites)} random 1.22 km windows in {len(districts)} districts held out from "
                              f"training ({', '.join(d.name for d in districts)}); imagery and references from {YEAR}; "
                              "pixels the references mark as cloud/no-data excluded.",
               "results": results, "inter_reference": ir}
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
