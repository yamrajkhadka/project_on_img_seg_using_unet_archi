"""Simplify the official Nepal administrative boundaries for the app.

Source: OCHA Common Operational Dataset (HDX "cod-ab-npl"), Survey Department of Nepal, CC BY-IGO.
https://data.humdata.org/dataset/cod-ab-npl  ->  npl_admin_boundaries.geojson.zip

Usage:
    python scripts/prepare_boundaries.py path/to/unzipped_dir
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.validation import make_valid

OUT = Path(__file__).resolve().parent.parent / "data" / "boundaries"
TOLERANCE = {0: 0.0008, 1: 0.0008, 2: 0.0008, 3: 0.0012}  # degrees (~90-130 m)
KEEP = {
    0: ["adm0_pcode", "adm0_name"],
    1: ["adm1_pcode", "adm1_name"],
    2: ["adm2_pcode", "adm2_name", "adm1_pcode", "adm1_name"],
    3: ["adm3_pcode", "adm3_name", "adm2_pcode", "adm2_name", "adm1_pcode", "adm1_name"],
}


def _round(obj, nd=5):
    if isinstance(obj, (list, tuple)):
        return [_round(o, nd) for o in obj]
    return round(obj, nd) if isinstance(obj, float) else obj


def main(src_dir: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for level in range(4):
        src = json.loads((Path(src_dir) / f"npl_admin{level}.geojson").read_text())
        feats = []
        for f in src["features"]:
            p = f["properties"]
            geom = make_valid(shape(f["geometry"])).simplify(TOLERANCE[level], preserve_topology=True)
            props = {k.replace("adm", "a"): p[k] for k in KEEP[level]}
            props["area_km2"] = round(float(p["area_sqkm"]), 2)
            props["lat"], props["lon"] = round(float(p["center_lat"]), 5), round(float(p["center_lon"]), 5)
            g = mapping(geom)
            feats.append({"type": "Feature", "properties": props,
                          "geometry": {"type": g["type"], "coordinates": _round(g["coordinates"])}})
        out = {"type": "FeatureCollection",
               "source": "OCHA COD-AB Nepal (Survey Department of Nepal), CC BY-IGO, valid 2024-03-14",
               "features": feats}
        path = OUT / f"nepal_adm{level}.geojson"
        path.write_text(json.dumps(out, separators=(",", ":")))
        print(f"ADM{level}: {len(feats)} features -> {path.name} ({path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main(sys.argv[1])
