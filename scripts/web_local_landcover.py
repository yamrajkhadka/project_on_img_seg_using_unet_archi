"""Land cover of every local unit (municipality / rural municipality) for the website, from the 10 m reference map.

For each of Nepal's 775 local units the Impact Observatory 10 m annual land-cover map (translated to the paper's seven
classes) is fetched over the unit's bounding box, masked to its exact boundary and summarised as class shares, for
2017 and for the latest year. Units are too small for the atlas's random sample, so this is the honest per-unit
figure; the U-Net's own estimates are shown for districts and provinces.

Output: web/data/local_landcover.json (resumable: units already done are skipped).

Usage:
    python scripts/web_local_landcover.py [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitor_nepal import boundaries  # noqa: E402
from monitor_nepal.config import CLASS_KEYS  # noqa: E402
from monitor_nepal.geo import Grid, lonlat_to_merc  # noqa: E402
from monitor_nepal.http import FetchError  # noqa: E402
from monitor_nepal.imagery import reference  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "web" / "data" / "local_landcover.json"
LAND = 6  # DeepGlobe ids 0-5 are land classes; 6 = unknown (clouds), 255 = no data


def mask(grid: Grid, geom) -> np.ndarray:
    """Pixels of ``grid`` inside the (multi)polygon ``geom`` (lon/lat)."""
    img = Image.new("L", (grid.width, grid.height), 0)
    draw = ImageDraw.Draw(img)

    def px(ring):
        out = []
        for lon, lat in ring.coords:
            x, y = lonlat_to_merc(lon, lat)
            out.append(((x - grid.x0) / grid.res, (grid.y1 - y) / grid.res))
        return out

    for poly in getattr(geom, "geoms", [geom]):
        draw.polygon(px(poly.exterior), fill=1)
        for hole in poly.interiors:
            draw.polygon(px(hole), fill=0)
    return np.asarray(img, bool)


def shares(region: boundaries.Region, year: int) -> dict:
    geom = boundaries.geometry(region)
    lon0, lat0, lon1, lat1 = geom.bounds
    side_m = max((lon1 - lon0) * 111_320 * math.cos(math.radians((lat0 + lat1) / 2)), (lat1 - lat0) * 110_574)
    res = max(10.0, side_m / 3500)
    grid = Grid.from_lonlat_bbox((lon0, lat0, lon1, lat1), res)
    _, dg = reference.fetch_io(grid, year)
    inside = mask(grid, geom)
    land = inside & (dg < LAND)
    n_inside = int(inside.sum())
    if n_inside == 0 or land.sum() < 0.5 * n_inside:
        raise FetchError("too little mapped land")
    counts = np.bincount(dg[land], minlength=LAND)[:LAND]
    return {"year": year, "res_m": round(res, 1),
            "share": {k: round(float(c) / float(land.sum()) * 100, 2) for k, c in zip(CLASS_KEYS[:LAND], counts)},
            "unmapped_pct": round(100 * (1 - float(land.sum()) / n_inside), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    years = (reference.years()[0], reference.years()[-1])
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [r for r in boundaries.regions(3)
            if r.code not in done or set(done[r.code].get("years", {})) != {str(y) for y in years}]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} done, {len(todo)} to do, years {years}", flush=True)

    def one(region):
        out = {}
        for y in years:
            try:
                out[str(y)] = shares(region, y)
            except (FetchError, ValueError) as exc:
                out[str(y)] = {"error": str(exc)[:120]}
        return region.code, out

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (code, out) in enumerate(ex.map(one, todo), 1):
            done[code] = {"years": out}
            if i % 25 == 0 or i == len(todo):
                OUT.write_text(json.dumps(done, separators=(",", ":")))
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
