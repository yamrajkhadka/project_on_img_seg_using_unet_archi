"""Nepal's administrative units: country, 7 provinces, 77 districts and 775 local units.

Boundaries: OCHA COD-AB (Survey Department of Nepal), CC BY-IGO, simplified to ~100 m.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from shapely import STRtree, points
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from .config import BOUNDARIES_DIR

LEVEL_NAMES = {0: "Country", 1: "Province", 2: "District", 3: "Local unit"}


@dataclass(frozen=True)
class Region:
    level: int
    code: str
    name: str
    area_km2: float
    lon: float
    lat: float
    parent_code: str | None = None
    parent_name: str | None = None
    province: str | None = None

    @property
    def label(self) -> str:
        if self.level == 0:
            return "Nepal (whole country)"
        if self.level == 1:
            return f"{self.name} Province"
        if self.level == 2:
            return f"{self.name} District · {self.province}"
        return f"{self.name} · {self.parent_name} District"

    @property
    def short(self) -> str:
        return {0: "Nepal", 1: f"{self.name} Province", 2: f"{self.name} District"}.get(self.level, self.name)


@lru_cache(maxsize=4)
def _load(level: int) -> tuple[list[Region], list[BaseGeometry]]:
    data = json.loads((BOUNDARIES_DIR / f"nepal_adm{level}.geojson").read_text())
    regions, geoms = [], []
    for f in data["features"]:
        p = f["properties"]
        code, name = p[f"a{level}_pcode"], p[f"a{level}_name"]
        parent_code = p.get(f"a{level - 1}_pcode") if level >= 2 else ("NP" if level == 1 else None)
        parent_name = p.get(f"a{level - 1}_name") if level >= 2 else ("Nepal" if level == 1 else None)
        regions.append(Region(level, code, name, p["area_km2"], p["lon"], p["lat"], parent_code, parent_name,
                              p.get("a1_name") if level >= 2 else (name if level == 1 else None)))
        geoms.append(shape(f["geometry"]))
    return regions, geoms


def regions(level: int) -> list[Region]:
    return sorted(_load(level)[0], key=lambda r: r.name)


def children(parent: Region) -> list[Region]:
    if parent.level >= 3:
        return []
    kids = _load(parent.level + 1)[0]
    if parent.level == 0:
        return sorted(kids, key=lambda r: r.name)
    return sorted([r for r in kids if r.parent_code == parent.code], key=lambda r: r.name)


def country() -> Region:
    return _load(0)[0][0]


@lru_cache(maxsize=2048)
def get(code: str) -> Region:
    level = {2: 0, 4: 1, 6: 2}.get(len(code), 3)
    for r in _load(level)[0]:
        if r.code == code:
            return r
    raise KeyError(code)


def geometry(region: Region) -> BaseGeometry:
    regs, geoms = _load(region.level)
    return geoms[[r.code for r in regs].index(region.code)]


def geojson(level: int) -> dict:
    return json.loads((BOUNDARIES_DIR / f"nepal_adm{level}.geojson").read_text())


def region_feature(region: Region) -> dict:
    data = geojson(region.level)
    for f in data["features"]:
        if f["properties"][f"a{region.level}_pcode"] == region.code:
            return f
    raise KeyError(region.code)


@lru_cache(maxsize=8)
def _tree(level: int) -> tuple[STRtree, list[Region]]:
    regs, geoms = _load(level)
    return STRtree(geoms), regs


def locate(lon: np.ndarray, lat: np.ndarray, level: int = 2) -> list[Region | None]:
    """Admin unit containing each point (None outside Nepal)."""
    tree, regs = _tree(level)
    pts = points(np.asarray(lon, float), np.asarray(lat, float))
    idx_pts, idx_geoms = tree.query(pts, predicate="within")
    out: list[Region | None] = [None] * len(pts)
    for i, g in zip(idx_pts, idx_geoms):
        out[int(i)] = regs[int(g)]
    return out


def contains(region: Region, lon: float, lat: float) -> bool:
    return geometry(region).contains(Point(lon, lat))


def random_points(region: Region, n: int, seed: int = 0, margin_m: float = 0.0) -> np.ndarray:
    """``n`` uniformly random (lon, lat) points inside the region (reproducible for a given seed).

    Uniform in lon/lat is very close to uniform in area at Nepal's size; ``margin_m`` keeps points that
    distance inside the boundary so a sampling window does not hang over the border.
    """
    geom = geometry(region)
    if margin_m > 0:
        shrunk = geom.buffer(-margin_m / 111_000)
        geom = shrunk if not shrunk.is_empty else geom
    pg = prep(geom)
    rng = np.random.default_rng(seed)
    lon0, lat0, lon1, lat1 = geom.bounds
    out = []
    while len(out) < n:
        k = max(64, 4 * (n - len(out)))
        cand = np.column_stack([rng.uniform(lon0, lon1, k), rng.uniform(lat0, lat1, k)])
        for lon, lat in cand:
            if pg.contains(Point(lon, lat)):
                out.append((lon, lat))
                if len(out) == n:
                    break
    return np.array(out)


def search(query: str, limit: int = 20) -> list[Region]:
    q = query.strip().lower()
    if not q:
        return []
    hits = [r for lvl in (1, 2, 3) for r in _load(lvl)[0] if q in r.name.lower()]
    hits.sort(key=lambda r: (not r.name.lower().startswith(q), r.level, r.name))
    return hits[:limit]
