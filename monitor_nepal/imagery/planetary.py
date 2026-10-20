"""Microsoft Planetary Computer: Sentinel-2 (2015-) and Landsat (1972-) for any date range.

We use three anonymous endpoints:

* STAC search - which scenes exist, when, and how cloudy;
* mosaic registration - turns "least-cloudy scenes in this window" into an XYZ tile source;
* tile ``.npy`` output - raw spectral-index values (NDVI, NDWI, NDSI, NDBI) for analysis.
"""
from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from urllib.parse import quote

import numpy as np

from ..geo import ORIGIN, Grid, zoom_resolution
from ..http import FetchError, get_tile_bytes, post_json
from .base import XYZSource, resample_array

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
DATA = "https://planetarycomputer.microsoft.com/api/data/v1"
ATTRIBUTION_S2 = "Contains modified Copernicus Sentinel-2 data · Microsoft Planetary Computer"
ATTRIBUTION_LANDSAT = "Landsat imagery courtesy of NASA/USGS · Microsoft Planetary Computer"


@dataclass(frozen=True)
class Product:
    key: str
    collection: str
    name: str
    render: str  # query string for natural-looking RGB
    native_gsd: float
    start: date
    end: date | None
    attribution: str
    max_zoom: int
    true_color: bool = True
    platforms: tuple[str, ...] = ()


PRODUCTS: dict[str, Product] = {
    "s2": Product("s2", "sentinel-2-l2a", "Sentinel-2 L2A",
                  "assets=visual&asset_bidx=visual%7C1%2C2%2C3&nodata=0", 10, date(2015, 7, 1), None,
                  ATTRIBUTION_S2, 15),
    "landsat": Product("landsat", "landsat-c2-l2", "Landsat 4-9 (TM/ETM+/OLI)",
                       "assets=red&assets=green&assets=blue&nodata=0&color_formula="
                       + quote("gamma RGB 2.7, saturation 1.5, sigmoidal RGB 15 0.55"), 30, date(1982, 8, 22), None,
                       ATTRIBUTION_LANDSAT, 13),
    "mss": Product("mss", "landsat-c2-l1", "Landsat 1-5 MSS (false colour)",
                   "assets=nir08&assets=red&assets=green&nodata=0&color_formula="
                   + quote("gamma RGB 2.6, saturation 1.0, sigmoidal RGB 14 0.5"), 60, date(1972, 7, 25),
                   date(2013, 1, 7), ATTRIBUTION_LANDSAT, 12, true_color=False),
}

# Normalised-difference indices: (name, sentinel-2 expression, landsat expression, meaning)
INDICES = {
    "ndvi": ("NDVI", "(B08-B04)/(B08+B04)", "(nir08-red)/(nir08+red)", "Vegetation greenness"),
    "ndwi": ("NDWI", "(B03-B08)/(B03+B08)", "(green-nir08)/(green+nir08)", "Open water (McFeeters)"),
    "ndsi": ("NDSI", "(B03-B11)/(B03+B11)", "(green-swir16)/(green+swir16)", "Snow and ice"),
    "ndbi": ("NDBI", "(B11-B08)/(B11+B08)", "(swir16-nir08)/(swir16+nir08)", "Built-up / bare surfaces"),
}


@dataclass(frozen=True)
class SceneInfo:
    id: str
    datetime: str
    cloud: float | None
    platform: str | None

    @property
    def day(self) -> str:
        return self.datetime[:10]


def _cloud_filter(cloud_max: float | None) -> dict | None:
    if cloud_max is None:
        return None
    return {"op": "<=", "args": [{"property": "eo:cloud_cover"}, cloud_max]}


def search(product: str, bbox: tuple[float, float, float, float], start: date, end: date,
           cloud_max: float | None = None, limit: int = 200) -> list[SceneInfo]:
    """Scenes intersecting ``bbox`` (lon/lat) between ``start`` and ``end``, oldest first."""
    p = PRODUCTS[product]
    body: dict = {"collections": [p.collection], "bbox": list(bbox),
                  "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
                  "limit": min(limit, 1000), "sortby": [{"field": "properties.datetime", "direction": "asc"}]}
    flt = _cloud_filter(cloud_max)
    if flt:
        body["filter-lang"] = "cql2-json"
        body["filter"] = flt
    data = post_json(f"{STAC}/search", body)
    out = []
    for f in data.get("features", []):
        pr = f.get("properties", {})
        out.append(SceneInfo(f["id"], pr.get("datetime", ""), pr.get("eo:cloud_cover"), pr.get("platform")))
    return out


@lru_cache(maxsize=512)
def _register(collection: str, bbox: tuple[float, float, float, float], start: str, end: str,
              cloud_max: float | None) -> str:
    body: dict = {"collections": [collection], "bbox": list(bbox),
                  "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
                  "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}]}
    flt = _cloud_filter(cloud_max)
    if flt:
        body["filter-lang"] = "cql2-json"
        body["filter"] = flt
    data = post_json(f"{DATA}/mosaic/register", body)
    sid = data.get("searchid") or data.get("id")
    if not sid:
        raise FetchError("Planetary Computer did not return a mosaic id.")
    return sid


def _round_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    # Register slightly padded, rounded boxes so neighbouring requests share one cached mosaic.
    return (math.floor(bbox[0] * 20) / 20, math.floor(bbox[1] * 20) / 20,
            math.ceil(bbox[2] * 20) / 20, math.ceil(bbox[3] * 20) / 20)


def mosaic_source(product: str, bbox: tuple[float, float, float, float], start: date, end: date,
                  cloud_max: float | None = 30) -> XYZSource:
    """Least-cloudy-first mosaic of every scene in the window, as an XYZ tile source."""
    p = PRODUCTS[product]
    sid = _register(p.collection, _round_bbox(bbox), start.isoformat(), end.isoformat(), cloud_max)
    url = (f"{DATA}/mosaic/{sid}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png?collection={p.collection}&{p.render}")
    span = f"{start.isoformat()}/{end.isoformat()}"
    return XYZSource(key=f"pc-{product}", name=p.name, url_template=url, attribution=p.attribution,
                     max_zoom=p.max_zoom, min_zoom=8, native_gsd=p.native_gsd, acquired=span,
                     label=f"{p.name} · least-cloudy mosaic {start:%d %b %Y} – {end:%d %b %Y}",
                     meta={"search_id": sid, "cloud_max": cloud_max, "product": product})


def item_source(product: str, item_id: str, when: str) -> XYZSource:
    """A single scene as an XYZ source (exact acquisition date)."""
    p = PRODUCTS[product]
    url = f"{DATA}/item/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png?collection={p.collection}&item={item_id}&{p.render}"
    return XYZSource(key=f"pc-{product}-item", name=p.name, url_template=url, attribution=p.attribution,
                     max_zoom=p.max_zoom, min_zoom=8, native_gsd=p.native_gsd, acquired=when[:10],
                     label=f"{p.name} · scene {when[:10]}", meta={"item": item_id, "product": product})


def fetch_index(product: str, index: str, grid: Grid, start: date, end: date, cloud_max: float | None = 30,
                workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Raw normalised-difference index on ``grid`` (float32, NaN where no data) and a validity mask."""
    p = PRODUCTS[product]
    if product == "mss":
        raise FetchError("Spectral indices need Sentinel-2 or Landsat 4-9 bands.")
    _, s2_expr, ls_expr, _ = INDICES[index]
    expr = s2_expr if product == "s2" else ls_expr
    sid = _register(p.collection, _round_bbox(grid.bounds_lonlat), start.isoformat(), end.isoformat(), cloud_max)
    # Indices are smooth; sample them at the sensor's native resolution and interpolate.
    z = max(8, min(p.max_zoom - 1, int(math.floor(math.log2(2 * ORIGIN / (256 * max(grid.res, p.native_gsd)))))))
    size = 2 * ORIGIN / 2 ** z
    tx0, tx1 = int((grid.x0 + ORIGIN) // size), int((grid.x1 - 1e-6 + ORIGIN) // size)
    ty0, ty1 = int((ORIGIN - grid.y1) // size), int((ORIGIN - grid.y0 - 1e-6) // size)
    tiles = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    if len(tiles) > 400:
        raise FetchError("Area too large for an index request; pick a smaller region.")
    canvas = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.nan, np.float32)

    def work(t):
        url = (f"{DATA}/mosaic/{sid}/tiles/WebMercatorQuad/{z}/{t[0]}/{t[1]}.npy?collection={p.collection}"
               f"&expression={quote(expr)}&asset_as_band=true&nodata=0")
        return t, get_tile_bytes(url)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (x, y), raw in ex.map(work, tiles):
            if not raw:
                continue
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            data, mask = (arr[0], arr[-1]) if arr.ndim == 3 else (arr, np.ones_like(arr))
            vals = np.where(mask > 0, data, np.nan).astype(np.float32)
            vals[(vals < -1) | (vals > 1)] = np.nan
            r, c = (y - ty0) * 256, (x - tx0) * 256
            canvas[r:r + 256, c:c + 256] = vals
    cgrid = Grid(-ORIGIN + tx0 * size, ORIGIN - ty0 * size, zoom_resolution(z), canvas.shape[1], canvas.shape[0])
    out = resample_array(canvas, cgrid, grid)
    return out, np.isfinite(out)
