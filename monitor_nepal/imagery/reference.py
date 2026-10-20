"""Independent reference map: Impact Observatory / Esri 10 m annual land cover (2017-2025).

Used to *validate* the U-Net (agreement, confusion matrix) - never to replace it.  Its nine classes
map cleanly onto the seven DeepGlobe classes the paper's network predicts.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from ..geo import Grid
from ..http import FetchError, get

SERVICE = "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer"
YEARS = tuple(range(2017, 2026))
ATTRIBUTION = "Land cover © Esri / Impact Observatory / Microsoft (CC BY 4.0)"

IO_CLASSES = {1: "Water", 2: "Trees", 4: "Flooded vegetation", 5: "Crops", 7: "Built area", 8: "Bare ground",
              9: "Snow/ice", 10: "Clouds", 11: "Rangeland"}
# IO class -> DeepGlobe class id (255 = no data)
IO_TO_DEEPGLOBE = {1: 4, 2: 3, 4: 4, 5: 1, 7: 0, 8: 5, 9: 5, 10: 6, 11: 2}
NODATA = 255


def _lut() -> np.ndarray:
    lut = np.full(256, NODATA, np.uint8)
    for k, v in IO_TO_DEEPGLOBE.items():
        lut[k] = v
    return lut


LUT = _lut()


# ESA WorldCover (2020 v100, 2021 v200) -> DeepGlobe ids.  A second, independently produced reference.
WC_TO_DEEPGLOBE = {10: 3, 20: 2, 30: 2, 40: 1, 50: 0, 60: 5, 70: 5, 80: 4, 90: 4, 95: 3, 100: 2}
WC_ATTRIBUTION = "ESA WorldCover © ESA WorldCover project / contains modified Copernicus Sentinel data (CC BY 4.0)"


def _wc_lut() -> np.ndarray:
    lut = np.full(256, NODATA, np.uint8)
    for k, v in WC_TO_DEEPGLOBE.items():
        lut[k] = v
    return lut


WC_LUT = _wc_lut()


def fetch_worldcover(grid: Grid, year: int = 2021) -> np.ndarray:
    """ESA WorldCover mapped to DeepGlobe ids on ``grid`` (255 = no data), via Planetary Computer."""
    import math

    from .planetary import DATA

    version = "v200" if year == 2021 else "v100"
    lon, lat = grid.center_lonlat
    item = f"ESA_WorldCover_10m_{year}_{version}_N{int(math.floor(lat / 3) * 3):02d}E{int(math.floor(lon / 3) * 3):03d}"
    url = (f"{DATA}/item/bbox/{grid.x0},{grid.y0},{grid.x1},{grid.y1}/{grid.width}x{grid.height}.npy"
           f"?collection=esa-worldcover&item={item}&assets=map&coord_crs=epsg:3857&dst_crs=epsg:3857"
           f"&resampling=nearest")
    r = get(url, timeout=90)
    arr = np.load(io.BytesIO(r.content), allow_pickle=False)
    data, mask = (arr[0], arr[-1]) if arr.ndim == 3 else (arr, np.ones_like(arr))
    out = WC_LUT[np.clip(data, 0, 255).astype(np.uint8)]
    out[mask == 0] = NODATA
    return out


def fetch_io(grid: Grid, year: int, *, max_side: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """IO classes on ``grid`` (uint8) and the same mapped to DeepGlobe ids (255 = no data)."""
    if year not in YEARS:
        raise FetchError(f"The 10 m reference map covers {YEARS[0]}-{YEARS[-1]}.")
    if max(grid.width, grid.height) > max_side:
        raise FetchError("Reference request too large.")
    t0 = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t1 = int(datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)
    r = get(f"{SERVICE}/exportImage", params={
        "bbox": f"{grid.x0},{grid.y0},{grid.x1},{grid.y1}", "bboxSR": 3857, "imageSR": 3857,
        "size": f"{grid.width},{grid.height}", "format": "tiff", "pixelType": "U8", "time": f"{t0},{t1}",
        "renderingRule": '{"rasterFunction":"None"}', "interpolation": "RSP_NearestNeighbor", "f": "image"},
        timeout=90)
    if "image" not in r.headers.get("content-type", ""):
        raise FetchError("The reference land-cover service returned an error.")
    io_cls = np.asarray(Image.open(io.BytesIO(r.content)))
    if io_cls.shape != (grid.height, grid.width):
        io_cls = np.asarray(Image.fromarray(io_cls).resize((grid.width, grid.height), Image.Resampling.NEAREST))
    return io_cls, LUT[io_cls]
