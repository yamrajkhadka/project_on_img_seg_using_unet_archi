"""Exports that open directly in GIS software (no GDAL needed)."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
from PIL import Image, TiffImagePlugin

from .config import CLASSES, PALETTES
from .geo import WEB_MERCATOR_WKT, Grid

_DOUBLE, _SHORT, _ASCII = 12, 3, 2


def _geo_ifd(grid: Grid, nodata: int | None = None) -> TiffImagePlugin.ImageFileDirectory_v2:
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (float(grid.res), float(grid.res), 0.0)  # ModelPixelScaleTag
    ifd.tagtype[33550] = _DOUBLE
    ifd[33922] = (0.0, 0.0, 0.0, float(grid.x0), float(grid.y1), 0.0)  # ModelTiepointTag
    ifd.tagtype[33922] = _DOUBLE
    ifd[34735] = (1, 1, 0, 4,  # GeoKeyDirectory header: version, revision, minor, key count
                  1024, 0, 1, 1,  # GTModelTypeGeoKey = projected
                  1025, 0, 1, 1,  # GTRasterTypeGeoKey = pixel is area
                  3072, 0, 1, 3857,  # ProjectedCSTypeGeoKey = EPSG:3857
                  3076, 0, 1, 9001)  # ProjLinearUnitsGeoKey = metre
    ifd.tagtype[34735] = _SHORT
    if nodata is not None:
        ifd[42113] = str(nodata)  # GDAL_NODATA
        ifd.tagtype[42113] = _ASCII
    return ifd


def classes_geotiff(classes: np.ndarray, grid: Grid, palette: str = "cartographic") -> bytes:
    """Paletted GeoTIFF of class ids 0-6 (colours embedded, so GIS shows the legend colours)."""
    im = Image.fromarray(classes.astype(np.uint8), mode="P")
    pal = [v for rgb in PALETTES[palette] for v in rgb]
    im.putpalette(pal + [0] * (768 - len(pal)))
    buf = io.BytesIO()
    im.save(buf, format="TIFF", tiffinfo=_geo_ifd(grid), compression="tiff_deflate")
    return buf.getvalue()


def rgb_geotiff(rgb: np.ndarray, grid: Grid) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="TIFF", tiffinfo=_geo_ifd(grid), compression="tiff_deflate")
    return buf.getvalue()


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def bundle(name: str, *, classes: np.ndarray, rgb: np.ndarray, grid: Grid, composition_csv: str, meta: dict,
           palette: str = "cartographic") -> bytes:
    """A zip with GeoTIFFs, PNG + world files, the composition table and provenance metadata."""
    pal = np.array(PALETTES[palette], np.uint8)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}_classes.tif", classes_geotiff(classes, grid, palette))
        z.writestr(f"{name}_imagery.tif", rgb_geotiff(rgb, grid))
        z.writestr(f"{name}_classes.png", png_bytes(pal[classes]))
        z.writestr(f"{name}_classes.pgw", grid.world_file())
        z.writestr(f"{name}_classes.prj", WEB_MERCATOR_WKT)
        z.writestr(f"{name}_composition.csv", composition_csv)
        legend = {c.id: {"name": c.name, "rgb": list(map(int, pal[c.id])), "definition": c.definition} for c in CLASSES}
        z.writestr(f"{name}_metadata.json", json.dumps({**meta, "legend": legend, "crs": "EPSG:3857",
                                                        "pixel_size_m_projected": grid.res,
                                                        "pixel_size_m_ground": grid.ground_res}, indent=2,
                                                       default=str))
    return buf.getvalue()
