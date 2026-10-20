"""Common imagery types and the XYZ-tile mosaicking engine."""
from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

from ..geo import ORIGIN, Grid, zoom_resolution
from ..http import EMPTY, FetchError, get_tile_bytes
from ..resample import cv2_linear

ProgressFn = Callable[[float, str], None]
MAX_TILES_PER_REQUEST = 600


@dataclass
class Scene:
    """An RGB image resampled onto a :class:`Grid`, plus provenance."""

    rgb: np.ndarray  # (H, W, 3) uint8
    grid: Grid
    valid: np.ndarray  # (H, W) bool - False where the provider had no data
    source_key: str
    source_name: str
    label: str
    attribution: str
    acquired: str | None = None  # ISO date (or "start/end") of the pixels, when known
    meta: dict = field(default_factory=dict)

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


@dataclass
class XYZSource:
    """Any web-mercator tile service addressed by ``{z}/{x}/{y}``."""

    key: str
    name: str
    url_template: str  # placeholders {z} {x} {y}
    attribution: str
    max_zoom: int
    min_zoom: int = 0
    tile_size: int = 256
    native_gsd: float | None = None  # ground metres per pixel of the underlying sensor
    acquired: str | None = None
    label: str | None = None
    meta: dict = field(default_factory=dict)

    def tile_url(self, z: int, x: int, y: int) -> str:
        return self.url_template.format(z=z, x=x, y=y)

    def choose_zoom(self, grid: Grid, detail_res: float | None = None) -> int:
        """Coarsest zoom whose pixels are at least as fine as ``detail_res`` (projected metres).

        ``detail_res`` defaults to the grid's own resolution (a display-quality fetch).
        """
        target = detail_res if detail_res is not None else grid.res
        factor = self.tile_size / 256  # a 512 px tile carries 2x the pixels of a 256 px tile
        for z in range(self.min_zoom, self.max_zoom + 1):
            if zoom_resolution(z) / factor <= target * 1.05:
                return z
        return self.max_zoom

    def fetch(self, grid: Grid, *, progress: ProgressFn | None = None, workers: int = 8,
              for_model: bool = False) -> Scene:
        """Imagery on ``grid``.

        ``for_model=True`` reproduces the training pipeline: sub-metre-to-metre tiles are sampled
        down with OpenCV-style bilinear interpolation (no antialiasing), which keeps the canopy and
        roof texture the network learned from.  Otherwise an antialiased, display-quality resample.
        """
        detail = None
        if for_model:
            # ~1 m ground detail is as good as 0.5 m for this network (DeepGlobe test), 2 m is not.
            detail = 1.0 / grid.row_scale().mean()
        rgb, valid, z = mosaic_xyz(self, grid, progress=progress, workers=workers, detail_res=detail,
                                   point_sample=for_model)
        return Scene(rgb=rgb, grid=grid, valid=valid, source_key=self.key, source_name=self.name,
                     label=self.label or self.name, attribution=self.attribution, acquired=self.acquired,
                     meta={**self.meta, "zoom": z, "for_model": for_model})


def _decode(data: bytes, tile_size: int) -> tuple[np.ndarray, np.ndarray]:
    if data == EMPTY:
        return np.zeros((tile_size, tile_size, 3), np.uint8), np.zeros((tile_size, tile_size), bool)
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.size != (tile_size, tile_size):
        im = im.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        rgba = np.asarray(im.convert("RGBA"))
        return np.ascontiguousarray(rgba[..., :3]), rgba[..., 3] > 0
    arr = np.asarray(im.convert("RGB"))
    return arr, np.ones(arr.shape[:2], bool)


def mosaic_xyz(src: XYZSource, grid: Grid, *, progress: ProgressFn | None = None, workers: int = 8,
               detail_res: float | None = None, point_sample: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
    """Fetch the tiles covering ``grid`` and resample them onto it."""
    z = src.choose_zoom(grid, detail_res)
    n = 2 ** z
    size = 2 * ORIGIN / n
    tx0 = max(0, int(math.floor((grid.x0 + ORIGIN) / size)))
    tx1 = min(n - 1, int(math.floor((grid.x1 - 1e-6 + ORIGIN) / size)))
    ty0 = max(0, int(math.floor((ORIGIN - grid.y1) / size)))
    ty1 = min(n - 1, int(math.floor((ORIGIN - grid.y0 - 1e-6) / size)))
    tiles = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    if len(tiles) > MAX_TILES_PER_REQUEST:
        raise FetchError(f"That area needs {len(tiles)} tiles from {src.name}; please choose a smaller area.")

    ts = src.tile_size
    canvas = np.zeros(((ty1 - ty0 + 1) * ts, (tx1 - tx0 + 1) * ts, 3), np.uint8)
    cmask = np.zeros(canvas.shape[:2], bool)
    done = 0

    def work(t: tuple[int, int]) -> tuple[tuple[int, int], bytes]:
        return t, get_tile_bytes(src.tile_url(z, t[0], t[1]))

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tiles)))) as ex:
        for fut in [ex.submit(work, t) for t in tiles]:
            try:
                (x, y), data = fut.result()
                arr, ok = _decode(data, ts)
            except FetchError as exc:
                errors.append(str(exc))
                continue
            except (OSError, ValueError) as exc:  # e.g. an HTML error page served with status 200
                errors.append(f"{src.name} returned an unreadable tile ({exc.__class__.__name__}).")
                continue
            r, c = (y - ty0) * ts, (x - tx0) * ts
            canvas[r:r + ts, c:c + ts] = arr
            cmask[r:r + ts, c:c + ts] = ok
            done += 1
            if progress:
                progress(done / len(tiles), f"{src.name}: {done}/{len(tiles)} tiles")
    if errors and done == 0:
        raise FetchError(errors[0])

    # Resample the canvas onto the grid using a floating-point crop box (exact georeferencing).
    cres = size / ts
    cx0, cy1 = -ORIGIN + tx0 * size, ORIGIN - ty0 * size
    box = ((grid.x0 - cx0) / cres, (cy1 - grid.y1) / cres, (grid.x1 - cx0) / cres, (cy1 - grid.y0) / cres)
    out_size = (grid.width, grid.height)
    if point_sample:
        rgb = cv2_linear(canvas, grid.width, grid.height, box=box)
    else:
        rgb = np.asarray(Image.fromarray(canvas).resize(out_size, Image.Resampling.LANCZOS, box=box))
    valid = np.asarray(Image.fromarray(cmask.astype(np.uint8) * 255).resize(out_size, Image.Resampling.BOX,
                                                                             box=box)) > 250
    return rgb, valid, z


def resample_array(arr: np.ndarray, src: Grid, dst: Grid, *, nearest: bool = False) -> np.ndarray:
    """Resample a 2-D/3-D array from one grid to another (same CRS)."""
    box = ((dst.x0 - src.x0) / src.res, (src.y1 - dst.y1) / src.res,
           (dst.x1 - src.x0) / src.res, (src.y1 - dst.y0) / src.res)
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if arr.ndim == 2:
        im = Image.fromarray(arr)
        return np.asarray(im.resize((dst.width, dst.height), method, box=box))
    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(arr[..., i])).resize((dst.width, dst.height), method,
                                                                                    box=box))
             for i in range(arr.shape[-1])]
    return np.stack(chans, axis=-1)
