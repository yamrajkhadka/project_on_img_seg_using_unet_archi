"""Web Mercator (EPSG:3857) grids, tile maths and ground-area helpers.

Every raster in the project lives on a north-up :class:`Grid` in Web Mercator, because the imagery
services we use are natively tiled in that projection.  Mercator stretches distances by
``1 / cos(latitude)``, so a grid keeps both its *projected* resolution (metres in EPSG:3857) and the
*ground* resolution the model actually sees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS = 6378137.0
ORIGIN = math.pi * EARTH_RADIUS  # 20037508.342789244
TILE_SIZE = 256


def lonlat_to_merc(lon: float, lat: float) -> tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    return math.radians(lon) * EARTH_RADIUS, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * EARTH_RADIUS


def merc_to_lonlat(x: float, y: float) -> tuple[float, float]:
    return math.degrees(x / EARTH_RADIUS), math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS)) - math.pi / 2)


def merc_y_to_lat(y: np.ndarray | float) -> np.ndarray | float:
    return np.degrees(2 * np.arctan(np.exp(np.asarray(y) / EARTH_RADIUS)) - np.pi / 2)


def zoom_resolution(z: int) -> float:
    """Projected metres per pixel of a 256-px XYZ tile at zoom ``z``."""
    return 2 * ORIGIN / (TILE_SIZE * 2 ** z)


def tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    size = 2 * ORIGIN / 2 ** z
    return -ORIGIN + x * size, ORIGIN - (y + 1) * size, -ORIGIN + (x + 1) * size, ORIGIN - y * size


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    x, y = lonlat_to_merc(lon, lat)
    size = 2 * ORIGIN / 2 ** z
    return int((x + ORIGIN) // size), int((ORIGIN - y) // size)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(min(1.0, a)))


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid in EPSG:3857.

    ``x0``/``y1`` are the left/top edges (metres), ``res`` the projected pixel size.
    """

    x0: float
    y1: float
    res: float
    width: int
    height: int

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def from_lonlat_bbox(cls, bbox: tuple[float, float, float, float], ground_res: float) -> "Grid":
        lon0, lat0, lon1, lat1 = bbox
        x0, y0 = lonlat_to_merc(lon0, lat0)
        x1, y1 = lonlat_to_merc(lon1, lat1)
        res = ground_res / math.cos(math.radians((lat0 + lat1) / 2))
        width = max(1, int(round((x1 - x0) / res)))
        height = max(1, int(round((y1 - y0) / res)))
        return cls(x0, y1, res, width, height)

    @classmethod
    def around(cls, lon: float, lat: float, size_m: float, ground_res: float,
               size_m_y: float | None = None) -> "Grid":
        """A grid centred on a point, ``size_m`` ground metres wide (and ``size_m_y`` tall)."""
        scale = 1 / math.cos(math.radians(lat))
        res = ground_res * scale
        width = max(1, int(round(size_m / ground_res)))
        height = max(1, int(round((size_m_y or size_m) / ground_res)))
        cx, cy = lonlat_to_merc(lon, lat)
        return cls(cx - width * res / 2, cy + height * res / 2, res, width, height)

    # ---- geometry -----------------------------------------------------------------------
    @property
    def x1(self) -> float:
        return self.x0 + self.width * self.res

    @property
    def y0(self) -> float:
        return self.y1 - self.height * self.res

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        lon0, lat0 = merc_to_lonlat(self.x0, self.y0)
        lon1, lat1 = merc_to_lonlat(self.x1, self.y1)
        return lon0, lat0, lon1, lat1

    @property
    def center_lonlat(self) -> tuple[float, float]:
        return merc_to_lonlat((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def ground_res(self) -> float:
        """Ground metres per pixel at the grid's centre latitude."""
        return self.res * math.cos(math.radians(self.center_lonlat[1]))

    def row_scale(self) -> np.ndarray:
        """cos(latitude) of every row centre: projected metres x this = ground metres."""
        ys = self.y1 - (np.arange(self.height) + 0.5) * self.res
        return np.cos(np.radians(merc_y_to_lat(ys)))

    def pixel_area_km2(self) -> np.ndarray:
        """Ground area (km^2) of one pixel in each row, shape ``(height, 1)``."""
        return ((self.res * self.row_scale()) ** 2 / 1e6)[:, None]

    @property
    def area_km2(self) -> float:
        return float(self.pixel_area_km2().sum() * self.width)

    @property
    def ground_size_km(self) -> tuple[float, float]:
        return self.width * self.ground_res / 1000, self.height * self.ground_res / 1000

    def with_resolution(self, ground_res: float) -> "Grid":
        """Same extent, different resolution."""
        res = ground_res / math.cos(math.radians(self.center_lonlat[1]))
        width = max(1, int(round((self.x1 - self.x0) / res)))
        height = max(1, int(round((self.y1 - self.y0) / res)))
        return Grid(self.x0, self.y1, res, width, height)

    def pixel_lonlat(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + (np.asarray(cols) + 0.5) * self.res
        ys = self.y1 - (np.asarray(rows) + 0.5) * self.res
        return np.degrees(xs / EARTH_RADIUS), merc_y_to_lat(ys)

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = lonlat_to_merc(lon, lat)
        return (self.y1 - y) / self.res, (x - self.x0) / self.res

    def world_file(self) -> str:
        """ESRI world file (.pgw/.jgw) text for this grid, so PNG exports open georeferenced in GIS."""
        return "\n".join(f"{v:.6f}" for v in (self.res, 0.0, 0.0, -self.res,
                                                self.x0 + self.res / 2, self.y1 - self.res / 2)) + "\n"

    def key(self) -> str:
        return f"{self.x0:.2f},{self.y1:.2f},{self.res:.4f},{self.width},{self.height}"


# WKT for EPSG:3857, written next to exported rasters (.prj).
WEB_MERCATOR_WKT = (
    'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],'
    'PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH],'
    'EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m '
    '+nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]'
)
