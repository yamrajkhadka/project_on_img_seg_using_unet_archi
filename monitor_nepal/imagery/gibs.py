"""NASA GIBS: daily global imagery since 2000 (MODIS) and 2012 (VIIRS), for context views."""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date

import numpy as np
from PIL import Image

from ..http import FetchError, get

WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
WMTS_3857 = ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{day}/"
             "GoogleMapsCompatible_Level{level}/{{z}}/{{y}}/{{x}}.{ext}")
ATTRIBUTION = "NASA EOSDIS Global Imagery Browse Services (GIBS)"


@dataclass(frozen=True)
class GibsLayer:
    id: str
    title: str
    start: date
    level: int  # max zoom of the GoogleMapsCompatible matrix
    ext: str
    note: str


LAYERS = {
    "modis_terra": GibsLayer("MODIS_Terra_CorrectedReflectance_TrueColor", "MODIS Terra true colour (250 m)",
                             date(2000, 2, 24), 9, "jpg", "Morning overpass (~10:30 local)."),
    "modis_aqua": GibsLayer("MODIS_Aqua_CorrectedReflectance_TrueColor", "MODIS Aqua true colour (250 m)",
                            date(2002, 7, 3), 9, "jpg", "Afternoon overpass (~13:30 local)."),
    "viirs_snpp": GibsLayer("VIIRS_SNPP_CorrectedReflectance_TrueColor", "VIIRS SNPP true colour (375 m)",
                            date(2015, 11, 24), 9, "jpg", "Afternoon overpass."),
    "viirs_noaa20": GibsLayer("VIIRS_NOAA20_CorrectedReflectance_TrueColor", "VIIRS NOAA-20 true colour",
                              date(2017, 12, 1), 9, "jpg", "Afternoon overpass."),
    "snow": GibsLayer("MODIS_Terra_NDSI_Snow_Cover", "MODIS snow cover (NDSI, 500 m)", date(2000, 2, 24), 8, "png",
                      "Daily snow cover fraction."),
    "fires": GibsLayer("VIIRS_SNPP_Thermal_Anomalies_375m_All", "VIIRS fire & thermal anomalies",
                       date(2012, 1, 20), 9, "png", "Active fire detections."),
    "aerosol": GibsLayer("MODIS_Combined_Value_Added_AOD", "Aerosol optical depth (MODIS)", date(2000, 2, 24), 6,
                         "png", "Smoke, dust and haze."),
}


def tile_url(layer_key: str, day: date) -> str:
    L = LAYERS[layer_key]
    return WMTS_3857.format(layer=L.id, day=day.isoformat(), level=L.level, ext=L.ext)


def snapshot(layer_key: str, day: date, bbox: tuple[float, float, float, float], width: int = 1024) -> np.ndarray:
    """Daily image of a lon/lat box (WMS, EPSG:4326)."""
    L = LAYERS[layer_key]
    if day < L.start:
        raise FetchError(f"{L.title} starts on {L.start.isoformat()}.")
    lon0, lat0, lon1, lat1 = bbox
    height = max(16, int(round(width * (lat1 - lat0) / (lon1 - lon0))))
    r = get(WMS, params={"SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": L.id,
                         "CRS": "EPSG:4326", "BBOX": f"{lat0},{lon0},{lat1},{lon1}", "WIDTH": width,
                         "HEIGHT": height, "FORMAT": "image/png" if L.ext == "png" else "image/jpeg",
                         "TIME": day.isoformat(), "TRANSPARENT": "TRUE"}, timeout=60)
    if not r.headers.get("content-type", "").startswith("image"):
        raise FetchError("NASA GIBS returned an error instead of an image.")
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"))
