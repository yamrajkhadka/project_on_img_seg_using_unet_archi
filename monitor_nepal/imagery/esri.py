"""Esri World Imagery (current) and the World Imagery Wayback archive (2014 onwards).

Wayback publishes a full snapshot of the World Imagery basemap every few weeks.  Most places do
not change between snapshots, so for time series we ask the ``tilemap`` service which release
actually *changed* at a location and keep only those distinct versions.  The metadata service
then tells us the real capture date, sensor and resolution of the pixels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from ..geo import lonlat_to_tile
from ..http import FetchError, get_json
from .base import XYZSource

CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
WAYBACK_TILE = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
                "default028mm/MapServer/tile/{release}/{{z}}/{{y}}/{{x}}")
TILEMAP = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tilemap/{release}/{z}/{y}/{x}/1/1"
CURRENT_TILE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ATTRIBUTION = "Imagery © Esri, Vantor (Maxar), Earthstar Geographics and the GIS user community"


@dataclass(frozen=True)
class WaybackRelease:
    number: int
    date: date
    metadata_url: str

    @property
    def label(self) -> str:
        return f"Wayback {self.date.isoformat()}"


@lru_cache(maxsize=1)
def _releases_cached() -> tuple[WaybackRelease, ...]:
    cfg = get_json(CONFIG_URL)
    out = []
    for num, item in cfg.items():
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", item.get("itemTitle", ""))
        if not m:
            continue
        out.append(WaybackRelease(int(num), date(int(m[1]), int(m[2]), int(m[3])), item.get("metadataLayerUrl", "")))
    if not out:
        raise FetchError("The Wayback configuration was empty.")
    return tuple(sorted(out, key=lambda r: r.date, reverse=True))


def wayback_releases() -> list[WaybackRelease]:
    """All Wayback releases, newest first."""
    return list(_releases_cached())


def current_source() -> XYZSource:
    return XYZSource(key="esri", name="Esri World Imagery (latest)", url_template=CURRENT_TILE,
                     attribution=ATTRIBUTION, max_zoom=18, native_gsd=0.5,
                     label="Esri World Imagery · latest mosaic")


def wayback_source(release: WaybackRelease) -> XYZSource:
    return XYZSource(key=f"wayback-{release.number}", name="Esri Wayback", attribution=ATTRIBUTION,
                     url_template=WAYBACK_TILE.format(release=release.number), max_zoom=18, native_gsd=0.5,
                     label=f"Esri Wayback · release {release.date.isoformat()}",
                     meta={"release": release.number, "release_date": release.date.isoformat()})


def local_versions(lon: float, lat: float, zoom: int = 16, limit: int = 30) -> list[WaybackRelease]:
    """Releases in which the imagery at (lon, lat) actually changed, newest first.

    One small ``tilemap`` request per distinct version, spaced out so the service is not hammered.
    """
    import time

    releases = wayback_releases()
    by_number = {r.number: r for r in releases}
    x, y = lonlat_to_tile(lon, lat, zoom)
    found: list[WaybackRelease] = []
    i = 0
    while i < len(releases) and len(found) < limit:
        rel = releases[i]
        if found:
            time.sleep(0.2)
        tm = get_json(TILEMAP.format(release=rel.number, z=zoom, y=y, x=x))
        if not tm.get("valid") or not tm.get("data") or tm["data"][0] == 0:
            break
        chosen = by_number.get((tm.get("select") or [rel.number])[0], rel)
        if not found or found[-1].number != chosen.number:
            found.append(chosen)
        older = [j for j, r in enumerate(releases) if r.date < chosen.date]
        if not older:
            break
        i = older[0]
    return found


@dataclass(frozen=True)
class CaptureInfo:
    date: date | None
    resolution_m: float | None
    accuracy_m: float | None
    sensor: str | None
    provider: str | None

    @property
    def text(self) -> str:
        bits = []
        if self.date:
            bits.append(f"captured {self.date.isoformat()}")
        if self.sensor:
            bits.append(self.sensor)
        if self.resolution_m:
            bits.append(f"{self.resolution_m:.2f} m")
        return " · ".join(bits) if bits else "capture date unknown"


SENSORS = {"WV01": "WorldView-1", "WV02": "WorldView-2", "WV03": "WorldView-3", "WV04": "WorldView-4",
           "GE01": "GeoEye-1", "QB02": "QuickBird", "IK02": "IKONOS", "PL": "Pléiades", "SPOT": "SPOT",
           **{f"LG0{i}": f"WorldView Legion {i}" for i in range(1, 7)}}


@lru_cache(maxsize=4096)
def capture_info(lon: float, lat: float, metadata_url: str, zoom: int = 16) -> CaptureInfo:
    """What the imagery at a point really is: capture date, sensor and resolution."""
    if not metadata_url:
        return CaptureInfo(None, None, None, None, None)
    d = 0.01
    data = get_json(f"{metadata_url}/identify", params={
        "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "sr": 4326, "layers": "all",
        "tolerance": 0, "mapExtent": f"{lon - d},{lat - d},{lon + d},{lat + d}", "imageDisplay": "400,400,96",
        "returnGeometry": "false", "f": "json"})
    best = None
    for res in data.get("results", []):
        a = res.get("attributes", {})
        lo, hi = a.get("MinMapLevel"), a.get("MaxMapLevel")
        try:
            if lo is not None and hi is not None and int(lo) <= zoom <= int(hi):
                best = a
                break
        except ValueError:
            pass
        best = best or a
    if not best:
        return CaptureInfo(None, None, None, None, None)
    raw = str(best.get("SRC_DATE") or "")
    when = date(int(raw[:4]), int(raw[4:6]), int(raw[6:8])) if re.fullmatch(r"\d{8}", raw) else None

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    sensor = best.get("SRC_DESC")
    return CaptureInfo(when, num(best.get("SRC_RES")), num(best.get("SRC_ACC")), SENSORS.get(sensor, sensor),
                       best.get("NICE_DESC"))


def current_capture_info(lon: float, lat: float) -> CaptureInfo:
    latest = wayback_releases()[0]
    return capture_info(round(lon, 5), round(lat, 5), latest.metadata_url)
