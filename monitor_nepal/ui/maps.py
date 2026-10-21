"""Folium map builders shared by the pages."""
from __future__ import annotations

import base64
import io
from datetime import date, timedelta

import folium
import numpy as np
import pandas as pd
from PIL import Image

from .. import boundaries
from ..config import PALETTES
from ..geo import Grid
from ..imagery import eox, gibs

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
CARTO_LIGHT = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_DARK = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
TOPO = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
NIGHT_LIGHTS = ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_Black_Marble/default/2016-01-01/"
                "GoogleMapsCompatible_Level8/{z}/{y}/{x}.png")

FIRE = "#e34948"
QUAKE = "#6250d6"


def base_map(center: tuple[float, float] = (28.3, 84.1), zoom: int = 7, *, basemap: str = "imagery",
             nasa_day: date | None = None, labels: bool = True) -> folium.Map:
    """center is (lat, lon)."""
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True,
                   max_zoom=19, min_zoom=5)
    eox_year = eox.years()[-1]  # the newest annual mosaic, whichever year that is
    layers = [
        ("Satellite (Esri World Imagery)", ESRI, "Imagery © Esri, Vantor/Maxar, Earthstar Geographics", 19),
        (f"Sentinel-2 cloudless {eox_year} (EOX)", eox.annual_source(eox_year).url_template,
         f"Sentinel-2 cloudless {eox_year} © EOX (CC BY-NC-SA 4.0)", 17),
        ("Streets (light)", CARTO_LIGHT, "© OpenStreetMap contributors © CARTO", 19),
        ("Streets (dark)", CARTO_DARK, "© OpenStreetMap contributors © CARTO", 19),
        ("Topographic", TOPO, "© OpenStreetMap contributors, SRTM · style © OpenTopoMap (CC-BY-SA)", 17),
        ("Night lights 2016 (NASA Black Marble)", NIGHT_LIGHTS, gibs.ATTRIBUTION, 8),
    ]
    if nasa_day:
        layers.insert(1, (f"NASA MODIS true colour · {nasa_day:%d %b %Y}", gibs.tile_url("modis_terra", nasa_day),
                          gibs.ATTRIBUTION, 9))
    order = {"imagery": 0, "nasa": 1, "light": 3 if nasa_day else 2, "dark": 4 if nasa_day else 3}
    first = order.get(basemap, 0)
    for i, (name, url, attr, maxz) in enumerate(layers):
        folium.TileLayer(url, name=name, attr=attr, max_zoom=19, max_native_zoom=maxz, overlay=False,
                         control=True, show=i == first).add_to(m)
    if labels:
        folium.TileLayer(ESRI_LABELS, name="Place names", attr="Esri", overlay=True, control=True, max_zoom=19,
                         opacity=0.85).add_to(m)
    return m


def add_boundaries(m: folium.Map, level: int = 1, *, name: str | None = None, color: str = "#ffffff",
                   weight: float = 1.2, show: bool = True, tooltip: bool = True) -> None:
    gj = boundaries.geojson(level)
    field = f"a{level}_name"
    folium.GeoJson(gj, name=name or {0: "Nepal", 1: "Provinces", 2: "Districts", 3: "Local units"}[level],
                   style_function=lambda _f: {"color": color, "weight": weight, "fillOpacity": 0, "opacity": 0.85},
                   highlight_function=lambda _f: {"weight": weight + 1.5, "color": "#ffd166"},
                   tooltip=folium.GeoJsonTooltip(fields=[field], aliases=[""]) if tooltip else None,
                   show=show).add_to(m)


def add_region(m: folium.Map, region, *, color: str = "#ffd166") -> None:
    folium.GeoJson(boundaries.region_feature(region), name=region.short,
                   style_function=lambda _f: {"color": color, "weight": 2.5, "fillOpacity": 0.0}).add_to(m)


def add_box(m: folium.Map, grid: Grid, *, color: str = "#ffd166", name: str = "Analysis area") -> None:
    lon0, lat0, lon1, lat1 = grid.bounds_lonlat
    folium.Rectangle([[lat0, lon0], [lat1, lon1]], color=color, weight=2, fill=False, name=name).add_to(m)


def png_data_uri(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def class_rgba(classes: np.ndarray, palette: str = "cartographic", alpha: int = 255,
               valid: np.ndarray | None = None) -> np.ndarray:
    pal = np.array(PALETTES[palette], np.uint8)
    rgba = np.concatenate([pal[classes], np.full(classes.shape + (1,), alpha, np.uint8)], -1)
    if valid is not None:
        rgba[~valid, 3] = 0
    return rgba


def add_raster(m: folium.Map, rgba: np.ndarray, grid: Grid, *, name: str, opacity: float = 0.75,
               max_side: int = 1600, show: bool = True) -> None:
    h, w = rgba.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    img = rgba[::step, ::step]
    lon0, lat0, lon1, lat1 = grid.bounds_lonlat
    folium.raster_layers.ImageOverlay(png_data_uri(img), bounds=[[lat0, lon0], [lat1, lon1]], opacity=opacity,
                                      name=name, interactive=False, cross_origin=False, zindex=5, show=show,
                                      pixelated=True).add_to(m)


def add_fires(m: folium.Map, df: pd.DataFrame, *, name: str = "Active fires") -> None:
    if df is None or df.empty:
        return
    fg = folium.FeatureGroup(name=f"{name} ({len(df)})", show=True)
    for r in df.itertuples():
        radius = 3 + min(7, (float(r.frp) if pd.notna(r.frp) else 1) ** 0.5)
        folium.CircleMarker((r.latitude, r.longitude), radius=radius, color="#7a1712", weight=1, fill=True,
                            fill_color=FIRE, fill_opacity=0.85,
                            tooltip=f"🔥 {r.district}, {r.province} · {r.time_utc:%d %b %H:%M} UTC · "
                                    f"{r.sensor} · confidence {r.confidence_class} · FRP {r.frp} MW").add_to(fg)
    fg.add_to(m)


def add_quakes(m: folium.Map, df: pd.DataFrame, *, name: str = "Earthquakes") -> None:
    if df is None or df.empty:
        return
    fg = folium.FeatureGroup(name=f"{name} ({len(df)})", show=True)
    for r in df.itertuples():
        mag = float(r.magnitude or 0)
        folium.CircleMarker((r.latitude, r.longitude), radius=max(4, 2.2 ** (mag - 1.2)), color=QUAKE, weight=2,
                            fill=True, fill_color=QUAKE, fill_opacity=0.25,
                            tooltip=f"M{mag:.1f} · {r.time_npt:%d %b %Y %H:%M} NPT · depth {r.depth_km:.0f} km · "
                                    f"{r.place}").add_to(fg)
    fg.add_to(m)


def add_places(m: folium.Map, df: pd.DataFrame, *, name: str = "Weather now") -> None:
    if df is None or df.empty:
        return
    fg = folium.FeatureGroup(name=name, show=True)
    for r in df.itertuples():
        t = "—" if pd.isna(r.temperature_c) else f"{r.temperature_c:.0f}°C"
        aqi = "" if pd.isna(r.us_aqi) else f" · AQI {int(r.us_aqi)}"
        folium.Marker((r.lat, r.lon), icon=folium.DivIcon(
            html=f'<div style="font:600 11px system-ui,sans-serif;color:#fff;background:rgba(17,17,16,.72);'
                 f'border-radius:9px;padding:1px 6px;white-space:nowrap;display:inline-block;'
                 f'transform:translate(-50%,-50%)">{r.place} {t}</div>'),
            tooltip=f"{r.place}: {t}, {r.condition}{aqi}").add_to(fg)
    fg.add_to(m)


def _mix(hex_color: str, other: tuple[int, int, int], t: float) -> str:
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(int(round(c + (o - c) * t)) for c, o in zip(rgb, other))


def sequential_ramp(hex_color: str, n: int = 5) -> list[str]:
    """One hue, light -> dark, built from a class colour (light tint ... darker shade)."""
    steps = []
    for i in range(n):
        t = i / (n - 1)
        if t <= 0.5:  # pale tint -> the class hue itself
            steps.append(_mix(hex_color, (246, 245, 241), 0.85 * (1 - t / 0.5)))
        else:  # the class hue -> a deeper shade
            steps.append(_mix(hex_color, (0, 0, 0), 0.45 * (t - 0.5) / 0.5))
    return steps


def nice_breaks(values, n: int = 5) -> list[float]:
    vmax = float(np.nanmax(values)) if len(values) else 1.0
    step = max(1.0, np.ceil(vmax / n / 5) * 5) if vmax > 10 else max(0.5, np.ceil(vmax / n * 2) / 2)
    return [round(step * i, 2) for i in range(n + 1)]


def add_choropleth(m: folium.Map, level: int, values: dict[str, float], *, label: str, color: str,
                   extra: dict[str, str] | None = None) -> tuple[list[float], list[str]]:
    """Colour admin units by value (pcode -> value).  Returns (breaks, colours) for a legend."""
    gj = boundaries.geojson(level)
    field = f"a{level}_pcode"
    vals = [v for v in values.values() if v is not None and not np.isnan(v)]
    breaks = nice_breaks(vals)
    ramp = sequential_ramp(color, len(breaks) - 1)
    for f in gj["features"]:
        code = f["properties"][field]
        v = values.get(code)
        f["properties"]["value"] = "no data" if v is None or np.isnan(v) else f"{v:.1f}%"
        f["properties"]["extra"] = (extra or {}).get(code, "")

    def style(feature):
        v = values.get(feature["properties"][field])
        if v is None or np.isnan(v):
            return {"fillColor": "#bdbbb4", "fillOpacity": 0.5, "color": "#ffffff", "weight": 0.6}
        idx = min(len(ramp) - 1, max(0, int(np.searchsorted(breaks, v, side="right") - 1)))
        return {"fillColor": ramp[idx], "fillOpacity": 0.85, "color": "#ffffff", "weight": 0.6}

    folium.GeoJson(gj, name=label, style_function=style,
                   highlight_function=lambda _f: {"weight": 2.2, "color": "#111110"},
                   tooltip=folium.GeoJsonTooltip(fields=[f"a{level}_name", "value", "extra"],
                                                 aliases=["", label, ""])).add_to(m)
    return breaks, ramp


def legend_html(breaks: list[float], ramp: list[str], unit: str = "%") -> str:
    items = "".join(f'<span style="display:inline-flex;align-items:center;margin-right:10px">'
                    f'<span style="width:16px;height:12px;background:{c};display:inline-block;margin-right:4px;'
                    f'border-radius:2px;box-shadow:0 0 0 1px rgba(127,127,127,.35) inset"></span>'
                    f'{breaks[i]:g}–{breaks[i + 1]:g}{unit}</span>' for i, c in enumerate(ramp))
    return f'<div style="font-size:.85rem;display:flex;flex-wrap:wrap;gap:2px 0">{items}</div>'


def finish(m: folium.Map) -> folium.Map:
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    return m


def yesterday() -> date:
    return date.today() - timedelta(days=1)
