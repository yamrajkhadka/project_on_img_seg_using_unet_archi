"""Cached data access for the app.  TTLs follow how fast each source changes."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from .. import boundaries
from ..analysis import pipeline
from ..geo import Grid
from ..history import sources
from ..imagery import ImagerySpec, esri, gibs, planetary
from ..live import fires, quakes, weather
from ..model import get_model

MIN = 60
HOUR = 3600
DAY = 86400


@st.cache_resource(show_spinner=False)
def model(key: str):
    return get_model(key)


# ---- imagery catalogues --------------------------------------------------------------------------
@st.cache_data(ttl=DAY, show_spinner=False)
def wayback_releases():
    return esri.wayback_releases()


@st.cache_data(ttl=DAY, show_spinner=False)
def local_versions(lon: float, lat: float):
    return esri.local_versions(round(lon, 4), round(lat, 4))


@st.cache_data(ttl=7 * DAY, show_spinner=False)
def capture_info(lon: float, lat: float, metadata_url: str):
    return esri.capture_info(round(lon, 4), round(lat, 4), metadata_url)


@st.cache_data(ttl=DAY, show_spinner=False)
def current_capture(lon: float, lat: float):
    return esri.current_capture_info(round(lon, 4), round(lat, 4))


@st.cache_data(ttl=HOUR, show_spinner=False)
def scenes(product: str, bbox: tuple, start: date, end: date, cloud_max: float | None = None):
    return planetary.search(product, bbox, start, end, cloud_max)


@st.cache_data(ttl=7 * DAY, show_spinner=False, max_entries=64)
def gibs_snapshot(layer: str, day: date, bbox: tuple, width: int = 900):
    return gibs.snapshot(layer, day, bbox, width)


# ---- live feeds -------------------------------------------------------------------------------------------
@st.cache_data(ttl=15 * MIN, show_spinner=False)
def fires_live(window: str = "24h") -> pd.DataFrame:
    return fires.live(window)


@st.cache_data(ttl=7 * DAY, show_spinner=False, max_entries=40)
def fires_archive(year: int, product: str = "VIIRS S-NPP") -> pd.DataFrame:
    return fires.archive(year, product)


def _firms_key() -> str | None:
    try:
        return st.secrets.get("FIRMS_MAP_KEY")
    except Exception:  # no secrets file
        return None


@st.cache_data(ttl=6 * HOUR, show_spinner=False, max_entries=32)
def _fires_api(start: date, days: int, product: str) -> pd.DataFrame:
    source = "MODIS_SP" if product == "MODIS" else "VIIRS_SNPP_SP"
    return fires.api_range(_firms_key(), start, days, source)


def fires_period(start: date, end: date, product: str = "VIIRS S-NPP") -> tuple[pd.DataFrame, list[str]]:
    """All detections in Nepal between two dates for one sensor, plus notes on any coverage gaps."""
    frames, notes = [], []
    today = date.today()
    last_archive = today.year - 1
    first_year = fires.ARCHIVE_PRODUCTS[product][1]
    if start.year < first_year:
        notes.append(f"{product} fire records begin in {first_year}.")
    for y in range(max(start.year, first_year), min(end.year, last_archive) + 1):
        df = fires_archive(y, product)
        if not df.empty:
            d = df.time_utc.dt.date
            frames.append(df[(d >= start) & (d <= end)])
    if end.year > last_archive:
        live_start = today - timedelta(days=6)
        cur_start = max(start, date(today.year, 1, 1))
        if cur_start < live_start:
            gap_end = min(end, live_start - timedelta(days=1))
            if _firms_key():
                frames.append(_fires_api(cur_start, (gap_end - cur_start).days + 1, product))
            else:
                notes.append(f"{cur_start:%d %b} – {gap_end:%d %b %Y}: FIRMS publishes this year's archive after it "
                             "ends; add a free FIRMS_MAP_KEY to the app secrets to fill the gap.")
        if end >= live_start:
            sensors = ("MODIS",) if product == "MODIS" else ("VIIRS S-NPP",)
            live = fires.live("7d", sensors=sensors)
            if not live.empty:
                d = (live.time_utc + timedelta(hours=5, minutes=45)).dt.date
                frames.append(live[(d >= max(start, live_start)) & (d <= end)])
    frames = [f for f in frames if f is not None and not f.empty]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["latitude", "longitude", "time_utc", "frp", "confidence_class", "daynight", "sensor", "district",
                 "province"])
    return out, notes


@st.cache_data(ttl=10 * MIN, show_spinner=False)
def quakes_recent(days: int = 30, min_magnitude: float = 2.5) -> pd.DataFrame:
    return quakes.recent(days, min_magnitude)


@st.cache_data(ttl=DAY, show_spinner=False, max_entries=64)
def quakes_between(start: date, end: date, min_magnitude: float) -> pd.DataFrame:
    return quakes.query(start, end, min_magnitude)


@st.cache_data(ttl=15 * MIN, show_spinner=False)
def weather_now() -> pd.DataFrame:
    return weather.current()


@st.cache_data(ttl=HOUR, show_spinner=False)
def rain_forecast() -> pd.DataFrame:
    return weather.forecast_rain()


@st.cache_data(ttl=3 * HOUR, show_spinner=False)
def rivers_recent() -> pd.DataFrame:
    from ..live import rivers

    return rivers.recent()


@st.cache_data(ttl=DAY, show_spinner=False, max_entries=64)
def rivers_on_day(day: date) -> pd.DataFrame:
    from ..live import rivers

    return rivers.on_day(day)


@st.cache_data(show_spinner=False)
def rivers_annual() -> pd.DataFrame | None:
    from ..live import rivers

    try:
        return rivers.annual()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def rivers_reference() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    from ..live import rivers

    try:
        return rivers.climatology(), rivers.thresholds()
    except FileNotFoundError:
        return None


@st.cache_data(ttl=15 * MIN, show_spinner=False, max_entries=16)
def last_24h(lat: float, lon: float) -> pd.DataFrame:
    return weather.last_24h(lat, lon)


@st.cache_data(ttl=6 * HOUR, show_spinner=False, max_entries=128)
def weather_hourly(lat: float, lon: float, day: date) -> pd.DataFrame:
    return weather.hourly(round(lat, 3), round(lon, 3), day)


@st.cache_data(ttl=6 * HOUR, show_spinner=False, max_entries=128)
def weather_daily(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    return weather.daily(round(lat, 3), round(lon, 3), start, end)


@st.cache_data(ttl=6 * HOUR, show_spinner=False, max_entries=64)
def air_hourly(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    return weather.air_quality_hourly(round(lat, 3), round(lon, 3), start, end)


# ---- history -------------------------------------------------------------------------------------------------
@st.cache_data(ttl=7 * DAY, show_spinner=False)
def wb(key: str) -> pd.DataFrame:
    return sources.world_bank(key)


@st.cache_data(ttl=7 * DAY, show_spinner=False)
def owid(key: str) -> pd.DataFrame:
    return sources.owid(key)


@st.cache_data(ttl=7 * DAY, show_spinner=False)
def climate(key: str) -> pd.DataFrame:
    return sources.climate(key)


# ---- analyses (memoised: re-opening a result is instant) ------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=10)
def classify_area(grid: Grid, spec: ImagerySpec, model_key: str, tta: int, min_object_px: int,
                  _progress=None) -> pipeline.AreaResult:
    model(model_key)
    return pipeline.classify_area(grid, spec, model_key, tta=tta, min_object_px=min_object_px, progress=_progress)


@st.cache_data(show_spinner=False, max_entries=16)
def survey(region_code: str, spec: ImagerySpec, model_key: str, n: int, seed: int, tta: int,
           _progress=None) -> pipeline.SurveyResult:
    model(model_key)
    return pipeline.survey(boundaries.get(region_code), spec, model_key, n=n, seed=seed, tta=tta, progress=_progress)


@st.cache_data(show_spinner=False, max_entries=16)
def survey_series(region_code: str, epochs: tuple[tuple[str, ImagerySpec], ...], model_key: str, n: int, seed: int,
                  tta: int, _progress=None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """The same sample windows classified at several epochs.

    Returns a long table (label, key, class, share, low, high, km2, n) and the per-window fractions per epoch
    (for paired change estimates).
    """
    model(model_key)
    region = boundaries.get(region_code)
    pts = pipeline.sample_points(region, n, seed)
    rows, fracs = [], {}
    for i, (label, spec) in enumerate(epochs):
        def prog(f, t, i=i, label=label):
            if _progress:
                _progress((i + f) / len(epochs), f"{label}: {t}")
        res = pipeline.survey(region, spec, model_key, n=n, seed=seed, tta=tta, points=pts, keep_thumbnails=0,
                              progress=prog)
        fracs[label] = res.fractions
        rows.append(res.estimate.assign(label=label))
    return pd.concat(rows, ignore_index=True), fracs


@st.cache_data(show_spinner=False, max_entries=32)
def reference_fractions(region_code: str, n: int, seed: int, year: int) -> np.ndarray:
    pts = pipeline.sample_points(boundaries.get(region_code), n, seed)
    return pipeline.reference_survey(pts, year)


def parallel(calls: dict) -> dict:
    """Run independent fetches concurrently: {name: (fn, args...)} -> {name: (result, error)}."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

    from ..http import FetchError

    ctx = get_script_run_ctx()

    def run(item):
        name, (fn, *args) = item
        add_script_run_ctx(threading.current_thread(), ctx)
        try:
            return name, (fn(*args), None)
        except (FetchError, ValueError, KeyError) as exc:
            return name, (None, str(exc))

    with ThreadPoolExecutor(max_workers=max(1, len(calls))) as ex:
        return dict(ex.map(run, calls.items()))


def recent_days(n: int) -> tuple[date, date]:
    end = date.today()
    return end - timedelta(days=n - 1), end
