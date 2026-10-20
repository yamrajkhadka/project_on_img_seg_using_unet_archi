"""River flow on Nepal's major rivers from GloFAS (via Open-Meteo): last 30 days, 7-day forecast, and how that
compares with the 2004-2023 climatology of the same model.

GloFAS is a hydrological *model* (ECMWF / Copernicus Emergency Management Service) on a 0.05° grid, not gauge
readings.  Each river uses the model cell on its main stem near a well-known gauging point (found by searching the
neighbourhood for the highest-flow cell).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import HISTORY_DIR
from ..http import get_json

FLOOD = "https://flood-api.open-meteo.com/v1/flood"
ATTRIBUTION = "River discharge: GloFAS v4 (Copernicus EMS / ECMWF) via Open-Meteo"


@dataclass(frozen=True)
class River:
    key: str
    name: str
    gauge: str
    lat: float
    lon: float
    province: str


RIVERS = (
    River("koshi", "Koshi", "Chatara", 26.775, 87.075, "Koshi"),
    River("narayani", "Narayani (Gandaki)", "Devghat", 27.675, 84.275, "Gandaki / Bagmati"),
    River("karnali", "Karnali", "Chisapani", 28.575, 81.225, "Karnali / Sudur Paschim"),
    River("mahakali", "Mahakali", "Banbasa", 28.825, 80.075, "Sudur Paschim"),
    River("bagmati", "Bagmati", "Karmaiya", 26.975, 85.325, "Madhesh"),
    River("rapti", "West Rapti", "Jalkundi", 28.025, 82.075, "Lumbini"),
)
BY_KEY = {r.key: r for r in RIVERS}


def recent(past_days: int = 30, forecast_days: int = 7) -> pd.DataFrame:
    """Daily discharge for every river: the last ``past_days`` and the next ``forecast_days`` (one request)."""
    data = get_json(FLOOD, params={"latitude": ",".join(str(r.lat) for r in RIVERS),
                                   "longitude": ",".join(str(r.lon) for r in RIVERS), "daily": "river_discharge",
                                   "past_days": past_days, "forecast_days": forecast_days}, timeout=60)
    data = data if isinstance(data, list) else [data]
    today = pd.Timestamp.now(tz="Asia/Kathmandu").normalize().tz_localize(None)
    rows = []
    for r, item in zip(RIVERS, data):
        d = item.get("daily", {})
        for t, q in zip(d.get("time", []), d.get("river_discharge", [])):
            ts = pd.Timestamp(t)
            rows.append({"river": r.key, "date": ts, "discharge": q, "forecast": ts > today})
    return pd.DataFrame(rows)


def on_day(day) -> pd.DataFrame:
    """Discharge of every river on one date (GloFAS reanalysis from 1984; one light request)."""
    data = get_json(FLOOD, params={"latitude": ",".join(str(r.lat) for r in RIVERS),
                                   "longitude": ",".join(str(r.lon) for r in RIVERS), "daily": "river_discharge",
                                   "start_date": str(day), "end_date": str(day)}, timeout=60)
    data = data if isinstance(data, list) else [data]
    rows = []
    for r, item in zip(RIVERS, data):
        q = (item.get("daily", {}).get("river_discharge") or [None])[0]
        rows.append({"river": r.key, "name": r.name, "gauge": r.gauge, "discharge": q})
    return pd.DataFrame(rows)


def annual() -> pd.DataFrame:
    """Bundled annual peak and mean flow per river (river, year, peak, mean, peak_date)."""
    return pd.read_csv(HISTORY_DIR / "rivers_annual.csv")


def history(river: River, start: str, end: str) -> pd.DataFrame:
    data = get_json(FLOOD, params={"latitude": river.lat, "longitude": river.lon, "daily": "river_discharge",
                                   "start_date": start, "end_date": end}, timeout=120)
    d = data.get("daily", {})
    return pd.DataFrame({"date": pd.to_datetime(d.get("time", [])), "discharge": d.get("river_discharge", [])})


def climatology() -> pd.DataFrame:
    """Bundled 2004-2023 day-of-year percentiles (river, doy, p10, p50, p90)."""
    return pd.read_csv(HISTORY_DIR / "rivers_climatology.csv")


def thresholds() -> pd.DataFrame:
    """Bundled flood levels: q2 / q10 = 2- and 10-year annual-maximum discharge (river, q2, q10, mean)."""
    return pd.read_csv(HISTORY_DIR / "rivers_thresholds.csv")


def status(discharge: float, q2: float, q10: float, p90: float) -> tuple[str, str]:
    """(label, status token) for one day's discharge."""
    if discharge >= q10:
        return "Above 10-year flood level", "critical"
    if discharge >= q2:
        return "Above 2-year flood level", "serious"
    if discharge >= p90:
        return "Well above normal", "warning"
    return "Normal range", "good"
