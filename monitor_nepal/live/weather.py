"""Weather, climate reanalysis and air quality from Open-Meteo (free, no key).

* forecast API  - current conditions and hourly data for the last ~3 months and next days
* archive API   - ERA5 reanalysis, hourly/daily, 1940 to ~5 days ago
* air-quality   - CAMS PM2.5 / PM10 / US AQI, current and historical
Open-Meteo is free for non-commercial use with fair limits, so long histories are bundled offline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from ..http import get_json

FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"
TZ = "Asia/Kathmandu"
ATTRIBUTION = "Weather: Open-Meteo.com (ERA5 / ECMWF, DWD, NOAA), air quality: CAMS"


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    note: str = ""


PLACES = (
    Place("Kathmandu", 27.7172, 85.3240, "Capital, Bagmati"),
    Place("Pokhara", 28.2096, 83.9856, "Gandaki"),
    Place("Biratnagar", 26.4525, 87.2718, "Koshi"),
    Place("Janakpur", 26.7288, 85.9263, "Madhesh"),
    Place("Birgunj", 27.0104, 84.8773, "Madhesh"),
    Place("Bharatpur", 27.6833, 84.4333, "Bagmati"),
    Place("Butwal", 27.7006, 83.4483, "Lumbini"),
    Place("Nepalgunj", 28.0500, 81.6167, "Lumbini"),
    Place("Birendranagar", 28.6019, 81.6339, "Karnali"),
    Place("Dhangadhi", 28.6833, 80.6000, "Sudur Paschim"),
    Place("Jomsom", 28.7804, 83.7232, "Mustang, 2,700 m"),
    Place("Namche Bazaar", 27.8048, 86.7140, "Everest region, 3,440 m"),
)

WEATHER_CODES = {0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Rime fog",
                 51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle", 61: "Light rain", 63: "Rain",
                 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain", 71: "Light snow", 73: "Snow",
                 75: "Heavy snow", 77: "Snow grains", 80: "Rain showers", 81: "Rain showers",
                 82: "Violent showers", 85: "Snow showers", 86: "Snow showers", 95: "Thunderstorm",
                 96: "Thunderstorm, hail", 99: "Thunderstorm, hail"}

# US EPA AQI categories (value upper bound, label, short label, status token)
AQI_BANDS = ((50, "Good", "Good", "good"), (100, "Moderate", "Moderate", "warning"),
             (150, "Unhealthy for sensitive groups", "Sensitive groups", "serious"),
             (200, "Unhealthy", "Unhealthy", "critical"), (300, "Very unhealthy", "Very unhealthy", "critical"),
             (10_000, "Hazardous", "Hazardous", "critical"))


def aqi_band(value: float | None, short: bool = False) -> tuple[str, str]:
    """(label, status) for a US AQI value."""
    if value is None or pd.isna(value):
        return "No data", "neutral"
    for hi, label, short_label, status in AQI_BANDS:
        if value <= hi:
            return (short_label if short else label), status
    return "Hazardous", "critical"


def _as_list(data) -> list[dict]:
    return data if isinstance(data, list) else [data]


def current(places: tuple[Place, ...] = PLACES) -> pd.DataFrame:
    """Current weather + air quality for several places in two requests."""
    lat = ",".join(str(p.lat) for p in places)
    lon = ",".join(str(p.lon) for p in places)
    w = _as_list(get_json(FORECAST, params={
        "latitude": lat, "longitude": lon, "timezone": TZ,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,"
                   "cloud_cover,wind_speed_10m,is_day"}))
    try:
        a = _as_list(get_json(AIR, params={"latitude": lat, "longitude": lon, "timezone": TZ,
                                           "current": "us_aqi,pm2_5,pm10"}))
    except Exception:  # air quality is a bonus; never block the weather
        a = [{} for _ in places]
    rows = []
    for p, wi, ai in zip(places, w, a):
        c, q = wi.get("current", {}), ai.get("current", {}) if ai else {}
        rows.append({"place": p.name, "note": p.note, "lat": p.lat, "lon": p.lon, "elevation_m": wi.get("elevation"),
                     "time": c.get("time"), "temperature_c": c.get("temperature_2m"),
                     "feels_like_c": c.get("apparent_temperature"), "humidity_pct": c.get("relative_humidity_2m"),
                     "precip_mm": c.get("precipitation"), "cloud_pct": c.get("cloud_cover"),
                     "wind_kmh": c.get("wind_speed_10m"),
                     "condition": WEATHER_CODES.get(c.get("weather_code"), "—"), "is_day": c.get("is_day"),
                     "us_aqi": q.get("us_aqi"), "pm2_5": q.get("pm2_5"), "pm10": q.get("pm10")})
    return pd.DataFrame(rows)


def hourly(lat: float, lon: float, day: date) -> pd.DataFrame:
    """Hourly weather on one local day (ERA5 for the past, forecast model for recent/upcoming days)."""
    recent = day >= date.today() - timedelta(days=60)
    url = FORECAST if recent else ARCHIVE
    params = {"latitude": lat, "longitude": lon, "timezone": TZ, "start_date": day.isoformat(),
              "end_date": day.isoformat(),
              "hourly": "temperature_2m,relative_humidity_2m,precipitation,snowfall,cloud_cover,wind_speed_10m,"
                        "shortwave_radiation"}
    data = get_json(url, params=params)
    h = data.get("hourly", {})
    df = pd.DataFrame(h)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    df.attrs["source"] = "Open-Meteo forecast (ECMWF/DWD blend)" if recent else "ERA5 reanalysis (Open-Meteo archive)"
    df.attrs["elevation"] = data.get("elevation")
    return df


def daily(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Daily weather between two dates (ERA5 + recent forecast model for the last few days)."""
    variables = ("temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,snowfall_sum,"
                 "sunshine_duration")
    frames = []
    cutoff = date.today() - timedelta(days=6)
    if start <= cutoff:
        data = get_json(ARCHIVE, params={"latitude": lat, "longitude": lon, "timezone": TZ,
                                         "start_date": start.isoformat(), "end_date": min(end, cutoff).isoformat(),
                                         "daily": variables})
        frames.append(pd.DataFrame(data.get("daily", {})))
    if end > cutoff:
        data = get_json(FORECAST, params={"latitude": lat, "longitude": lon, "timezone": TZ,
                                          "start_date": max(start, cutoff + timedelta(days=1)).isoformat(),
                                          "end_date": min(end, date.today() + timedelta(days=7)).isoformat(),
                                          "daily": variables})
        frames.append(pd.DataFrame(data.get("daily", {})))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


def forecast_rain(places: tuple[Place, ...] = PLACES, days: int = 2) -> pd.DataFrame:
    """Forecast precipitation (mm) for today and the next day(s) at each place, one request."""
    data = _as_list(get_json(FORECAST, params={
        "latitude": ",".join(str(p.lat) for p in places), "longitude": ",".join(str(p.lon) for p in places),
        "timezone": TZ, "forecast_days": days, "daily": "precipitation_sum,precipitation_probability_max"}))
    rows = []
    for p, d in zip(places, data):
        daily = d.get("daily", {})
        for t, mm, prob in zip(daily.get("time", []), daily.get("precipitation_sum", []),
                               daily.get("precipitation_probability_max", [])):
            rows.append({"place": p.name, "date": t, "rain_mm": mm, "probability": prob})
    return pd.DataFrame(rows)


def last_24h(lat: float, lon: float) -> pd.DataFrame:
    """Hourly temperature and US AQI for the past 24 hours (for sparklines)."""
    w = get_json(FORECAST, params={"latitude": lat, "longitude": lon, "timezone": TZ, "past_days": 1,
                                   "forecast_days": 1, "hourly": "temperature_2m"})
    df = pd.DataFrame(w.get("hourly", {}))
    try:
        a = get_json(AIR, params={"latitude": lat, "longitude": lon, "timezone": TZ, "past_days": 1,
                                  "forecast_days": 1, "hourly": "us_aqi"})
        df = df.merge(pd.DataFrame(a.get("hourly", {})), on="time", how="left")
    except Exception:
        df["us_aqi"] = None
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"])
    now = pd.Timestamp.now(tz=TZ).tz_localize(None)
    return df[(df.time <= now) & (df.time > now - pd.Timedelta(hours=24))].reset_index(drop=True)


def air_quality_hourly(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    data = get_json(AIR, params={"latitude": lat, "longitude": lon, "timezone": TZ, "start_date": start.isoformat(),
                                 "end_date": end.isoformat(), "hourly": "pm2_5,pm10,us_aqi"})
    df = pd.DataFrame(data.get("hourly", {}))
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df
