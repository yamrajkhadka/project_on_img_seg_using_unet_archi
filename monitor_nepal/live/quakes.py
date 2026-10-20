"""Earthquakes in and around Nepal from the USGS FDSN event service (1900-present, live)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .. import boundaries
from ..geo import haversine_km
from ..http import get_json

FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
ATTRIBUTION = "USGS Earthquake Hazards Program (ComCat)"
# Nepal plus a ~100 km margin: Himalayan quakes just across the border are felt strongly in Nepal.
SEARCH_BOX = (79.2, 25.4, 89.2, 31.3)
KATHMANDU = (85.3240, 27.7172)
NPT = timedelta(hours=5, minutes=45)


FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{level}_{period}.geojson"


def query(start: date | datetime, end: date | datetime | None = None, min_magnitude: float = 2.5,
          limit: int = 20000) -> pd.DataFrame:
    params = {"format": "geojson", "starttime": start.isoformat(), "minmagnitude": min_magnitude,
              "minlongitude": SEARCH_BOX[0], "minlatitude": SEARCH_BOX[1], "maxlongitude": SEARCH_BOX[2],
              "maxlatitude": SEARCH_BOX[3], "orderby": "time", "limit": limit}
    if end is not None:
        params["endtime"] = end.isoformat()
    return _frame(get_json(FDSN, params=params, timeout=60).get("features", []))


def _frame(features: list[dict]) -> pd.DataFrame:
    rows = []
    for f in features:
        p, (lon, lat, depth) = f["properties"], f["geometry"]["coordinates"][:3]
        rows.append({"time_utc": pd.to_datetime(p["time"], unit="ms", utc=True), "magnitude": p.get("mag"),
                     "mag_type": p.get("magType"), "depth_km": depth, "longitude": lon, "latitude": lat,
                     "place": p.get("place"), "felt": p.get("felt"), "alert": p.get("alert"),
                     "tsunami": p.get("tsunami"), "url": p.get("url"), "id": f.get("id")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    provinces = boundaries.locate(df.longitude.to_numpy(), df.latitude.to_numpy(), level=1)
    df["province"] = [p.name if p else None for p in provinces]
    df["in_nepal"] = df.province.notna()
    df["km_from_kathmandu"] = [round(haversine_km(KATHMANDU[0], KATHMANDU[1], lo, la)) for lo, la in
                               zip(df.longitude, df.latitude)]
    df["time_npt"] = (df.time_utc + NPT).dt.tz_localize(None)
    return df.sort_values("time_utc", ascending=False).reset_index(drop=True)


def recent(days: int = 30, min_magnitude: float = 2.5) -> pd.DataFrame:
    """Recent events from USGS's CDN-cached summary feeds (fast and robust); FDSN query otherwise."""
    if days > 30 or min_magnitude < 2.5:
        return query(datetime.now(timezone.utc) - timedelta(days=days), None, min_magnitude)
    period = "week" if days <= 7 else "month"
    feats = get_json(FEED.format(level="2.5", period=period), timeout=60).get("features", [])
    since = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    lon0, lat0, lon1, lat1 = SEARCH_BOX
    keep = [f for f in feats
            if f["properties"].get("time", 0) >= since and (f["properties"].get("mag") or 0) >= min_magnitude
            and lon0 <= f["geometry"]["coordinates"][0] <= lon1 and lat0 <= f["geometry"]["coordinates"][1] <= lat1]
    return _frame(keep)


def energy_joules(magnitude: pd.Series) -> pd.Series:
    """Radiated seismic energy, log10 E = 1.5 M + 4.8 (Gutenberg-Richter), for comparing years."""
    return np.power(10.0, 1.5 * magnitude.astype(float) + 4.8)
