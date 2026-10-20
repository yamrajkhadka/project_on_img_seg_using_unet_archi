"""Active fires from NASA FIRMS: near-real-time feeds (last 24 h / 48 h / 7 days) and the yearly archive.

FIRMS publishes open CSVs per region (no key needed).  We keep detections that fall inside Nepal and
tag each with its district and province.
"""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .. import boundaries
from ..config import NEPAL_BBOX
from ..http import FetchError, get

LIVE = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/{folder}/csv/{prefix}_South_Asia_{window}.csv"
ARCHIVE = "https://firms.modaps.eosdis.nasa.gov/data/country/{product}/{year}/{product}_{year}_Nepal.csv"
ATTRIBUTION = "NASA FIRMS (MODIS C6.1, VIIRS S-NPP / NOAA-20 / NOAA-21 375 m)"

SENSORS = {
    "VIIRS S-NPP": ("suomi-npp-viirs-c2", "SUOMI_VIIRS_C2"),
    "VIIRS NOAA-20": ("noaa-20-viirs-c2", "J1_VIIRS_C2"),
    "VIIRS NOAA-21": ("noaa-21-viirs-c2", "J2_VIIRS_C2"),
    "MODIS": ("modis-c6.1", "MODIS_C6_1"),
}
ARCHIVE_PRODUCTS = {"MODIS": ("modis", 2000), "VIIRS S-NPP": ("viirs-snpp", 2012)}


def _clean(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    if df.empty:
        return df
    lon0, lat0, lon1, lat1 = NEPAL_BBOX
    df = df[(df.longitude.between(lon0, lon1)) & (df.latitude.between(lat0, lat1))].copy()
    if df.empty:
        return df
    districts = boundaries.locate(df.longitude.to_numpy(), df.latitude.to_numpy(), level=2)
    df["district"] = [d.name if d else None for d in districts]
    df["province"] = [d.province if d else None for d in districts]
    df = df[df.district.notna()].copy()
    hhmm = df.acq_time.astype(str).str.zfill(4)
    df["time_utc"] = pd.to_datetime(df.acq_date.astype(str) + " " + hhmm.str[:2] + ":" + hhmm.str[2:], utc=True)
    df["sensor"] = sensor
    conf = df["confidence"].astype(str).str.lower()
    # VIIRS reports l/n/h, MODIS a 0-100 score
    df["confidence_class"] = np.select(
        [conf.isin(["h", "high"]), conf.isin(["n", "nominal"]), conf.isin(["l", "low"])],
        ["high", "nominal", "low"],
        default=pd.cut(pd.to_numeric(conf, errors="coerce"), [-1, 30, 80, 101],
                       labels=["low", "nominal", "high"]).astype(str))
    return df[["latitude", "longitude", "time_utc", "frp", "confidence_class", "daynight", "sensor", "district",
               "province"]].reset_index(drop=True)


def live(window: str = "24h", sensors: tuple[str, ...] = ("VIIRS S-NPP", "VIIRS NOAA-20", "VIIRS NOAA-21")) -> pd.DataFrame:
    """Detections inside Nepal for ``window`` in {24h, 48h, 7d}."""
    from concurrent.futures import ThreadPoolExecutor

    def one(s: str):
        folder, prefix = SENSORS[s]
        try:
            r = get(LIVE.format(folder=folder, prefix=prefix, window=window), timeout=45)
            return _clean(pd.read_csv(io.BytesIO(r.content)), s), None
        except (FetchError, pd.errors.ParserError) as exc:
            return None, str(exc)

    frames, errors = [], []
    with ThreadPoolExecutor(max_workers=len(sensors)) as ex:
        for df, err in ex.map(one, sensors):
            if df is not None:
                frames.append(df)
            if err:
                errors.append(err)
    if not frames and errors:
        raise FetchError(errors[0])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def archive(year: int, product: str = "VIIRS S-NPP") -> pd.DataFrame:
    """A full year of detections inside Nepal (FIRMS publishes the archive after the year ends)."""
    code, first = ARCHIVE_PRODUCTS[product]
    if year < first:
        raise FetchError(f"{product} fire records start in {first}.")
    r = get(ARCHIVE.format(product=code, year=year), timeout=90)
    return _clean(pd.read_csv(io.BytesIO(r.content)), product)


API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{day}"


def api_range(map_key: str, start: date, days: int = 10, source: str = "VIIRS_SNPP_SP") -> pd.DataFrame:
    """Any date range via the FIRMS area API (needs a free MAP_KEY from firms.modaps.eosdis.nasa.gov/api)."""
    lon0, lat0, lon1, lat1 = NEPAL_BBOX
    frames = []
    day = start
    while days > 0:
        n = min(10, days)
        r = get(API.format(key=map_key, source=source, bbox=f"{lon0},{lat0},{lon1},{lat1}", days=n,
                           day=day.isoformat()), timeout=90)
        if r.content.strip() and not r.content.startswith(b"Invalid"):
            frames.append(pd.read_csv(io.BytesIO(r.content)))
        day += timedelta(days=n)
        days -= n
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _clean(df, "VIIRS S-NPP") if not df.empty else df


def latest_archive_year(today: date | None = None) -> int:
    today = today or datetime.now(timezone.utc).date()
    return today.year - 1


def daily_counts(df: pd.DataFrame, start: date, end: date) -> pd.Series:
    """Detections per local (Nepal) calendar day, zero-filled."""
    idx = pd.date_range(start, end, freq="D").date
    if df.empty:
        return pd.Series(0, index=idx)
    local = (df.time_utc + timedelta(hours=5, minutes=45)).dt.date
    return local.value_counts().reindex(idx, fill_value=0).sort_index()
