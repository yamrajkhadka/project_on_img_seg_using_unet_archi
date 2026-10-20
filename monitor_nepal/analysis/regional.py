"""Helpers to restrict point datasets (fires, earthquakes) to a region."""
from __future__ import annotations

import numpy as np
import pandas as pd
from shapely import contains_xy

from .. import boundaries
from ..boundaries import Region


def in_region(df: pd.DataFrame, region: Region, *, lon: str = "longitude", lat: str = "latitude") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    if region.level == 0:
        if "district" in df:
            return df["district"].notna()
        return pd.Series(contains_xy(boundaries.geometry(region), df[lon].to_numpy(), df[lat].to_numpy()),
                         index=df.index)
    if region.level == 1 and "province" in df:
        return df["province"] == region.name
    if region.level == 2 and "district" in df:
        return df["district"] == region.name
    return pd.Series(contains_xy(boundaries.geometry(region), df[lon].to_numpy(), df[lat].to_numpy()), index=df.index)


def bbox(region: Region, margin_deg: float = 0.0) -> tuple[float, float, float, float]:
    lon0, lat0, lon1, lat1 = boundaries.geometry(region).bounds
    return lon0 - margin_deg, lat0 - margin_deg, lon1 + margin_deg, lat1 + margin_deg


def near_region(df: pd.DataFrame, region: Region, km: float = 50) -> pd.Series:
    """Points inside the region's bounding box grown by ``km`` (for earthquakes felt in the region)."""
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    m = km / 111.0
    lon0, lat0, lon1, lat1 = bbox(region, m)
    return df.longitude.between(lon0, lon1) & df.latitude.between(lat0, lat1)


def region_grid_size_km(region: Region) -> tuple[float, float]:
    lon0, lat0, lon1, lat1 = boundaries.geometry(region).bounds
    return (lon1 - lon0) * 111.32 * np.cos(np.radians((lat0 + lat1) / 2)), (lat1 - lat0) * 110.57
