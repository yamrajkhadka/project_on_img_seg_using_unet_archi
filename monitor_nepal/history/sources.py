"""Long-run national series for Nepal.

* World Bank WDI (1960-)         population, urbanisation, forest and farm land, ...
* HYDE 3.3 via Our World in Data  population, cropland, grazing land and built-up area back to 10,000 BCE
* World Bank CCKP climate         CRU TS 4.08 (1901-2023) and ERA5 (1950-2023) national averages

Every fetch falls back to a snapshot in ``data/history`` (refreshed by ``scripts/build_history.py``), so
the long-term pages keep working if an API is down.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd

from ..config import HISTORY_DIR
from ..http import FetchError, get, get_json

WB = "https://api.worldbank.org/v2/country/NPL/indicator/{code}"
OWID = "https://ourworldindata.org/grapher/{slug}.csv"
CCKP = "https://cckpapi.worldbank.org/cckp/v1/{dataset}/NPL"


@dataclass(frozen=True)
class Indicator:
    code: str
    name: str
    unit: str
    note: str = ""


WB_INDICATORS = {
    "population": Indicator("SP.POP.TOTL", "Population", "people"),
    "urban_share": Indicator("SP.URB.TOTL.IN.ZS", "Urban population", "% of total"),
    "forest_share": Indicator("AG.LND.FRST.ZS", "Forest area", "% of land area", "FAO Forest Resources Assessment"),
    "agri_share": Indicator("AG.LND.AGRI.ZS", "Agricultural land", "% of land area", "FAO"),
    "arable_share": Indicator("AG.LND.ARBL.ZS", "Arable land", "% of land area", "FAO"),
    "cereal_yield": Indicator("AG.YLD.CREL.KG", "Cereal yield", "kg per hectare", "FAO"),
    "density": Indicator("EN.POP.DNST", "Population density", "people per km²"),
    "electricity": Indicator("EG.ELC.ACCS.ZS", "Access to electricity", "% of population"),
    "protected": Indicator("ER.LND.PTLD.ZS", "Terrestrial protected areas", "% of land area"),
    "gdp_pc": Indicator("NY.GDP.PCAP.CD", "GDP per capita", "current US$"),
}

OWID_SERIES = {
    "population_long": ("population", {"population_historical": "population"}),
    "land_use_long": ("land-use-over-the-long-term", {"grazing_c": "grazing_ha", "cropland_c": "cropland_ha",
                                                      "uopp_c": "builtup_ha"}),
    "urban_rural_long": ("urban-and-rural-population-2050", {"urban_population": "urban",
                                                             "rural_population": "rural"}),
}

CLIMATE = {
    "cru_tas": ("cru-x0.5_timeseries_tas_timeseries_annual_1901-2023_mean_historical_cru_ts4.08_mean",
                "Mean air temperature (CRU TS 4.08)", "°C"),
    "cru_pr": ("cru-x0.5_timeseries_pr_timeseries_annual_1901-2023_mean_historical_cru_ts4.08_mean",
               "Precipitation (CRU TS 4.08)", "mm/year"),
    "era5_tas": ("era5-x0.25_timeseries_tas_timeseries_annual_1950-2023_mean_historical_era5_x0.25_mean",
                 "Mean air temperature (ERA5)", "°C"),
    "era5_pr": ("era5-x0.25_timeseries_pr_timeseries_annual_1950-2023_mean_historical_era5_x0.25_mean",
                "Precipitation (ERA5)", "mm/year"),
}


def _snapshot(name: str) -> pd.DataFrame:
    path = HISTORY_DIR / f"{name}.csv"
    if not path.exists():
        raise FetchError(f"No live data and no bundled snapshot for {name}.")
    df = pd.read_csv(path)
    df.attrs["snapshot"] = True
    return df


def world_bank(key: str) -> pd.DataFrame:
    """Columns: year, value."""
    ind = WB_INDICATORS[key]
    try:
        data = get_json(WB.format(code=ind.code), params={"format": "json", "per_page": 200})
        rows = [{"year": int(r["date"]), "value": r["value"]} for r in (data[1] or []) if r.get("value") is not None]
        df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
        if df.empty:
            raise FetchError("empty")
        return df
    except (FetchError, IndexError, KeyError, TypeError):
        return _snapshot(f"wb_{key}")


def owid(key: str) -> pd.DataFrame:
    slug, columns = OWID_SERIES[key]
    try:
        r = get(OWID.format(slug=slug), params={"country": "~NPL", "csvType": "filtered",
                                                "useColumnShortNames": "true"}, timeout=60)
        df = pd.read_csv(io.BytesIO(r.content))
        df = df[df["code"] == "NPL"].rename(columns=columns)
        return df[["year", *columns.values()]].sort_values("year").reset_index(drop=True)
    except (FetchError, KeyError, ValueError):
        return _snapshot(f"owid_{key}")


def climate(key: str) -> pd.DataFrame:
    """National annual series: columns year, value."""
    dataset = CLIMATE[key][0]
    try:
        data = get_json(CCKP.format(dataset=dataset), params={"_format": "json"}, timeout=60)
        series = data["data"]["NPL"]
        df = pd.DataFrame({"year": [int(k[:4]) for k in series], "value": list(series.values())})
        return df.sort_values("year").reset_index(drop=True)
    except (FetchError, KeyError, TypeError, ValueError):
        return _snapshot(f"climate_{key}")
