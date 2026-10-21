"""Reusable input widgets: regions, places and imagery/date selection."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import streamlit as st

from .. import boundaries
from ..boundaries import Region
from ..imagery import FIDELITY, KINDS, ImagerySpec, eox
from . import data


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    km: float  # suggested window size
    why: str


PLACES = (
    Place("Kathmandu core (Durbar Square → Thamel)", 27.7090, 85.3110, 3.0, "Dense historic city"),
    Place("Kathmandu Valley east (Bhaktapur)", 27.6722, 85.4280, 4.0, "Urban growth into farmland"),
    Place("Pokhara & Phewa Lake", 28.2100, 83.9640, 5.0, "Lake, city, forested hills"),
    Place("Nepalgunj (paper Fig. 8)", 28.0500, 81.6167, 4.0, "Terai city and fields used in the paper"),
    Place("Dharan (IoE Purwanchal Campus)", 26.8125, 87.2830, 4.0, "The authors' campus town"),
    Place("Biratnagar", 26.4550, 87.2750, 4.0, "Eastern Terai industrial city"),
    Place("Chitwan — Sauraha & Rapti River", 27.5780, 84.4950, 5.0, "National park edge, river, farms"),
    Place("Koshi River barrage", 26.5260, 86.9350, 5.0, "Braided river and floodplain"),
    Place("Lumbini", 27.4833, 83.2760, 4.0, "Birthplace of the Buddha, Terai farmland"),
    Place("Janakpur", 26.7288, 85.9263, 4.0, "Madhesh city and ponds"),
    Place("Melamchi Bazaar (2021 flood)", 27.8330, 85.5770, 3.0, "Debris-flow-hit valley"),
    Place("Rara Lake", 29.5260, 82.0870, 5.0, "Nepal's largest lake"),
    Place("Lo Manthang, Upper Mustang", 29.1830, 83.9580, 5.0, "Trans-Himalayan desert"),
    Place("Khumbu Glacier & Everest Base Camp", 27.9870, 86.8350, 5.0, "Glaciers, rock and snow"),
)


@st.cache_data(show_spinner=False)
def _region_options() -> list[str]:
    codes = [boundaries.country().code]
    for level in (1, 2, 3):
        codes += [r.code for r in boundaries.regions(level)]
    return codes


def _label(code: str) -> str:
    return boundaries.get(code).label


def region_picker(key: str, *, label: str = "Region", default: str = "NP", help: str | None = None,
                  max_level: int = 3) -> Region:
    options = [c for c in _region_options() if boundaries.get(c).level <= max_level]
    current = st.session_state.get(key, default)
    idx = options.index(current) if current in options else 0
    code = st.selectbox(label, options, index=idx, format_func=_label, key=key, filter_mode="fuzzy",
                        help=help or "Type to search any province, district or municipality.")
    return boundaries.get(code)


def imagery_picker(key: str, *, kinds: tuple[str, ...] = tuple(KINDS), default: str = "esri",
                   bbox: tuple[float, float, float, float] | None = None, compact: bool = False) -> ImagerySpec:
    kind = st.selectbox("Imagery", kinds, index=kinds.index(default) if default in kinds else 0,
                        format_func=KINDS.get, key=f"{key}_kind")
    level, why = FIDELITY[kind]
    color = {"Best": "green", "Good": "blue", "Experimental": "orange"}.get(level, "red")
    if not compact:
        st.badge(f"Model fit: {level}", color=color, help=why)
    if kind == "esri":
        return ImagerySpec("esri")
    if kind == "wayback":
        rels = data.wayback_releases()
        opts = [r.number for r in rels]
        labels = {r.number: r.date.strftime("%d %b %Y") for r in rels}
        num = st.selectbox("Wayback release", opts, index=len(opts) - 1, format_func=lambda n: labels[n],
                           key=f"{key}_release", help="Release date of the archived mosaic. The pixels may be older; "
                                                       "the capture date is shown after the analysis.")
        return ImagerySpec("wayback", release=num)
    if kind == "eox":
        year = st.select_slider("Year", options=list(eox.years()), value=eox.years()[-1], key=f"{key}_year")
        return ImagerySpec("eox", year=year)
    first = {"s2": date(2015, 7, 1), "landsat": date(1982, 9, 1), "mss": date(1972, 8, 1)}[kind]
    last = date.today() if kind != "mss" else date(1992, 12, 31)
    default_end = min(last, date.today())
    default_start = max(first, default_end - timedelta(days=90))
    if kind == "mss":
        default_start, default_end = date(1976, 10, 1), date(1977, 3, 31)
    rng = st.date_input("Date window", value=(default_start, default_end), min_value=first, max_value=last,
                        key=f"{key}_window", help="The least-cloudy scenes in this window are mosaicked.")
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        start, end = rng
    else:
        start = rng[0] if isinstance(rng, (tuple, list)) else rng
        end = start + timedelta(days=30)
    cloud = st.slider("Max scene cloud cover", 0, 100, 20 if kind == "s2" else 30, step=5, key=f"{key}_cloud",
                      format="%d%%")
    return ImagerySpec(kind, start=start, end=end, cloud_max=float(cloud))
