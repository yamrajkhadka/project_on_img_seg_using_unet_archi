"""A short, sourced-from-the-record timeline of Nepal since unification, for annotating long charts.

Categories: "state" (political turning points), "land" (policies and processes that changed land
cover), "disaster" and "conservation".  Kept deliberately brief and limited to well-documented events.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache

from ..config import HISTORY_DIR

AUTO_EVENTS = HISTORY_DIR / "events_auto.csv"  # written monthly by scripts/extend_years.py


@dataclass(frozen=True)
class Event:
    year: int
    title: str
    category: str
    detail: str
    end_year: int | None = None


EVENTS: tuple[Event, ...] = (
    Event(1768, "Unification: Kathmandu taken", "state",
          "Prithvi Narayan Shah of Gorkha takes Kathmandu during Indra Jatra; it becomes the capital of the "
          "unified kingdom."),
    Event(1814, "Anglo-Nepalese War", "state",
          "Ends with the Treaty of Sugauli (1815-16): Nepal loses Sikkim, Kumaon, Garhwal and much of the Terai.",
          1816),
    Event(1846, "Rana rule begins", "state", "After the Kot massacre Jung Bahadur Rana seizes power; "
          "hereditary Rana prime ministers rule for 104 years."),
    Event(1860, "Western Terai returned", "state",
          "Banke, Bardiya, Kailali and Kanchanpur (the 'Naya Muluk') are restored to Nepal."),
    Event(1923, "Sovereignty recognised", "state", "Nepal-Britain Treaty of Friendship."),
    Event(1934, "Nepal-Bihar earthquake", "disaster",
          "M ~8.0 on 15 January; thousands killed and much of Kathmandu Valley destroyed."),
    Event(1951, "Democracy; Rana rule ends", "state", "Nepal opens to the outside world."),
    Event(1953, "Everest first climbed", "conservation", "Tenzing Norgay and Edmund Hillary, 29 May."),
    Event(1955, "Terai opened to settlement", "land",
          "Malaria control and resettlement schemes (e.g. Rapti valley, Chitwan) bring large-scale clearing of "
          "Terai forest for farms."),
    Event(1957, "Private forests nationalised", "land", "Private Forests Nationalization Act."),
    Event(1960, "Panchayat era", "state", "Party-less Panchayat system under the king.", 1990),
    Event(1964, "Land reform", "land", "Lands Act sets ceilings on holdings and tenancy rights."),
    Event(1973, "First national park", "conservation", "Chitwan National Park established."),
    Event(1976, "Sagarmatha National Park", "conservation", "Everest region protected; UNESCO site in 1979."),
    Event(1978, "Community forestry begins", "land",
          "Panchayat forest rules let local communities manage forests."),
    Event(1988, "Udayapur earthquake", "disaster", "M 6.9 in eastern Nepal, 21 August."),
    Event(1990, "Multiparty democracy restored", "state", "People's Movement; constitutional monarchy."),
    Event(1993, "Forest Act 1993", "land",
          "Community forest user groups get legal rights - a key driver of Nepal's forest recovery."),
    Event(1996, "Armed conflict", "state", "Maoist insurgency; ends with the Comprehensive Peace Accord.", 2006),
    Event(2008, "Federal republic", "state", "Monarchy abolished on 28 May."),
    Event(2015, "Gorkha earthquake", "disaster",
          "Mw 7.8 on 25 April and Mw 7.3 on 12 May; about 9,000 deaths, tens of thousands of landslides."),
    Event(2015, "New constitution", "state", "Seven provinces; 753 local governments elected in 2017."),
    Event(2021, "Melamchi flood", "disaster", "Debris flood in Sindhupalchok, 15 June."),
    Event(2022, "Tiger numbers nearly triple", "conservation", "National survey counts 355 wild tigers (121 in 2010)."),
    Event(2023, "Jajarkot earthquake", "disaster", "M 5.7 on 3 November in Karnali Province."),
    Event(2024, "September floods", "disaster",
          "Extreme rain on 27-28 September; worst flooding in decades in Kathmandu Valley."),
)

CATEGORY_LABELS = {"state": "State & politics", "land": "Land policy", "disaster": "Disasters",
                   "conservation": "Conservation & milestones"}


def all_events() -> tuple[Event, ...]:
    """The curated timeline, continued by events recorded automatically (strong earthquakes, major floods)."""
    try:
        return EVENTS + _auto(AUTO_EVENTS.stat().st_mtime)
    except FileNotFoundError:
        return EVENTS


@lru_cache(maxsize=2)
def _auto(_mtime: float) -> tuple[Event, ...]:
    with AUTO_EVENTS.open() as f:
        return tuple(Event(int(r["year"]), r["title"], r["category"], r["detail"]) for r in csv.DictReader(f))


def between(start: int, end: int, categories: tuple[str, ...] | None = None) -> list[Event]:
    return [e for e in all_events() if start <= e.year <= end and (categories is None or e.category in categories)]
