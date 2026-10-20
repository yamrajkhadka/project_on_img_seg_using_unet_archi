"""Project-wide constants: land-cover classes, palettes, paths and Nepal facts."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
ATLAS_DIR = DATA_DIR / "atlas"
HISTORY_DIR = DATA_DIR / "history"

ONNX_MODEL_PATH = MODELS_DIR / "unet_deepglobe.onnx"
KERAS_MODEL_PATH = ROOT / "deepglobe_unet_jay.keras"

USER_AGENT = "MonitorNepal/1.0 (+https://github.com/yamrajkhadka/project_on_img_seg_using_unet_archi)"
HTTP_TIMEOUT = float(os.environ.get("MONITOR_NEPAL_HTTP_TIMEOUT", "30"))
# Optional on-disk cache for imagery tiles (used by the batch scripts; the app keeps tiles in memory).
DISK_CACHE_DIR = os.environ.get("MONITOR_NEPAL_CACHE")

# DeepGlobe tiles are 2448 px at 0.5 m and the paper resizes them to 224 px, so the network
# learned to see the world at 2448 * 0.5 / 224 = 5.464 m per pixel.  Feeding imagery at that
# ground sample distance (GSD) is the single most important thing for good predictions.
MODEL_GSD_M = 2448 * 0.5 / 224
MODEL_TILE = 224

NEPAL_BBOX = (80.05, 26.34, 88.21, 30.45)  # lon_min, lat_min, lon_max, lat_max
NEPAL_CENTER = (84.1, 28.3)
NEPAL_TZ = "Asia/Kathmandu"
NEPAL_UNIFICATION_YEAR = 1768  # Kathmandu taken by Prithvi Narayan Shah; capital of the unified kingdom


@dataclass(frozen=True)
class LandCoverClass:
    id: int
    key: str
    name: str
    paper_rgb: tuple[int, int, int]  # DeepGlobe colour, as published in the paper (Table 2)
    light: str  # chart/map colour validated for light surfaces (CVD-safe in stack order)
    dark: str  # same hue stepped for dark surfaces
    definition: str  # DeepGlobe class definition (Demir et al., 2018)


CLASSES: tuple[LandCoverClass, ...] = (
    LandCoverClass(0, "urban", "Urban land", (0, 255, 255), "#1497b0", "#19a5bf",
                   "Man-made, built-up areas with human artifacts."),
    LandCoverClass(1, "agriculture", "Agriculture land", (255, 255, 0), "#eda100", "#c98500",
                   "Farms, planned plantation, cropland, orchards, vineyards, nurseries and horticulture."),
    LandCoverClass(2, "rangeland", "Rangeland", (255, 0, 255), "#c450b5", "#d55cc6",
                   "Any non-forest, non-farm green land: grass and shrubs."),
    LandCoverClass(3, "forest", "Forest land", (0, 255, 0), "#008300", "#1f9a1f",
                   "Land with tree-crown cover, including clear-cuts."),
    LandCoverClass(4, "water", "Water", (0, 0, 255), "#2a78d6", "#3987e5",
                   "Rivers, lakes, wetlands and ponds."),
    LandCoverClass(5, "barren", "Barren land", (255, 255, 255), "#b5712b", "#c07a35",
                   "Mountain, rock, sand, river bars and other land without vegetation."),
    LandCoverClass(6, "unknown", "Unknown", (0, 0, 0), "#898781", "#898781",
                   "Clouds and anything the network cannot place."),
)
NUM_CLASSES = len(CLASSES)
CLASS_KEYS = [c.key for c in CLASSES]
CLASS_NAMES = [c.name for c in CLASSES]
CLASS_BY_KEY = {c.key: c for c in CLASSES}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


# Map palettes.  "cartographic" uses the same hues as the paper but is stepped so that it reads on
# satellite imagery and matches the charts; "paper" reproduces the DeepGlobe colours exactly.
PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "cartographic": [hex_to_rgb(c.light) for c in CLASSES],
    "paper": [c.paper_rgb for c in CLASSES],
}
PALETTE_LABELS = {"cartographic": "Cartographic (matches charts)", "paper": "DeepGlobe (as in the paper)"}


def chart_colors(dark: bool = False) -> list[str]:
    return [c.dark if dark else c.light for c in CLASSES]


@dataclass(frozen=True)
class Paper:
    title: str = ("Land Cover Segmentation from Satellite Imagery Using U-Net with Custom Loss "
                  "and Morphological Postprocessing")
    authors: str = "Yamraj Khadka and Rajat Pradhan"
    short_authors: str = "Khadka & Pradhan"
    affiliation: str = "IoE Purwanchal Campus, Dharan, Nepal"
    journal: str = "Journal of Engineering Issues and Solutions"
    volume: str = "5"
    issue: str = "2"
    pages: str = "240-253"
    year: int = 2026
    doi: str = "10.3126/joeis.v5i2.97825"
    url: str = "https://www.nepjol.info/index.php/joeis/article/view/97825"
    pdf_url: str = "https://www.nepjol.info/index.php/joeis/article/view/97825/74255"
    demo_url: str = "https://4classifying-every-pixel-of-the-planets.streamlit.app/"
    code_url: str = "https://github.com/yamrajkhadka/project_on_img_seg_using_unet_archi"

    @property
    def citation(self) -> str:
        return (f"{self.authors.replace(' and ', ' & ')} ({self.year}). {self.title}. "
                f"{self.journal}, {self.volume}({self.issue}), {self.pages}. https://doi.org/{self.doi}")

    @property
    def bibtex(self) -> str:
        return (
            "@article{khadka2026landcover,\n"
            f"  title   = {{{self.title}}},\n"
            f"  author  = {{Khadka, Yamraj and Pradhan, Rajat}},\n"
            f"  journal = {{{self.journal}}},\n"
            f"  volume  = {{{self.volume}}},\n"
            f"  number  = {{{self.issue}}},\n"
            f"  pages   = {{{self.pages.replace('-', '--')}}},\n"
            f"  year    = {{{self.year}}},\n"
            f"  doi     = {{{self.doi}}}\n"
            "}"
        )


PAPER = Paper()

# Per-class IoU reported in the paper (Table 3).
PAPER_IOU = {"urban": 0.7187, "agriculture": 0.7170, "rangeland": 0.3525, "forest": 0.7216,
             "water": 0.6582, "barren": 0.5681, "unknown": 0.9826}
PAPER_MEAN_IOU = 0.6741
