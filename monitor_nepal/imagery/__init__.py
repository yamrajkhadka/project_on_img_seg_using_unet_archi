"""Imagery sources behind one hashable :class:`ImagerySpec`."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from ..geo import Grid
from . import eox, esri, planetary
from .base import ProgressFn, Scene, XYZSource

KINDS = {
    "esri": "Esri World Imagery · latest (≈0.5 m)",
    "wayback": "Esri Wayback · historical high-res (2014–)",
    "eox": "Sentinel-2 cloudless · annual mosaic (2016–2025)",
    "s2": "Sentinel-2 L2A · any date (2015–, 10 m)",
    "landsat": "Landsat 4–9 · any date (1982–, 30 m)",
    "mss": "Landsat MSS · false colour (1972–1992, 60 m)",
}

# How far each source is from the model's training distribution (0.5 m DeepGlobe resized to 5.5 m).
FIDELITY = {
    "esri": ("Best", "Same kind of sub-metre imagery the network was trained on."),
    "wayback": ("Best", "Same kind of sub-metre imagery the network was trained on."),
    "eox": ("Good", "10 m Sentinel-2, upsampled to the model scale: coarse textures, consistent colour."),
    "s2": ("Good", "10 m Sentinel-2, upsampled to the model scale; clouds may remain."),
    "landsat": ("Experimental", "30 m pixels upsampled ~5x: only large patterns are meaningful."),
    "mss": ("Indicative only", "No blue band; shown as false colour. The RGB model is not reliable here."),
}


@dataclass(frozen=True)
class ImagerySpec:
    kind: str
    year: int | None = None  # eox
    release: int | None = None  # wayback
    start: date | None = None  # s2 / landsat / mss window
    end: date | None = None
    cloud_max: float | None = 30
    item: str | None = None  # a single scene id

    def label(self) -> str:
        if self.kind == "eox":
            return f"Sentinel-2 cloudless {self.year}"
        if self.kind == "wayback":
            rel = next((r for r in esri.wayback_releases() if r.number == self.release), None)
            return f"Esri Wayback {rel.date.isoformat() if rel else self.release}"
        if self.kind in ("s2", "landsat", "mss"):
            name = planetary.PRODUCTS[self.kind].name
            if self.item:
                return f"{name} scene"
            return f"{name} {self.start:%b %Y}–{self.end:%b %Y}"
        return KINDS[self.kind]

    def short_label(self) -> str:
        """Compact name for tiles and captions."""
        if self.kind == "esri":
            return "Esri latest"
        if self.kind == "wayback":
            rel = next((r for r in esri.wayback_releases() if r.number == self.release), None)
            return f"Wayback {rel.date:%b %Y}" if rel else "Wayback"
        if self.kind == "eox":
            return f"S2 cloudless {self.year}"
        return {"s2": "Sentinel-2", "landsat": "Landsat", "mss": "Landsat MSS"}[self.kind]

    def with_window(self, start: date, end: date) -> "ImagerySpec":
        return replace(self, start=start, end=end)


def resolve(spec: ImagerySpec, bbox_lonlat: tuple[float, float, float, float]) -> XYZSource:
    if spec.kind == "esri":
        return esri.current_source()
    if spec.kind == "wayback":
        rel = next((r for r in esri.wayback_releases() if r.number == spec.release), None)
        if rel is None:
            raise ValueError(f"Unknown Wayback release {spec.release}")
        return esri.wayback_source(rel)
    if spec.kind == "eox":
        return eox.annual_source(int(spec.year or eox.years()[-1]))
    if spec.kind in planetary.PRODUCTS:
        if spec.item:
            return planetary.item_source(spec.kind, spec.item, spec.start.isoformat() if spec.start else "")
        if not (spec.start and spec.end):
            raise ValueError("Sentinel-2/Landsat need a date window.")
        return planetary.mosaic_source(spec.kind, bbox_lonlat, spec.start, spec.end, spec.cloud_max)
    raise ValueError(f"Unknown imagery kind {spec.kind!r}")


def fetch(spec: ImagerySpec, grid: Grid, progress: ProgressFn | None = None, for_model: bool = False) -> Scene:
    return resolve(spec, grid.bounds_lonlat).fetch(grid, progress=progress, for_model=for_model)


__all__ = ["ImagerySpec", "KINDS", "FIDELITY", "Scene", "XYZSource", "resolve", "fetch"]
