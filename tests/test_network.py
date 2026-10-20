"""Live-service tests (opt-in): MONITOR_NEPAL_NETWORK_TESTS=1 pytest -q tests/test_network.py"""
from __future__ import annotations

import os
from datetime import date

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("MONITOR_NEPAL_NETWORK_TESTS") != "1",
                                reason="set MONITOR_NEPAL_NETWORK_TESTS=1 to hit the live services")


def test_imagery_sources():
    from monitor_nepal import imagery
    from monitor_nepal.config import MODEL_GSD_M
    from monitor_nepal.geo import Grid

    g = Grid.around(85.324, 27.7172, 600, MODEL_GSD_M)
    for spec in (imagery.ImagerySpec("esri"), imagery.ImagerySpec("eox", year=2024),
                 imagery.ImagerySpec("s2", start=date(2024, 10, 1), end=date(2024, 12, 31), cloud_max=20)):
        sc = imagery.fetch(spec, g, for_model=spec.kind == "esri")
        assert sc.rgb.shape == (g.height, g.width, 3) and sc.valid_fraction > 0.9


def test_live_feeds():
    from monitor_nepal.live import quakes, weather

    assert len(weather.current()) >= 10
    assert quakes.recent(30, 2.5) is not None


def test_reference_maps():
    from monitor_nepal.config import MODEL_GSD_M
    from monitor_nepal.geo import Grid
    from monitor_nepal.imagery import reference

    g = Grid.around(85.324, 27.7172, 600, MODEL_GSD_M)
    io_cls, dg = reference.fetch_io(g, 2023)
    assert dg.shape == g.shape and (dg == 0).mean() > 0.5  # central Kathmandu is built-up
    wc = reference.fetch_worldcover(g, 2021)
    assert wc.shape == g.shape


def test_fire_archive_and_history():
    from monitor_nepal.history import sources
    from monitor_nepal.live import fires

    df = fires.archive(date.today().year - 2)
    assert len(df) > 1000 and df.district.notna().all()
    assert sources.world_bank("population").year.max() >= date.today().year - 3
