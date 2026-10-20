"""Alert rules on synthetic live data."""
from __future__ import annotations

import pandas as pd

from monitor_nepal.live import alerts


def _now():
    return pd.Timestamp.now(tz="UTC")


def test_all_clear():
    out = alerts.compute(None, None, None, None)
    assert [a.status for a in out] == ["good"]


def test_rules_fire_and_sort():
    t = _now()
    fires = pd.DataFrame({"time_utc": [t - pd.Timedelta(hours=2)] * 30 + [t - pd.Timedelta(days=3)] * 6,
                          "province": ["Koshi"] * 36})
    quakes = pd.DataFrame({"time_utc": [t - pd.Timedelta(days=1)], "magnitude": [5.8], "place": ["Test"],
                           "time_npt": [pd.Timestamp("2026-10-19 10:00")], "km_from_kathmandu": [120]})
    now = pd.DataFrame({"place": ["A", "B"], "us_aqi": [180.0, 120.0]})
    rain = pd.DataFrame({"place": ["C"], "date": ["2026-10-21"], "rain_mm": [130.0], "probability": [90]})
    out = alerts.compute(fires, quakes, now, rain)
    titles = [a.title for a in out]
    assert out[0].status == "critical"
    assert "Fire activity above normal" in titles
    assert "Very heavy rain forecast" in titles
    assert any(t.startswith("M5.8") for t in titles)
    assert [a.status for a in out] == sorted([a.status for a in out],
                                             key={"critical": 0, "serious": 1, "warning": 2, "good": 3}.get)


def test_river_status_and_flood_watch():
    from monitor_nepal.live import rivers

    assert rivers.status(500, 8000, 10000, 1200)[1] == "good"
    assert rivers.status(1500, 8000, 10000, 1200)[1] == "warning"
    assert rivers.status(8500, 8000, 10000, 1200)[1] == "serious"
    assert rivers.status(11000, 8000, 10000, 1200)[1] == "critical"
    today = pd.Timestamp.now(tz="Asia/Kathmandu").normalize().tz_localize(None)
    recent = pd.DataFrame({"river": ["koshi"] * 3, "date": [today - pd.Timedelta(days=20), today,
                                                             today + pd.Timedelta(days=2)],
                           "discharge": [12000.0, 3000.0, 8500.0], "forecast": [False, False, True]})
    thr = pd.DataFrame({"river": ["koshi"], "q2": [8000.0], "q10": [10000.0], "mean": [1800.0], "years": [30]})
    out = alerts.rivers(recent, thr, {"koshi": "Koshi"})
    assert len(out) == 1 and out[0].status == "serious" and "forecast" in out[0].detail  # old 12,000 ignored
    merged = alerts.merge(alerts.compute(None, None, None, None), out)
    assert [a.status for a in merged] == ["serious"]
