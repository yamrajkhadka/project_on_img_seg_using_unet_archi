"""Simple, transparent alert rules over the live feeds."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# 24-hour rainfall categories used across South Asia (India Meteorological Department).
HEAVY_RAIN_MM, VERY_HEAVY_RAIN_MM = 64.5, 115.6


@dataclass(frozen=True)
class Alert:
    status: str  # good | warning | serious | critical
    title: str
    detail: str


def compute(fires7: pd.DataFrame | None, quakes30: pd.DataFrame | None, now: pd.DataFrame | None,
            rain: pd.DataFrame | None) -> list[Alert]:
    alerts: list[Alert] = []
    t_now = pd.Timestamp.now(tz="UTC")

    if now is not None and len(now) and "us_aqi" in now:
        bad = now[now.us_aqi > 150].sort_values("us_aqi", ascending=False)
        sensitive = now[(now.us_aqi > 100) & (now.us_aqi <= 150)].sort_values("us_aqi", ascending=False)
        if len(bad):
            alerts.append(Alert("critical", "Unhealthy air",
                                ", ".join(f"{r.place} {r.us_aqi:.0f}" for r in bad.itertuples()) + " (US AQI)"))
        if len(sensitive):
            alerts.append(Alert("serious", "Air unhealthy for sensitive groups",
                                ", ".join(f"{r.place} {r.us_aqi:.0f}" for r in sensitive.itertuples())))

    if fires7 is not None and len(fires7):
        last = fires7[fires7.time_utc >= t_now - pd.Timedelta(hours=24)]
        earlier = fires7[fires7.time_utc < t_now - pd.Timedelta(hours=24)]
        per_day = len(earlier) / 6 if len(earlier) else 0.0
        if len(last) >= 20 and len(last) > 2 * max(per_day, 1.0):
            top = last.province.value_counts().idxmax()
            alerts.append(Alert("warning", "Fire activity above normal",
                                f"{len(last)} detections in 24 h vs {per_day:.0f}/day before; most in {top}"))

    if quakes30 is not None and len(quakes30):
        week = quakes30[quakes30.time_utc >= t_now - pd.Timedelta(days=7)]
        strong = week[week.magnitude >= 4.5]
        if len(strong):
            q = strong.loc[strong.magnitude.idxmax()]
            alerts.append(Alert("critical" if q.magnitude >= 5.5 else "warning", f"M{q.magnitude:.1f} earthquake",
                                f"{q.place} · {q.time_npt:%d %b %H:%M} NPT · {q.km_from_kathmandu} km from Kathmandu"))

    if rain is not None and len(rain):
        heavy = rain[rain.rain_mm >= HEAVY_RAIN_MM].sort_values("rain_mm", ascending=False)
        if len(heavy):
            very = heavy.rain_mm.max() >= VERY_HEAVY_RAIN_MM
            alerts.append(Alert("serious" if very else "warning", "Very heavy rain forecast" if very else
                                "Heavy rain forecast",
                                ", ".join(f"{r.place} {r.rain_mm:.0f} mm ({pd.Timestamp(r.date):%a})"
                                          for r in heavy.head(4).itertuples())))

    if not alerts:
        alerts.append(Alert("good", "All clear", "Air quality, fires, earthquakes and the rain forecast are within "
                                                 "normal ranges."))
    return sort_alerts(alerts)


def rivers(recent: pd.DataFrame | None, thresholds: pd.DataFrame | None, names: dict[str, str]) -> list[Alert]:
    """Flood watch when any river's flow (now or forecast) reaches its 2- or 10-year flood level."""
    if recent is None or thresholds is None or recent.empty:
        return []
    out = []
    thr = thresholds.set_index("river")
    since = pd.Timestamp.now(tz="Asia/Kathmandu").normalize().tz_localize(None) - pd.Timedelta(days=1)
    for key, g in recent[recent.date >= since].groupby("river"):
        if key not in thr.index:
            continue
        peak = g.loc[g.discharge.idxmax()]
        q2, q10 = thr.loc[key, "q2"], thr.loc[key, "q10"]
        if peak.discharge >= q2:
            when = "forecast" if peak.forecast else "observed"
            out.append(Alert("critical" if peak.discharge >= q10 else "serious",
                             f"Flood watch: {names.get(key, key)}",
                             f"{peak.discharge:,.0f} m³/s {when} {pd.Timestamp(peak.date):%d %b} "
                             f"(2-year flood level {q2:,.0f})"))
    return out


def merge(general: list[Alert], extra: list[Alert]) -> list[Alert]:
    """Combine rule sets; an 'all clear' item is dropped as soon as anything else is raised."""
    items = general + extra
    if any(a.status != "good" for a in items):
        items = [a for a in items if a.status != "good"]
    return sort_alerts(items)


def sort_alerts(alerts: list[Alert]) -> list[Alert]:
    order = {"critical": 0, "serious": 1, "warning": 2, "good": 3}
    return sorted(alerts, key=lambda a: order[a.status])
