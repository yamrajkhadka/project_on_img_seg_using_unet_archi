"""Live Nepal: what is happening right now, refreshed every 10 minutes."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from monitor_nepal.analysis import atlas
from monitor_nepal.config import CLASS_BY_KEY, CLASSES, NEPAL_BBOX
from monitor_nepal.http import FetchError
from monitor_nepal.live import alerts, weather
from monitor_nepal.live import fires as fires_mod
from monitor_nepal.live import rivers as rivers_mod
from monitor_nepal.live.rivers import RIVERS
from monitor_nepal.model import MODELS
from monitor_nepal.ui import charts, common, data, maps

common.header(
    "Nepal, right now",
    "Active fires, earthquakes, weather and air quality across the country, the newest satellite passes, and "
    "the national land-cover estimate from the U-Net. Everything on this page refreshes automatically.",
    eyebrow=f"Live · {common.now_npt():%A %d %B %Y, %H:%M} Nepal time",
)

with st.container(horizontal=True, vertical_alignment="center", gap="small"):
    st.caption("Came here from the paper? The original segmentation demo lives on, upgraded, in the Model lab.")
    st.page_link("views/model_lab.py", label="Open the Model lab", icon=":material/science:")


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except (FetchError, ValueError, KeyError) as exc:
        return None, str(exc)


@st.fragment(run_every="10m")
def live_panel() -> None:
    start, end = data.recent_days(4)
    got = data.parallel(
        {
            "fires": (data.fires_live, "7d"),
            "quakes": (data.quakes_recent, 30, 2.5),
            "now": (data.weather_now,),
            "s2": (data.scenes, "s2", NEPAL_BBOX, start, end, None),
            "ktm": (data.last_24h, 27.7172, 85.3240),
            "rain": (data.rain_forecast,),
            "rivers": (data.rivers_recent,),
        }
    )
    fires7, e1 = got["fires"]
    quakes30, e2 = got["quakes"]
    now, e3 = got["now"]
    s2, e4 = got["s2"]
    ktm_hist, _ = got["ktm"]
    rain, _ = got["rain"]

    # --- alerts ---------------------------------------------------------------------------------------------
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        ref = data.rivers_reference()
        flows, _ = got["rivers"]
        items = alerts.merge(alerts.compute(fires7, quakes30, now, rain),
                             alerts.rivers(flows, ref[1] if ref else None, {r.key: r.name for r in RIVERS}))
        for a in items:
            with st.container(border=True, width="content"):
                common.status_badge(a.title, a.status)
                st.caption(a.detail)

    # --- KPI row: every tile carries a sparkline so the row reads as one system ------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        if fires7 is not None:
            days = fires_mod.daily_counts(fires7, date.today() - timedelta(days=6), date.today())
            last24 = int((fires7.time_utc >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)).sum())
            st.metric(
                "Fires · 24 h",
                f"{last24:,}",
                delta=f"{last24 - days.mean():+.0f} vs 7-day mean",
                delta_color="inverse",
                border=True,
                chart_data=list(days.values),
                chart_type="bar",
                help="VIIRS detections (S-NPP, NOAA-20, NOAA-21) inside Nepal. Bars: detections per day, last 7 days.",
            )
        else:
            st.metric("Fires · 24 h", "—", border=True, help=e1)
    with k2:
        if quakes30 is not None:
            q7 = quakes30[quakes30.time_utc >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)]
            per_day = quakes30.time_npt.dt.date.value_counts().reindex(
                pd.date_range(date.today() - timedelta(days=29), date.today()).date, fill_value=0
            )
            strongest = f"max M{q7.magnitude.max():.1f}" if len(q7) else "none ≥ M2.5"
            st.metric(
                "Quakes · 7 days",
                f"{len(q7)}",
                delta=strongest,
                delta_color="off",
                delta_arrow="off",
                border=True,
                chart_data=list(per_day.values),
                chart_type="bar",
                help="USGS catalogue, magnitude ≥ 2.5, Nepal plus a ~100 km margin. Bars: events per day, "
                "last 30 days.",
            )
        else:
            st.metric("Quakes · 7 days", "—", border=True, help=e2)
    ktm = now[now.place == "Kathmandu"].iloc[0] if now is not None and len(now) else None
    with k3:
        if ktm is not None and pd.notna(ktm.us_aqi):
            label, status = weather.aqi_band(ktm.us_aqi, short=True)
            full, _ = weather.aqi_band(ktm.us_aqi)
            spark = ktm_hist.us_aqi.dropna().tolist() if ktm_hist is not None and "us_aqi" in ktm_hist else None
            st.metric(
                "Kathmandu AQI",
                f"{ktm.us_aqi:.0f}",
                delta=label,
                delta_color="off",
                delta_arrow="off",
                border=True,
                chart_data=spark or None,
                chart_type="line",
                help=f"US EPA air-quality index from CAMS: {full}. PM2.5 {ktm.pm2_5:.0f} µg/m³. Line: last 24 hours.",
            )
        else:
            st.metric("Kathmandu AQI", "—", border=True, help=e3)
    with k4:
        if ktm is not None:
            spark = ktm_hist.temperature_2m.dropna().tolist() if ktm_hist is not None else None
            st.metric(
                "Kathmandu now",
                f"{ktm.temperature_c:.0f} °C",
                delta=ktm.condition,
                delta_color="off",
                delta_arrow="off",
                border=True,
                chart_data=spark or None,
                chart_type="line",
                help=f"Feels like {ktm.feels_like_c:.0f} °C, humidity {ktm.humidity_pct:.0f}%, "
                f"wind {ktm.wind_kmh:.0f} km/h. Line: last 24 hours.",
            )
        else:
            st.metric("Kathmandu now", "—", border=True)
    with k5:
        if s2:
            last = max(s.datetime for s in s2)
            per_day = pd.Series([s.day for s in s2]).value_counts().sort_index()
            st.metric(
                "Newest Sentinel-2",
                f"{pd.Timestamp(last):%d %b}",
                delta=common.ago(pd.Timestamp(last)),
                delta_color="off",
                delta_arrow="off",
                border=True,
                chart_data=list(per_day.values),
                chart_type="bar",
                help=f"{len(s2)} scenes over Nepal in the last {len(per_day)} days (Microsoft Planetary "
                "Computer). Bars: scenes per day.",
            )
        else:
            st.metric("Newest Sentinel-2", "—", border=True, help=e4)

    # --- map ---------------------------------------------------------------------------------------------
    left, right = st.columns([2.2, 1], gap="medium")
    with left:
        window = st.segmented_control(
            "Fires shown",
            ["24h", "7d"],
            default="24h",
            key="live_fire_window",
            format_func={"24h": "Last 24 hours", "7d": "Last 7 days"}.get,
        )
        m = maps.base_map(nasa_day=maps.yesterday())
        maps.add_boundaries(m, 1, name="Provinces")
        maps.add_boundaries(m, 2, name="Districts", weight=0.5, show=False)
        if fires7 is not None and len(fires7):
            shown = (
                fires7
                if window == "7d"
                else fires7[fires7.time_utc >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)]
            )
            maps.add_fires(m, shown, name="Active fires")
        if quakes30 is not None:
            maps.add_quakes(m, quakes30, name="Earthquakes, 30 days")
        if now is not None:
            maps.add_places(m, now)
        st_folium(maps.finish(m), height=540, use_container_width=True, returned_objects=[], key="live_map")
        st.caption(
            "Layers (top right): today's NASA true-colour image, Sentinel-2 2025 mosaic, districts. "
            "Fire dots scale with radiative power; earthquake circles with magnitude."
        )
    with right:
        if fires7 is not None and len(fires7):
            sub = (
                fires7
                if window == "7d"
                else fires7[fires7.time_utc >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)]
            )
            by_prov = (
                sub.groupby("province")
                .size()
                .reindex(
                    ["Koshi", "Madhesh", "Bagmati", "Gandaki", "Lumbini", "Karnali", "Sudur Paschim"], fill_value=0
                )
            )
            fig = charts.bars(
                list(by_prov.index),
                list(by_prov.values),
                name="Fires",
                unit="detections",
                horizontal=True,
                height=260,
                color=charts.fire_color(),
                title=f"Fire detections by province · {'7 days' if window == '7d' else '24 h'}",
            )
            common.chart_and_table(fig, by_prov.rename("detections").reset_index(), key="live_fires_prov")
        if quakes30 is not None and len(quakes30):
            st.markdown(f"**Latest earthquakes nearby** · {len(quakes30)} in 30 days")
            for r in quakes30.head(4).itertuples():
                st.markdown(
                    f"**M{r.magnitude:.1f}** · {r.time_npt:%d %b, %H:%M} · {r.place}  \n"
                    f"<span style='opacity:.7'>{r.depth_km:.0f} km deep · {r.km_from_kathmandu} km from "
                    f"Kathmandu</span>",
                    unsafe_allow_html=True,
                )

    # --- weather & air table, full width ----------------------------------------------------------------------
    if now is not None:
        st.markdown("**Weather and air quality now**")
        tbl = now[
            [
                "place",
                "note",
                "temperature_c",
                "feels_like_c",
                "condition",
                "humidity_pct",
                "wind_kmh",
                "us_aqi",
                "pm2_5",
            ]
        ].copy()
        tbl.insert(8, "air", [weather.aqi_band(v)[0] for v in tbl.us_aqi])
        st.dataframe(
            tbl,
            hide_index=True,
            column_config={
                "place": "Place",
                "note": "Where",
                "temperature_c": st.column_config.NumberColumn("°C", format="%.0f"),
                "feels_like_c": st.column_config.NumberColumn("Feels like", format="%.0f °C"),
                "condition": "Sky",
                "humidity_pct": st.column_config.NumberColumn("Humidity", format="%.0f%%"),
                "wind_kmh": st.column_config.NumberColumn("Wind", format="%.0f km/h"),
                "us_aqi": st.column_config.NumberColumn("US AQI", format="%.0f"),
                "air": "Air quality",
                "pm2_5": st.column_config.NumberColumn("PM2.5", format="%.0f µg/m³"),
            },
        )

    # --- rivers ---------------------------------------------------------------------------------------------
    flows, e_r = got["rivers"]
    if flows is not None and not flows.empty:
        st.markdown("**Rivers: flow now and the 7-day forecast**")
        ref = data.rivers_reference()
        clim, thr = ref if ref else (None, None)
        today = pd.Timestamp.now(tz="Asia/Kathmandu").normalize().tz_localize(None)
        tiles = st.columns(len(RIVERS))
        for col, r in zip(tiles, RIVERS):
            d = flows[flows.river == r.key].sort_values("date")
            now_row = d[d.date <= today].tail(1)
            if now_row.empty:
                continue
            q_now = float(now_row.discharge.iloc[0])
            delta, help_txt = None, f"GloFAS model cell on the main stem near {r.gauge}."
            if clim is not None and thr is not None:
                c = clim[(clim.river == r.key) & (clim.doy == min(365, today.dayofyear))]
                t = thr[thr.river == r.key]
                if len(c) and len(t):
                    pct = q_now / float(c.p50.iloc[0]) * 100 if float(c.p50.iloc[0]) else float("nan")
                    label, token = rivers_mod.status(q_now, float(t.q2.iloc[0]), float(t.q10.iloc[0]),
                                                     float(c.p90.iloc[0]))
                    short = {"good": "", "warning": " · high", "serious": " · 2-yr flood", "critical": " · 10-yr flood"}
                    delta = f"{pct:.0f}% of normal{short[token]}"
                    help_txt += f" Status: {label}."
                    help_txt += (f" Normal for the date (2004–2023 median): {float(c.p50.iloc[0]):,.0f} m³/s; 2-year "
                                 f"flood level {float(t.q2.iloc[0]):,.0f} m³/s.")
            with col:
                st.metric(f"{r.name.split(' (')[0]} · m³/s", f"{q_now:,.0f}", delta=delta, delta_color="off",
                          delta_arrow="off",
                          border=True, chart_data=list(d.discharge.round(0)), chart_type="line", help=help_txt)
        with st.expander("River hydrographs: last 30 days, forecast and the normal range", icon=":material/water:"):
            fig = charts.river_hydrographs(flows, clim, RIVERS)
            common.chart_and_table(fig, flows.assign(discharge=flows.discharge.round(1)), key="live_rivers")
            st.caption(rivers_mod.ATTRIBUTION + ". Modelled flows, not gauge readings; use Nepal's Department of "
                       "Hydrology and Meteorology (hydrology.gov.np) for official flood warnings.")
    elif e_r:
        st.caption(f"River flows unavailable: {e_r}")

    # --- quakes table ---------------------------------------------------------------------------------------
    if quakes30 is not None and len(quakes30):
        with st.expander(
            f"Earthquakes in and around Nepal, last 30 days ({len(quakes30)})",
            expanded=False,
            icon=":material/vibration:",
        ):
            st.dataframe(
                quakes30[["time_npt", "magnitude", "depth_km", "place", "km_from_kathmandu", "url"]],
                hide_index=True,
                column_config={
                    "time_npt": st.column_config.DatetimeColumn("Time (NPT)", format="D MMM YYYY, HH:mm"),
                    "magnitude": st.column_config.NumberColumn("Mag.", format="%.1f"),
                    "depth_km": st.column_config.NumberColumn("Depth km", format="%.0f"),
                    "place": "Location",
                    "km_from_kathmandu": st.column_config.NumberColumn("km from KTM"),
                    "url": st.column_config.LinkColumn("USGS", display_text="details"),
                },
            )


live_panel()

# --- national land cover (atlas) -------------------------------------------------------------------------
st.subheader("Nepal's land cover, estimated by the U-Net", anchor=False)
settings = common.settings()
summary = atlas.national_latest(settings["model"])
if summary is None:
    st.info(
        "The national atlas has not been built yet. Run `python scripts/build_atlas.py` to create it.",
        icon=":material/info:",
    )
else:
    est, epoch_label, n_sites = summary
    common.preliminary_note(settings["model"])
    c1, c2 = st.columns([2.2, 1], gap="medium")
    with c1:
        fig = charts.composition_bar(
            est.set_index("key")["share"].reindex([c for c in est.key]).values,
            title=f"Share of Nepal's area · {epoch_label}",
        )
        common.chart_and_table(fig, est[["class", "share", "low", "high", "km2"]].round(2), key="live_atlas")
    with c2:
        forest = est[est.key == "forest"].iloc[0]
        st.metric(
            "Forest",
            f"{forest.share:.1f}%",
            border=True,
            help=f"95% interval {forest.low:.1f}–{forest.high:.1f}% from {n_sites} sample sites.",
        )
        urban = est[est.key == "urban"].iloc[0]
        st.metric(
            "Built-up", f"{urban.share:.1f}%", border=True, help=f"95% interval {urban.low:.1f}–{urban.high:.1f}%."
        )
        st.page_link("views/time_machine.py", label="See how it changed, year by year", icon=":material/history:")
    common.note(
        f"Design-based estimate from {n_sites} stratified random sites (every district), classified by the "
        f"{MODELS[settings['model']].short} model. Barren land includes rock, glaciers and permanent snow."
    )

    # --- district map ---------------------------------------------------------------------------------------
    ranked = atlas.headline_epochs(settings["model"])
    if ranked:
        epoch = ranked[0]
        table = atlas.district_table(settings["model"], epoch)
        st.markdown(f"**Land cover by district · {atlas.epoch_label(epoch)}**")
        cls_key = st.segmented_control(
            "Class", [c.key for c in CLASSES if c.key != "unknown"], default="forest", key="live_cls",
            format_func=lambda k: CLASS_BY_KEY[k].name, label_visibility="collapsed")
        cls_key = cls_key or "forest"
        values = dict(zip(table.district_code, table[cls_key]))
        extra = {r.district_code: f"± {1.96 * getattr(r, cls_key + '_se'):.1f} pp · {r.n} sites"
                 for r in table.itertuples()}
        left, right = st.columns([2.2, 1], gap="medium")
        with left:
            m = maps.base_map(basemap="light", labels=False)
            color = CLASS_BY_KEY[cls_key].light
            breaks, ramp = maps.add_choropleth(m, 2, values, label=f"{CLASS_BY_KEY[cls_key].name} share",
                                               color=color, extra=extra)
            st_folium(maps.finish(m), height=430, use_container_width=True, returned_objects=[], key="live_choro")
            st.html(maps.legend_html(breaks, ramp))
        with right:
            ranked = table.sort_values(cls_key, ascending=False)[["district", "province", cls_key, "n"]]
            st.markdown(f"**Highest {CLASS_BY_KEY[cls_key].name.lower()} share**")
            st.dataframe(ranked.head(8), hide_index=True, column_config={
                cls_key: st.column_config.NumberColumn("Share", format="%.1f%%"), "n": "Sites"})
            st.caption("District values come from that district's sample sites (8+ each); hover the map for the "
                       "95% margin.")

with st.expander("Sources and update frequency", icon=":material/database:"):
    st.markdown(
        "- **Fires:** NASA FIRMS VIIRS 375 m (S-NPP, NOAA-20, NOAA-21), updated within ~3 hours of each overpass.\n"
        "- **Earthquakes:** USGS ComCat, updated within minutes; completeness in Nepal is roughly M ≥ 3.5–4.\n"
        "- **Weather:** Open-Meteo (ECMWF/DWD models); **air quality:** CAMS global forecast (US AQI).\n"
        "- **Satellite passes:** Sentinel-2 L2A via Microsoft Planetary Computer; daily imagery via NASA GIBS.\n"
        "- **Rivers:** GloFAS v4 modelled discharge (Copernicus EMS) via Open-Meteo, updated daily with a 7-day "
        "forecast; *normal* = 2004–2023 median for the date, flood levels from the same years' annual maxima.\n"
        "- **Alerts:** US AQI > 150 (unhealthy) or > 100 (sensitive groups); 24-h fire detections > 2× the previous "
        "6-day mean; earthquakes ≥ M4.5 in 7 days; forecast rain ≥ 64.5 mm/day (heavy) or ≥ 115.6 mm/day "
        "(very heavy); river flow at or above its 2-year flood level.\n"
        "- **Land cover:** this project's U-Net applied to Sentinel-2 cloudless mosaics and Esri World Imagery."
    )
