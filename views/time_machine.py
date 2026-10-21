"""Time machine: any region of Nepal, at any time scale — a day, a month, years, decades, a century, or since 1768."""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from monitor_nepal.analysis import atlas, regional, stats
from monitor_nepal.config import MODEL_GSD_M, NEPAL_UNIFICATION_YEAR, PALETTES
from monitor_nepal.geo import Grid
from monitor_nepal.history import events as hist_events
from monitor_nepal.http import FetchError
from monitor_nepal.imagery import ImagerySpec, eox, gibs, planetary
from monitor_nepal.live import rivers as rivers_mod
from monitor_nepal.live.rivers import RIVERS
from monitor_nepal.ui import charts, common, data, maps, widgets

common.header(
    "Time machine",
    "Pick any region and any time scale. A single <b>day</b> or <b>month</b> (weather, fires, earthquakes, the "
    "satellite passes and a fresh land-cover map), <b>years</b> and <b>decades</b> (the U-Net run on every year's "
    "imagery, climate and hazards), a <b>century</b>, or the whole story <b>since Nepal's unification in "
    f"{NEPAL_UNIFICATION_YEAR}</b> — each period shown with the best evidence that exists for it.",
    eyebrow="Monitor",
)
settings = common.settings()
TODAY = date.today()

c1, c2 = st.columns([1.2, 2], gap="large")
with c1:
    region = widgets.region_picker("tm_region", default="NP")
with c2:
    scale = st.segmented_control(
        "Time scale",
        ["day", "month", "year", "decade", "century", "1768"],
        default="year",
        key="tm_scale",
        format_func={
            "day": "Day",
            "month": "Month",
            "year": "Years",
            "decade": "Decades",
            "century": "Century",
            "1768": f"Since {NEPAL_UNIFICATION_YEAR}",
        }.get,
    )
lat, lon = region.lat, region.lon
w_km, h_km = regional.region_grid_size_km(region)
SMALL = max(w_km, h_km) <= 12  # small enough for a wall-to-wall map


def safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except (FetchError, ValueError, KeyError) as exc:
        return None, str(exc)


def region_mask(grid: Grid) -> np.ndarray:
    """True for pixels inside the selected region's polygon."""
    from shapely import contains_xy

    from monitor_nepal import boundaries

    rows, cols = np.mgrid[0 : grid.height, 0 : grid.width]
    lon_px, lat_px = grid.pixel_lonlat(rows, cols)
    return contains_xy(boundaries.geometry(region), lon_px, lat_px)


def provenance(text: str) -> None:
    common.note(f"<b>Evidence:</b> {text}")


def land_cover_now(spec: ImagerySpec, key: str, label: str) -> None:
    """Classify the region with one imagery spec: wall-to-wall if small, otherwise a sample survey."""
    signature = (region.code, spec, settings["model"])
    common.model_source_warning(settings["model"], spec.kind)
    if st.button(f"Classify land cover · {label}", key=key, icon=":material/play_arrow:"):
        prog = st.progress(0.0, "Starting…")
        try:
            if SMALL:
                lon0, lat0, lon1, lat1 = regional.bbox(region)
                grid = Grid.from_lonlat_bbox((lon0, lat0, lon1, lat1), MODEL_GSD_M)
                res = data.classify_area(
                    grid, spec, settings["model"], settings["tta"], 64, _progress=lambda f, t: prog.progress(f, t)
                )
                st.session_state[key + "_res"] = (signature, "map", res)
            else:
                res = data.survey(
                    region.code,
                    spec,
                    settings["model"],
                    40,
                    0,
                    settings["tta"],
                    _progress=lambda f, t: prog.progress(f, t),
                )
                st.session_state[key + "_res"] = (signature, "survey", res)
        except FetchError as exc:
            st.error(str(exc))
        prog.empty()
    got = st.session_state.get(key + "_res")
    if got and got[0] == signature:
        _, kind, res = got
        if kind == "map":
            comp = stats.composition(res.classes, res.grid, res.scene.valid)
            fig = charts.composition_bar(comp.share.values, title=f"Land cover · {label}")
            common.chart_and_table(fig, comp[["class", "km2", "share"]].round(3), key=key + "_comp")
            pal = np.array(PALETTES[settings["palette"]], np.uint8)
            a, b = st.columns(2)
            a.image(res.scene.rgb, caption=res.scene.label, width="stretch")
            b.image(pal[res.classes], caption="U-Net land cover", width="stretch")
            common.legend()
        else:
            fig = charts.composition_ci(res.estimate, title=f"Land cover · {label} · {len(res.samples)} sample windows")
            common.chart_and_table(
                fig, res.estimate[["class", "share", "low", "high", "km2"]].round(2), key=key + "_est"
            )


# =====================================================================================================================
def day_view() -> None:
    d = st.date_input(
        "Day",
        value=TODAY - timedelta(days=1),
        min_value=date(1940, 1, 1),
        max_value=TODAY,
        key="tm_day",
        format="DD/MM/YYYY",
    )
    hourly, e_w = safe(data.weather_hourly, lat, lon, d)
    product = "VIIRS S-NPP" if d.year >= 2012 else "MODIS"
    if d.year >= 2000:
        got_fires, e_f = safe(data.fires_period, d, d, product)
        fires, notes = got_fires if got_fires else (None, [f"Fire records unavailable: {e_f}"])
    else:
        fires, notes = pd.DataFrame(), ["No satellite fire records before 2000."]
    if fires is not None and not fires.empty:
        local_day = (fires.time_utc + timedelta(hours=5, minutes=45)).dt.date
        fires = fires[(local_day == d) & regional.in_region(fires, region)]
    q, e_q = safe(data.quakes_between, d, d + timedelta(days=1), 2.5)
    if q is not None and not q.empty:
        q = q[regional.near_region(q, region, 50) & (q.time_npt.dt.date == d)]

    k = st.columns(5)
    if hourly is not None and not hourly.empty:
        k[0].metric(
            "Temperature",
            f"{hourly.temperature_2m.min():.0f}–{hourly.temperature_2m.max():.0f} °C",
            border=True,
            help=hourly.attrs.get("source"),
        )
        k[1].metric("Rain", f"{hourly.precipitation.sum():.1f} mm", border=True)
        k[2].metric("Snowfall", f"{hourly.snowfall.sum():.1f} cm", border=True)
    else:
        k[0].metric("Temperature", "—", border=True, help=e_w)
    k[3].metric(
        "Fire detections",
        "—" if fires is None else f"{len(fires)}",
        border=True,
        help=f"{product} detections inside {region.short} on this day (Nepal time).",
    )
    k[4].metric(
        "Earthquakes ≥ M2.5",
        "—" if q is None else f"{len(q)}",
        delta=None if q is None or q.empty else f"max M{q.magnitude.max():.1f}",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    for n in notes:
        st.caption(n)

    if hourly is not None and not hourly.empty:
        a, b = st.columns(2)
        with a:
            fig = charts.line(
                hourly,
                "time",
                "temperature_2m",
                name="Temperature",
                unit="°C",
                title="Temperature through the day",
                height=260,
                markers=True,
            )
            common.chart_and_table(fig, hourly[["time", "temperature_2m", "relative_humidity_2m"]], key="tm_d_t")
        with b:
            fig = charts.bars(
                hourly.time, hourly.precipitation, name="Rain", unit="mm", title="Rain per hour", height=260
            )
            common.chart_and_table(fig, hourly[["time", "precipitation", "snowfall", "cloud_cover"]], key="tm_d_p")
        provenance(
            f"{hourly.attrs.get('source')} at the centre of {region.short} ({hourly.attrs.get('elevation') or 0:,.0f} m above sea level)."
        )

    if d >= date(1984, 1, 1):
        flows_day, e_rd = safe(data.rivers_on_day, d)
        ref = data.rivers_reference()
        if flows_day is not None and not flows_day.dropna(subset=["discharge"]).empty:
            tbl = flows_day.dropna(subset=["discharge"]).copy()
            if ref:
                clim, thr = ref
                doy = min(365, d.timetuple().tm_yday)
                med = clim[clim.doy == doy].set_index("river").p50
                p90 = clim[clim.doy == doy].set_index("river").p90
                t = thr.set_index("river")
                tbl["normal"] = tbl.river.map(med)
                tbl["pct_normal"] = tbl.discharge / tbl.normal * 100
                tbl["status"] = [rivers_mod.status(q, t.q2.get(k, np.inf), t.q10.get(k, np.inf), p90.get(k, np.inf))[0]
                                 for k, q in zip(tbl.river, tbl.discharge)]
            st.markdown(f"**Major rivers on {d:%d %b %Y}** (GloFAS model)")
            st.dataframe(tbl.drop(columns=["river"]), hide_index=True, column_config={
                "name": "River", "gauge": "Near",
                "discharge": st.column_config.NumberColumn("Flow (m³/s)", format="%,.0f"),
                "normal": st.column_config.NumberColumn("Normal for the date", format="%,.0f"),
                "pct_normal": st.column_config.NumberColumn("% of normal", format="%.0f%%"), "status": "Status"})

    st.subheader("From space that day", anchor=False)
    bb = regional.bbox(region, 0.05)
    cols = st.columns(2)
    for col, layer in zip(cols, ("modis_terra", "viirs_snpp")):
        L = gibs.LAYERS[layer]
        with col:
            if d < L.start:
                st.caption(f"{L.title}: starts {L.start:%Y}.")
                continue
            img, err = safe(data.gibs_snapshot, layer, d, bb, 900)
            if img is not None:
                st.image(img, caption=f"{L.title} · {d:%d %b %Y}", width="stretch")
            else:
                st.caption(f"{L.title}: {err}")
    if d >= date(2015, 7, 1):
        sc, err = safe(data.scenes, "s2", bb, d - timedelta(days=7), min(TODAY, d + timedelta(days=7)))
        if sc:
            df = pd.DataFrame([{"date": s.day, "cloud_pct": s.cloud, "platform": s.platform} for s in sc])
            summary = (
                df.groupby("date")
                .agg(scenes=("cloud_pct", "size"), cloud_pct=("cloud_pct", "mean"))
                .reset_index()
                .sort_values("date", ascending=False)
            )
            st.markdown(f"**Sentinel-2 passes within a week of {d:%d %b %Y}**")
            st.dataframe(
                summary,
                hide_index=True,
                height=200,
                column_config={
                    "cloud_pct": st.column_config.ProgressColumn(
                        "Mean cloud", format="%.0f%%", min_value=0, max_value=100
                    )
                },
            )
            land_cover_now(
                ImagerySpec("s2", start=d - timedelta(days=7), end=min(TODAY, d + timedelta(days=7)), cloud_max=40),
                "tm_day_lc",
                f"Sentinel-2 around {d:%d %b %Y}",
            )
    elif d >= date(1982, 9, 1):
        land_cover_now(
            ImagerySpec("landsat", start=d - timedelta(days=30), end=d + timedelta(days=30), cloud_max=40),
            "tm_day_lc",
            f"Landsat around {d:%d %b %Y} (experimental)",
        )
    if fires is not None and not fires.empty:
        m = maps.base_map(
            center=(lat, lon),
            zoom=8 if region.level <= 1 else 9,
            nasa_day=d if d.year >= 2000 else None,
            basemap="nasa" if d.year >= 2000 else "imagery",
        )
        if region.level:
            maps.add_region(m, region)
        maps.add_fires(m, fires)
        st_folium(maps.finish(m), height=420, use_container_width=True, returned_objects=[], key="tm_day_map")


# =====================================================================================================================
def month_view() -> None:
    c1, c2 = st.columns(2)
    last_full = TODAY.replace(day=1) - timedelta(days=1)
    year = c1.selectbox("Year", list(range(TODAY.year, 1939, -1)), index=0 if TODAY.month > 1 else 1, key="tm_m_year")
    months = list(range(1, 13)) if year < TODAY.year else list(range(1, TODAY.month + 1))
    default_m = last_full.month if year == last_full.year else months[-1]
    month = c2.selectbox(
        "Month",
        months,
        index=months.index(default_m) if default_m in months else 0,
        format_func=lambda m: calendar.month_name[m],
        key="tm_m_month",
    )
    start = date(year, month, 1)
    end = min(TODAY, date(year, month, calendar.monthrange(year, month)[1]))
    daily, e_w = safe(data.weather_daily, lat, lon, start, end)
    if daily is not None and not daily.empty:
        prev = []
        for py in range(year - 5, year):
            pe = date(py, month, calendar.monthrange(py, month)[1])
            dd, _ = safe(data.weather_daily, lat, lon, date(py, month, 1), pe)
            if dd is not None and not dd.empty:
                prev.append({"year": py, "t": dd.temperature_2m_mean.mean(), "p": dd.precipitation_sum.sum()})
        prev = pd.DataFrame(prev)
        k = st.columns(4)
        k[0].metric(
            "Mean temperature",
            f"{daily.temperature_2m_mean.mean():.1f} °C",
            None if prev.empty else f"{daily.temperature_2m_mean.mean() - prev.t.mean():+.1f} °C vs previous 5 yrs",
            delta_color="off",
            border=True,
        )
        k[1].metric(
            "Total rain",
            f"{daily.precipitation_sum.sum():.0f} mm",
            None
            if prev.empty or prev.p.mean() == 0
            else f"{(daily.precipitation_sum.sum() / prev.p.mean() - 1) * 100:+.0f}% vs previous 5 yrs",
            delta_color="off",
            border=True,
        )
        k[2].metric(
            "Wettest day",
            f"{daily.precipitation_sum.max():.0f} mm",
            f"{pd.Timestamp(daily.time[daily.precipitation_sum.idxmax()]):%d %b}",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
        k[3].metric("Snowfall", f"{daily.snowfall_sum.sum():.0f} cm", border=True)
        a, b = st.columns(2)
        with a:
            fig = charts.line(
                daily,
                "time",
                "temperature_2m_mean",
                name="Mean",
                unit="°C",
                band=("temperature_2m_min", "temperature_2m_max"),
                title="Daily temperature (band: min–max)",
                height=270,
            )
            common.chart_and_table(fig, daily, key="tm_m_t")
        with b:
            fig = charts.bars(
                daily.time, daily.precipitation_sum, name="Rain", unit="mm", title="Daily rain", height=270
            )
            common.chart_and_table(fig, daily[["time", "precipitation_sum", "snowfall_sum"]], key="tm_m_p")
    else:
        st.warning(f"Weather unavailable: {e_w}")

    product = "VIIRS S-NPP" if year >= 2012 else "MODIS"
    got_fires, e_f = safe(data.fires_period, start, end, product) if year >= 2000 else (None, None)
    if got_fires:
        fires, notes = got_fires
        fires = fires[regional.in_region(fires, region)] if not fires.empty else fires
        from monitor_nepal.live import fires as fires_mod

        counts = fires_mod.daily_counts(fires, start, end)
        fig = charts.bars(
            list(counts.index),
            list(counts.values),
            name="Fires",
            unit="detections",
            height=250,
            title=f"Fire detections per day · {region.short} · {product}",
            color=charts.fire_color(),
        )
        common.chart_and_table(fig, counts.rename("detections").rename_axis("day").reset_index(), key="tm_m_f")
        for n in notes:
            st.caption(n)
    elif e_f:
        st.caption(f"Fire records unavailable: {e_f}")
    q, _ = safe(data.quakes_between, start, end + timedelta(days=1), 2.5)
    if q is not None and not q.empty:
        q = q[regional.near_region(q, region, 50)]
        if len(q):
            st.markdown(f"**Earthquakes ≥ M2.5 near {region.short}: {len(q)}** (strongest M{q.magnitude.max():.1f})")
            st.dataframe(q[["time_npt", "magnitude", "depth_km", "place"]], hide_index=True, height=180)
    if start >= date(2015, 7, 1):
        sc, _ = safe(data.scenes, "s2", regional.bbox(region, 0.02), start, end)
        if sc:
            df = pd.DataFrame([{"date": pd.Timestamp(s.day), "cloud": s.cloud} for s in sc])
            fig = go.Figure(
                go.Scatter(
                    x=df.date,
                    y=df.cloud,
                    mode="markers",
                    marker=dict(size=9, color=charts.single_color(), line=dict(color=common.surface(), width=2)),
                    hovertemplate="%{x|%d %b}: %{y:.0f}% cloud<extra></extra>",
                )
            )
            charts.style(
                fig,
                220,
                legend=False,
                hovermode="closest",
                title=f"Sentinel-2 passes over {region.short} and their cloud cover",
            )
            fig.update_yaxes(ticksuffix="%", range=[-5, 105])
            common.chart_and_table(fig, df, key="tm_m_s2")
        land_cover_now(
            ImagerySpec("s2", start=start, end=end, cloud_max=40),
            "tm_month_lc",
            f"least-cloudy Sentinel-2, {calendar.month_name[month]} {year}",
        )
        snow_col, ndvi_col = st.columns(2)
        with snow_col:
            if st.button("Snow cover this month (NDSI)", key="tm_m_ndsi", icon=":material/ac_unit:",
                         help="The U-Net has no snow class (snow counts as barren land), so snow is measured "
                              "physically: Normalised Difference Snow Index from Sentinel-2, snow where NDSI > 0.4."):
                try:
                    with st.spinner("Computing NDSI from Sentinel-2…"):
                        lon0, lat0, lon1, lat1 = regional.bbox(region)
                        res_m = max(20.0, max(w_km, h_km) * 1000 / 900)
                        g = Grid.from_lonlat_bbox((lon0, lat0, lon1, lat1), res_m)
                        nd, ok = planetary.fetch_index("s2", "ndsi", g, start, end, 40)
                        st.session_state["tm_ndsi"] = ((region.code, start), nd, ok & region_mask(g))
                except FetchError as exc:
                    st.error(str(exc))
            got_ndsi = st.session_state.get("tm_ndsi")
            if got_ndsi and got_ndsi[0] == (region.code, start) and got_ndsi[2].any():
                _, nd, ok = got_ndsi
                snow = (nd > 0.4) & ok
                st.metric("Snow-covered", f"{snow.sum() / ok.sum() * 100:.1f}%", border=True,
                          help="Share of the region's valid Sentinel-2 pixels with NDSI > 0.4 in the least-cloudy "
                               "scenes of the month (clouds can be mistaken for snow at the margins).")
                light, deep = np.array([241, 240, 236], float), np.array([16, 66, 129], float)
                r = np.clip(np.nan_to_num(nd, nan=-1.0), 0, 1)[..., None]
                img = (light + (deep - light) * r).astype(np.uint8)
                img[~ok] = (200, 200, 200)
                st.image(img, caption="NDSI: pale = no snow · deep blue = snow and ice", width="stretch")
        with ndvi_col:
            if st.button("Vegetation greenness this month (NDVI)", key="tm_m_ndvi", icon=":material/eco:"):
                try:
                    with st.spinner("Computing NDVI from Sentinel-2…"):
                        lon0, lat0, lon1, lat1 = regional.bbox(region)
                        res_m = max(20.0, max(w_km, h_km) * 1000 / 900)
                        g = Grid.from_lonlat_bbox((lon0, lat0, lon1, lat1), res_m)
                        nd, ok = planetary.fetch_index("s2", "ndvi", g, start, end, 40)
                        st.session_state["tm_ndvi"] = ((region.code, start), nd, ok & region_mask(g))
                except FetchError as exc:
                    st.error(str(exc))
            got_ndvi = st.session_state.get("tm_ndvi")
            if got_ndvi and got_ndvi[0] == (region.code, start) and got_ndvi[2].any():
                _, nd, ok = got_ndvi
                st.metric("Mean NDVI", f"{np.nanmean(nd[ok]):.2f}", border=True,
                          help="−1…1; dense healthy vegetation > 0.6, bare soil or rock ~0.1, water < 0.")
                # one hue, light -> dark: pale for bare ground and snow, deep green for dense vegetation
                r = np.clip(np.nan_to_num(nd, nan=0.0) / 0.85, 0, 1)[..., None]
                light, dark = np.array([241, 240, 236], float), np.array([12, 84, 34], float)
                rgb = (light + (dark - light) * r).astype(np.uint8)
                rgb[~ok] = (200, 200, 200)
                st.image(rgb, caption="NDVI: pale = bare ground, rock, snow or water · deep green = dense vegetation",
                         width="stretch")


# =====================================================================================================================
def year_view() -> None:
    y0, y1 = st.slider("Years", 2000, TODAY.year, (2016, TODAY.year - 1), key="tm_years")
    tab_lc, tab_fire, tab_quake, tab_river, tab_clim, tab_nat = st.tabs(
        ["Land cover", "Fires", "Earthquakes", "Rivers", "Climate", "National statistics"],
        key="tm_y_tabs",
        on_change="rerun",
    )
    if tab_lc.open:
        with tab_lc:
            trend = atlas.trend(region, settings["model"]) if atlas.available() else pd.DataFrame()
            if not trend.empty and trend["n"].iloc[0] >= 8:
                trend = trend[trend.label.str.extract(r"(\d{4})")[0].astype(float).between(y0, y1)]
                trend = trend.assign(year=trend.label.str.extract(r"(\d{4})")[0].astype(int))
                ref = atlas.reference_trend(region)
                ref = ref[ref.year.between(y0, y1)] if not ref.empty else None
                common.preliminary_note(settings["model"])
                common.note("Each year is one Sentinel-2 cloudless mosaic, and the mosaics were processed differently "
                            "from year to year. Differences of a few points between years can come from that "
                            "processing rather than from the land, so read the trend next to the independent "
                            "reference map in grey.")
                fig = charts.class_trends(trend, x="year", ref=ref, title=f"Land cover of {region.short}, year by year")
                common.chart_and_table(
                    fig, trend[["label", "class", "share", "low", "high", "km2", "n"]].round(2), key="tm_y_lc"
                )
                eps = atlas.epochs(settings["model"])
                yr = [e for e in eps if e.startswith("eox-")]
                if len(yr) >= 2:
                    ch = atlas.change(region, settings["model"], yr[0], yr[-1])
                    if ch is not None:
                        st.markdown(
                            f"**Change {atlas.epoch_label(yr[0])} → {atlas.epoch_label(yr[-1])}** "
                            f"(paired estimate over {int(ch.n.iloc[0])} fixed sites)"
                        )
                        st.dataframe(
                            ch[ch.key != "unknown"][["class", "change_pp", "low", "high", "km2", "p_value"]].round(3),
                            hide_index=True,
                            column_config={
                                "change_pp": st.column_config.NumberColumn("Change (pp)", format="%+.2f"),
                                "km2": st.column_config.NumberColumn("≈ km²", format="%+.0f"),
                                "p_value": st.column_config.NumberColumn("p", format="%.3f"),
                            },
                        )
                provenance(
                    "the U-Net applied to Sentinel-2 cloudless annual mosaics (EOX) at fixed random sites; grey "
                    "lines show Impact Observatory's independent 10 m map at the same sites."
                )
            else:
                years = [y for y in eox.years() if y0 <= y <= y1]
                common.model_source_warning(settings["model"], "eox")
                st.write(
                    f"Run the U-Net on the Sentinel-2 cloudless mosaic of every year {years[0] if years else ''}"
                    f"–{years[-1] if years else ''} at the same {24} random sites in {region.short}."
                )
                if years and st.button("Run the yearly analysis", key="tm_y_run", icon=":material/play_arrow:"):
                    prog = st.progress(0.0)
                    try:
                        epochs = tuple((str(y), ImagerySpec("eox", year=y)) for y in years)
                        out = data.survey_series(
                            region.code, epochs, settings["model"], 24, 0, 1, _progress=lambda f, t: prog.progress(f, t)
                        )
                        st.session_state["tm_y_series"] = ((region.code, tuple(years), settings["model"]), out)
                    except FetchError as exc:
                        st.error(str(exc))
                    prog.empty()
                got = st.session_state.get("tm_y_series")
                if got and got[0] == (region.code, tuple(years), settings["model"]):
                    long, _ = got[1]
                    long = long.assign(year=long.label.astype(int))
                    fig = charts.class_trends(long, x="year", title=f"Land cover of {region.short}, year by year")
                    common.chart_and_table(
                        fig, long[["year", "class", "share", "low", "high", "km2"]].round(2), key="tm_y_lc2"
                    )
    if tab_fire.open:
        with tab_fire:
            product = st.segmented_control(
                "Sensor",
                ["VIIRS S-NPP", "MODIS"],
                default="VIIRS S-NPP",
                key="tm_y_sensor",
                help="VIIRS (from 2012) sees smaller fires; MODIS (from 2000) has the longest record.",
            )
            first = 2012 if product == "VIIRS S-NPP" else 2000
            ys = list(range(max(y0, first), min(y1, TODAY.year - 1) + 1))
            if not ys:
                st.info(f"{product} archives cover {first}–{TODAY.year - 1}.")
            else:
                frames = []
                prog = st.progress(0.0, "Loading fire archives…")
                for i, y in enumerate(ys):
                    df, _ = safe(data.fires_archive, y, product)
                    if df is not None and not df.empty:
                        df = df[regional.in_region(df, region)]
                        frames.append(df.assign(year=y, month=(df.time_utc + timedelta(hours=5, minutes=45)).dt.month))
                    prog.progress((i + 1) / len(ys))
                prog.empty()
                allf = pd.concat(frames) if frames else pd.DataFrame(columns=["year", "month"])
                per_year = allf.groupby("year").size().reindex(ys, fill_value=0)
                fig = charts.bars(
                    [str(y) for y in per_year.index],
                    per_year.values,
                    name="Fires",
                    unit="detections",
                    title=f"Fire detections per year · {region.short} · {product}",
                    color=charts.fire_color(),
                )
                common.chart_and_table(
                    fig, per_year.rename("detections").rename_axis("year").reset_index(), key="tm_y_f"
                )
                hm = (
                    allf.groupby(["year", "month"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(index=ys, columns=range(1, 13), fill_value=0)
                )
                hm.columns = [calendar.month_abbr[m] for m in hm.columns]
                fig = charts.heatmap(hm, title="When the fires burn: detections by month", unit="detections")
                common.chart_and_table(fig, hm.reset_index(), key="tm_y_fh")
                provenance(
                    "NASA FIRMS archive (standard processing). Nepal's fire season peaks in March–May, mostly "
                    "forest-floor and agricultural burning."
                )
    if tab_quake.open:
        with tab_quake:
            q, err = safe(data.quakes_between, date(y0, 1, 1), min(TODAY, date(y1, 12, 31)), 4.0)
            if q is None:
                st.warning(err)
            else:
                q = q[regional.near_region(q, region, 100)]
                per = q.groupby(q.time_npt.dt.year).size().reindex(range(y0, y1 + 1), fill_value=0)
                fig = charts.bars(
                    [str(y) for y in per.index],
                    per.values,
                    name="Earthquakes",
                    unit="events",
                    title=f"Earthquakes ≥ M4 within ~100 km of {region.short}",
                    color=charts.quake_color(),
                )
                common.chart_and_table(fig, per.rename("events").rename_axis("year").reset_index(), key="tm_y_q")
                if len(q):
                    st.dataframe(
                        q.nlargest(8, "magnitude")[["time_npt", "magnitude", "depth_km", "place"]], hide_index=True
                    )
    if tab_river.open:
        with tab_river:
            ann = data.rivers_annual()
            ref = data.rivers_reference()
            if ann is None or ref is None:
                st.info("River climatology not bundled yet (run scripts/build_rivers.py).")
            else:
                key = st.segmented_control("River", [r.key for r in RIVERS], default="koshi", key="tm_y_river",
                                           format_func=lambda k: rivers_mod.BY_KEY[k].name) or "koshi"
                r = rivers_mod.BY_KEY[key]
                a = ann[(ann.river == key) & ann.year.between(y0, y1)]
                thr = ref[1].set_index("river").loc[key]
                if a.empty:
                    st.info("The bundled river record covers 2004–2023.")
                else:
                    fig = charts.bars(a.year.astype(str), a.peak, name="Annual peak", unit="m³/s",
                                      title=f"{r.name} at {r.gauge}: highest daily flow each year", height=300)
                    for level, label in ((thr.q2, "2-year flood level"), (thr.q10, "10-year flood level")):
                        fig.add_hline(y=level, line_width=1, line_color=charts.INK_MUTED,
                                      annotation_text=f"{label} · {level:,.0f}", annotation_position="top left",
                                      annotation=dict(font=dict(size=11, color=charts.INK_MUTED)))
                    common.chart_and_table(fig, a, key="tm_y_river_peaks")
                    provenance("GloFAS v4 river-discharge reanalysis (Copernicus EMS) via Open-Meteo, main-stem model "
                               "cell; flood levels from the 2004–2023 annual maxima of the same model.")
    if tab_clim.open:
        with tab_clim:
            span = y1 - y0 + 1
            if span > 20:
                st.info(
                    "Pick 20 years or fewer for region-level climate (the free weather API is rate-limited); "
                    "the Century view has national climate back to 1901."
                )
            elif st.toggle(f"Show ERA5 climate for {y0}–{y1} at the centre of {region.short}", key="tm_y_clim"):
                d, err = safe(
                    data.weather_daily, lat, lon, date(y0, 1, 1), min(TODAY - timedelta(days=7), date(y1, 12, 31))
                )
                if d is None:
                    st.error(err)
                else:
                    d["year"] = d.time.dt.year
                    ann = (
                        d.groupby("year")
                        .agg(
                            temperature=("temperature_2m_mean", "mean"),
                            rain=("precipitation_sum", "sum"),
                            snow=("snowfall_sum", "sum"),
                        )
                        .reset_index()
                    )
                    a, b = st.columns(2)
                    with a:
                        fig = charts.line(
                            ann,
                            "year",
                            "temperature",
                            name="Mean temperature",
                            unit="°C",
                            markers=True,
                            title=f"Annual mean temperature · centre of {region.short}",
                            height=280,
                        )
                        common.chart_and_table(fig, ann.round(2), key="tm_y_ct")
                    with b:
                        fig = charts.bars(
                            ann.year.astype(str),
                            ann.rain,
                            name="Precipitation",
                            unit="mm",
                            title="Annual precipitation",
                            height=280,
                        )
                        common.chart_and_table(fig, ann.round(1), key="tm_y_cp")
    if tab_nat.open:
        with tab_nat:
            national_indicators(y0, y1)


def national_indicators(y0: int, y1: int) -> None:
    keys = ["forest_share", "agri_share", "urban_share", "population"]
    cols = st.columns(2)
    for i, key in enumerate(keys):
        df, err = safe(data.wb, key)
        with cols[i % 2]:
            if df is None or df.empty:
                st.caption(err)
                continue
            from monitor_nepal.history.sources import WB_INDICATORS

            ind = WB_INDICATORS[key]
            d = df[df.year.between(y0, y1)]
            if d.empty:
                st.caption(f"{ind.name}: no data for {y0}–{y1}.")
                continue
            fig = charts.line(
                d, "year", "value", name=ind.name, unit=ind.unit, markers=True, height=240, title=f"{ind.name} · Nepal"
            )
            common.chart_and_table(fig, d, key=f"tm_nat_{key}")
    provenance("World Bank World Development Indicators (forest and farm land from FAO). National totals.")


# =====================================================================================================================
DECADE_WINDOWS = {
    "1970s": ("mss", date(1976, 10, 1), date(1977, 3, 31)),
    "1980s": ("landsat", date(1988, 10, 1), date(1989, 3, 31)),
    "1990s": ("landsat", date(1998, 10, 1), date(1999, 3, 31)),
    "2000s": ("landsat", date(2008, 10, 1), date(2009, 3, 31)),
    "2010s": ("landsat", date(2018, 10, 1), date(2019, 3, 31)),
    "2020s": ("landsat", date(2023, 10, 1), date(2024, 3, 31)),
}


def decade_view() -> None:
    tab_lc, tab_clim, tab_people, tab_q = st.tabs(
        ["Land cover from Landsat", "Climate", "People & land", "Earthquakes"], key="tm_dec_tabs", on_change="rerun"
    )
    if tab_lc.open:
        with tab_lc:
            st.write(
                "One dry-season (October–March) Landsat mosaic per decade, classified at the same random sites. "
                "Landsat's 30 m pixels are coarse for this network, so read these as broad patterns."
            )
            st.badge("Experimental", color="orange", icon=":material/science:")
            common.model_source_warning(settings["model"], "landsat")
            decs = [d for d in DECADE_WINDOWS if d != "1970s"]
            if st.button("Run the decade analysis", key="tm_dec_run", icon=":material/play_arrow:"):
                prog = st.progress(0.0)
                try:
                    epochs = tuple(
                        (dec, ImagerySpec(k, start=a, end=b, cloud_max=30))
                        for dec, (k, a, b) in DECADE_WINDOWS.items()
                        if dec in decs
                    )
                    out = data.survey_series(
                        region.code, epochs, settings["model"], 24, 0, 1, _progress=lambda f, t: prog.progress(f, t)
                    )
                    st.session_state["tm_dec"] = ((region.code, settings["model"]), out)
                except FetchError as exc:
                    st.error(str(exc))
                prog.empty()
            got = st.session_state.get("tm_dec")
            if got and got[0] == (region.code, settings["model"]):
                long, fracs = got[1]
                long = long.assign(decade=long.label)
                fig = charts.stacked_columns(
                    long.rename(columns={"decade": "x"}),
                    x="x",
                    title=f"Land cover of {region.short} by decade (Landsat)",
                )
                common.chart_and_table(
                    fig, long[["label", "class", "share", "low", "high", "km2"]].round(2), key="tm_dec_lc"
                )
            with st.expander(
                "What the 1970s looked like (Landsat MSS, false colour — red is vegetation)",
                icon=":material/photo_camera:",
            ):
                if st.button("Show 1970s image of the region centre", key="tm_mss"):
                    try:
                        g = Grid.around(lon, lat, min(20000, max(5000, w_km * 500)), 30.0)
                        k_, a_, b_ = DECADE_WINDOWS["1970s"]
                        sc = planetary.mosaic_source("mss", g.bounds_lonlat, a_, b_, 40).fetch(g)
                        st.image(
                            sc.rgb,
                            caption=f"Landsat MSS, winter 1976–77 · {g.ground_size_km[0]:.0f} km wide",
                            width=640,
                        )
                    except FetchError as exc:
                        st.error(str(exc))
    if tab_clim.open:
        with tab_clim:
            cru_t, _ = safe(data.climate, "cru_tas")
            cru_p, _ = safe(data.climate, "cru_pr")
            if cru_t is not None:
                t = cru_t.assign(decade=(cru_t.year // 10) * 10).groupby("decade").value.mean().reset_index()
                base = t[t.decade.between(1901, 1930)].value.mean()
                t["anomaly"] = t.value - base
                a, b = st.columns(2)
                with a:
                    fig = charts.bars(
                        [f"{d}s" for d in t.decade],
                        t.anomaly.round(2),
                        name="Anomaly",
                        unit="°C",
                        title="Nepal's temperature by decade (vs 1900s–1920s)",
                        height=280,
                    )
                    common.chart_and_table(fig, t.round(3), key="tm_dec_t")
                if cru_p is not None:
                    p = cru_p.assign(decade=(cru_p.year // 10) * 10).groupby("decade").value.mean().reset_index()
                    with b:
                        fig = charts.bars(
                            [f"{d}s" for d in p.decade],
                            p.value.round(0),
                            name="Precipitation",
                            unit="mm/yr",
                            title="Average annual precipitation by decade",
                            height=280,
                        )
                        common.chart_and_table(fig, p.round(1), key="tm_dec_p")
                provenance("CRU TS 4.08 via the World Bank Climate Change Knowledge Portal; national averages.")
    if tab_people.open:
        with tab_people:
            rows = []
            for key in ("population", "urban_share", "agri_share", "forest_share"):
                df, _ = safe(data.wb, key)
                if df is not None and not df.empty:
                    g = df.assign(decade=(df.year // 10) * 10).groupby("decade").value.mean()
                    for dec, v in g.items():
                        rows.append({"decade": f"{dec}s", "indicator": key, "value": v})
            if rows:
                tbl = pd.DataFrame(rows).pivot(index="decade", columns="indicator", values="value").reset_index()
                tbl = tbl.rename(
                    columns={
                        "population": "Population",
                        "urban_share": "Urban %",
                        "agri_share": "Farm land %",
                        "forest_share": "Forest %",
                    }
                )
                st.dataframe(
                    tbl,
                    hide_index=True,
                    column_config={
                        "Population": st.column_config.NumberColumn(format="%.0f"),
                        "Urban %": st.column_config.NumberColumn(format="%.1f"),
                        "Farm land %": st.column_config.NumberColumn(format="%.1f"),
                        "Forest %": st.column_config.NumberColumn(format="%.1f"),
                    },
                )
                provenance("World Bank WDI decade averages (forest area from 1990).")
    if tab_q.open:
        with tab_q:
            q, err = safe(data.quakes_between, date(1900, 1, 1), TODAY, 5.0)
            if q is not None:
                q = q[regional.near_region(q, region, 150)]
                dec = (
                    q.groupby((q.time_npt.dt.year // 10) * 10)
                    .size()
                    .reindex(range(1900, TODAY.year + 1, 10), fill_value=0)
                )
                fig = charts.bars(
                    [f"{d}s" for d in dec.index],
                    dec.values,
                    name="Earthquakes",
                    unit="events",
                    title=f"Earthquakes ≥ M5 within ~150 km of {region.short}, by decade",
                    color=charts.quake_color(),
                )
                common.chart_and_table(fig, dec.rename("events").rename_axis("decade").reset_index(), key="tm_dec_q")
                provenance(
                    "USGS ComCat (ISC-GEM before 1973). Older decades only record the larger events, so counts "
                    "rise partly because detection improved."
                )


# =====================================================================================================================
def century_view() -> None:
    evs = hist_events.between(1900, TODAY.year)
    cru_t, e1 = safe(data.climate, "cru_tas")
    era_t, _ = safe(data.climate, "era5_tas")
    if cru_t is not None:
        base = cru_t[cru_t.year.between(1901, 1930)].value.mean()
        d = cru_t.assign(anomaly=cru_t.value - base)
        d["smooth"] = d.anomaly.rolling(11, center=True, min_periods=5).mean()
        slope = np.polyfit(d.year[d.year >= 1971], d.anomaly[d.year >= 1971], 1)[0] * 10
        fig = go.Figure()
        fig.add_bar(
            x=d.year,
            y=d.anomaly,
            name="Yearly anomaly",
            marker=dict(color=charts.rgba(charts.single_color(), 0.45)),
            hovertemplate="%{x}: %{y:+.2f} °C<extra></extra>",
        )
        fig.add_scatter(
            x=d.year,
            y=d.smooth,
            name="11-year mean",
            mode="lines",
            line=dict(color=charts.fire_color(), width=2),
            hovertemplate="%{x}: %{y:+.2f} °C (11-yr)<extra></extra>",
        )
        charts.style(fig, 340, title="Nepal is warming: temperature vs the 1901–1930 average")
        fig.update_yaxes(ticksuffix=" °C")
        charts.add_events(fig, [e for e in evs if e.category in ("state", "disaster")])
        common.chart_and_table(fig, d.round(3), key="tm_c_t")
        st.caption(
            f"Since 1971 Nepal has warmed by about {slope:.2f} °C per decade (CRU TS 4.08, national mean; "
            "diamonds mark historical events — hover for details)."
        )
    else:
        st.warning(e1)
    a, b = st.columns(2)
    pop, _ = safe(data.owid, "population_long")
    lu, _ = safe(data.owid, "land_use_long")
    with a:
        if pop is not None:
            d = pop[pop.year >= 1900]
            fig = charts.line(
                d,
                "year",
                "population",
                name="Population",
                unit="people",
                markers=True,
                title="Population of Nepal since 1900",
                height=300,
            )
            common.chart_and_table(fig, d, key="tm_c_pop")
    with b:
        if lu is not None:
            d = lu[lu.year >= 1900].copy()
            d["cropland_km2"] = d.cropland_ha / 100
            fig = charts.line(
                d,
                "year",
                "cropland_km2",
                name="Cropland",
                unit="km²",
                markers=True,
                title="Cropland since 1900 (HYDE reconstruction)",
                height=300,
            )
            common.chart_and_table(fig, d, key="tm_c_crop")
    q, _ = safe(data.quakes_between, date(1900, 1, 1), TODAY, 6.0)
    if q is not None and not q.empty:
        q = q[regional.near_region(q, region, 200)]
        fig = go.Figure(
            go.Scatter(
                x=q.time_npt,
                y=q.magnitude,
                mode="markers",
                marker=dict(
                    size=(q.magnitude - 5) * 7,
                    color=charts.rgba(charts.quake_color(), 0.6),
                    line=dict(color=common.surface(), width=2),
                ),
                text=q.place,
                hovertemplate="%{x|%d %b %Y}: M%{y:.1f}<br>%{text}<extra></extra>",
            )
        )
        charts.style(
            fig,
            300,
            legend=False,
            hovermode="closest",
            title=f"Strong earthquakes (≥ M6) within ~200 km of {region.short} since 1900",
        )
        common.chart_and_table(fig, q[["time_npt", "magnitude", "depth_km", "place"]], key="tm_c_q")
    st.subheader("The satellite era", anchor=False)
    st.write(
        "Satellites have watched Nepal since 1972 (Landsat). For year-by-year land cover since 2016 use the "
        "**Years** view; for the Landsat decades, the **Decades** view."
    )
    provenance(
        "climate: CRU TS 4.08 (national); population and cropland: HYDE 3.3 via Our World in Data; "
        "earthquakes: USGS/ISC-GEM. Long-run series are national even when a smaller region is selected."
    )


def since_1768_view() -> None:
    years = TODAY.year - NEPAL_UNIFICATION_YEAR
    st.markdown(f"### {years} years of Nepal")
    st.write(
        "No satellite saw Nepal before 1972, so the long story is told by historical reconstructions "
        "(HYDE 3.3, built from censuses, historical sources and land-use models), then national statistics, "
        "then satellites — each era is labelled on the charts."
    )
    bands = [
        (NEPAL_UNIFICATION_YEAR, 1960, "Historical reconstruction (HYDE)"),
        (1960, 1972, ""),
        (1972, TODAY.year, "Statistics + satellites"),
    ]
    evs = hist_events.between(NEPAL_UNIFICATION_YEAR, TODAY.year)
    pop, e1 = safe(data.owid, "population_long")
    lu, e2 = safe(data.owid, "land_use_long")
    ur, _ = safe(data.owid, "urban_rural_long")
    if pop is not None:
        d = pop[pop.year.between(1700, TODAY.year)]
        fig = charts.line(
            d,
            "year",
            "population",
            name="Population",
            unit="people",
            markers=True,
            title=f"Population since {NEPAL_UNIFICATION_YEAR} · diamonds mark events (hover)",
            height=360,
        )
        charts.add_era_bands(fig, bands)
        charts.add_events(fig, evs)
        fig.update_xaxes(range=[NEPAL_UNIFICATION_YEAR - 8, TODAY.year + 2])
        common.chart_and_table(fig, d, key="tm_1768_pop")
        p0 = float(np.interp(NEPAL_UNIFICATION_YEAR, d.year, d.population))
        st.caption(
            f"About {p0 / 1e6:.1f} million people around {NEPAL_UNIFICATION_YEAR}; "
            f"{d.population.iloc[-1] / 1e6:.1f} million by {int(d.year.iloc[-1])} (HYDE / UN)."
        )
    if lu is not None:
        d = lu[lu.year.between(1700, TODAY.year)].copy()
        long = d.melt(
            id_vars="year", value_vars=["cropland_ha", "grazing_ha", "builtup_ha"], var_name="use", value_name="ha"
        )
        long["km2"] = long.ha / 100
        names = {"cropland_ha": "Cropland", "grazing_ha": "Grazing land", "builtup_ha": "Built-up land"}
        cols = st.columns(3)
        for col, (k, n) in zip(cols, names.items()):
            with col:
                dd = long[long.use == k]
                dd = dd[dd.year >= NEPAL_UNIFICATION_YEAR - 8]
                fig = charts.line(dd, "year", "km2", name=n, unit="km²", markers=True, title=n, height=260)
                common.chart_and_table(fig, dd[["year", "km2"]], key=f"tm_1768_{k}")
        provenance(
            "HYDE 3.3 land use (Klein Goldewijk et al.) via Our World in Data. Reconstructions before the "
            "1960s carry wide uncertainty."
        )
    if ur is not None:
        d = ur[ur.year.between(1950, TODAY.year)].copy()
        d["urban_share"] = d.urban / (d.urban + d.rural) * 100
        fig = charts.line(
            d,
            "year",
            "urban_share",
            name="Urban share",
            unit="% of people",
            markers=True,
            title="Share of Nepalis living in areas classed as urban",
            height=280,
        )
        common.chart_and_table(fig, d, key="tm_1768_urban")
        st.caption(
            "The steep rise around 2000–2015 is partly definitional: Nepal's 2017 restructuring turned many "
            "rural areas into municipalities, and the latest UN/World Bank series adopt that national "
            "definition (the 2021 census counts 66% of people in urban municipalities). The U-Net's *Urban land* "
            "class measures physical built-up area instead — see the Years view."
        )
    st.markdown("#### Timeline")
    left, right = st.columns(2, gap="large")
    half = (len(evs) + 1) // 2
    for col, chunk in ((left, evs[:half]), (right, evs[half:])):
        with col:
            for e in chunk:
                yrs = f"{e.year}–{e.end_year}" if e.end_year else str(e.year)
                st.markdown(
                    f"**{yrs} · {e.title}**  \n<span style='opacity:.75'>{e.detail} "
                    f"<i>({hist_events.CATEGORY_LABELS[e.category]})</i></span>",
                    unsafe_allow_html=True,
                )


{
    "day": day_view,
    "month": month_view,
    "year": year_view,
    "decade": decade_view,
    "century": century_view,
    "1768": since_1768_view,
}[scale or "year"]()
