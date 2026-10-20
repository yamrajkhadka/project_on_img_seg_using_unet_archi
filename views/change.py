"""Change detection: classify the same place at two dates and measure what changed."""

from __future__ import annotations

import folium
import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage
from streamlit_folium import st_folium

from monitor_nepal import imagery
from monitor_nepal.analysis import pipeline, stats
from monitor_nepal.config import CLASS_NAMES, MODEL_GSD_M, PALETTES
from monitor_nepal.geo import Grid
from monitor_nepal.http import FetchError
from monitor_nepal.ui import charts, common, components, data, maps, widgets

common.header(
    "Change detection",
    "Classify the same place at two moments and see exactly what changed: where farmland became city, forest was "
    "cleared or a river moved. Both dates are mapped on an identical 5.5 m grid, so every pixel is compared "
    "like-for-like.",
    eyebrow="Monitor",
)
settings = common.settings()
palette = settings["palette"]
pal = np.array(PALETTES[palette], np.uint8)

c_area, c_before, c_after = st.columns([1.1, 1, 1], gap="large")
with c_area:
    st.markdown("**Where**")
    place = st.selectbox("Place", widgets.PLACES, index=1, format_func=lambda p: p.name, key="ch_place")
    use_custom = st.toggle("Use my own coordinates", key="ch_custom")
    lon, lat = place.lon, place.lat
    if use_custom:
        a, b = st.columns(2)
        lat = a.number_input("Latitude", 26.3, 30.5, place.lat, step=0.001, format="%.4f", key="ch_lat")
        lon = b.number_input("Longitude", 80.0, 88.3, place.lon, step=0.001, format="%.4f", key="ch_lon")
    size_km = st.slider("Window size", 1.0, 6.0, 3.0, 0.5, format="%.1f km", key="ch_size")
    if st.button(
        "List image dates available here",
        icon=":material/event:",
        key="ch_versions",
        help="Asks the Wayback archive which releases actually changed at this spot.",
    ):
        try:
            with st.spinner("Querying the Wayback archive…"):
                vers = data.local_versions(lon, lat)
                rows = []
                for v in vers[:12]:
                    info = data.capture_info(lon, lat, v.metadata_url)
                    rows.append(
                        {
                            "release": v.date,
                            "captured": info.date,
                            "sensor": info.sensor,
                            "resolution_m": info.resolution_m,
                        }
                    )
                st.session_state["ch_versions_tbl"] = pd.DataFrame(rows)
        except FetchError as exc:
            st.error(str(exc))
    if "ch_versions_tbl" in st.session_state:
        st.dataframe(
            st.session_state["ch_versions_tbl"],
            hide_index=True,
            height=220,
            column_config={
                "release": st.column_config.DateColumn("Wayback release"),
                "captured": st.column_config.DateColumn("Captured"),
                "resolution_m": st.column_config.NumberColumn("m", format="%.2f"),
            },
        )
with c_before:
    st.markdown("**Before**")
    spec_a = widgets.imagery_picker("ch_a", kinds=("wayback", "eox", "s2", "landsat"), default="wayback", compact=True)
with c_after:
    st.markdown("**After**")
    spec_b = widgets.imagery_picker(
        "ch_b", kinds=("esri", "wayback", "eox", "s2", "landsat"), default="esri", compact=True
    )

common.model_source_warning(settings["model"], spec_a.kind, spec_b.kind)
if spec_a.kind in ("esri", "wayback") and spec_b.kind not in ("esri", "wayback"):
    st.caption(
        "Tip: comparing sub-metre imagery with Sentinel-2/Landsat mixes sensor differences into the change. "
        "For the cleanest result compare like with like (Wayback ↔ latest Esri, or EOX year ↔ EOX year)."
    )

if st.button("Detect change", type="primary", icon=":material/compare:", key="ch_run"):
    grid = Grid.around(lon, lat, size_km * 1000, MODEL_GSD_M)
    prog = st.progress(0.0, "Before…")
    try:
        ra = data.classify_area(
            grid,
            spec_a,
            settings["model"],
            settings["tta"],
            pipeline.PAPER_MIN_OBJECT_PX,
            _progress=lambda f, t: prog.progress(0.5 * f, f"Before · {t}"),
        )
        rb = data.classify_area(
            grid,
            spec_b,
            settings["model"],
            settings["tta"],
            pipeline.PAPER_MIN_OBJECT_PX,
            _progress=lambda f, t: prog.progress(0.5 + 0.5 * f, f"After · {t}"),
        )
        st.session_state["ch_result"] = (ra, rb)
    except FetchError as exc:
        st.error(str(exc), icon=":material/cloud_off:")
    prog.empty()

res = st.session_state.get("ch_result")
if not res:
    st.info(
        "Choose a place and two dates, then press **Detect change**. A good first try: *Kathmandu Valley east "
        "(Bhaktapur)* — the oldest Wayback release (2014) against today's imagery.",
        icon=":material/info:",
    )
    st.stop()

ra, rb = res
grid = ra.grid
valid = ra.scene.valid & rb.scene.valid
a, b = ra.classes, rb.classes
changed = (a != b) & valid
trans = stats.transitions(a, b, grid, valid)
total = float(trans.values.sum())
before_km2, after_km2 = trans.sum(1), trans.sum(0)
net = pd.DataFrame({"class": CLASS_NAMES, "before_km2": before_km2.values, "after_km2": after_km2.values})
net["change"] = net.after_km2 - net.before_km2


def when(r) -> str:
    if r.spec.kind in ("esri", "wayback"):
        try:
            lon_c, lat_c = r.grid.center_lonlat
            if r.spec.kind == "esri":
                info = data.current_capture(lon_c, lat_c)
            else:
                rel = next(x for x in data.wayback_releases() if x.number == r.spec.release)
                info = data.capture_info(lon_c, lat_c, rel.metadata_url)
            if info.date:
                return info.date.strftime("%d %b %Y")
        except (FetchError, StopIteration):
            pass
    return r.scene.acquired or r.spec.label()


lab_a, lab_b = f"Before · {when(ra)}", f"After · {when(rb)}"
m1, m2, m3, m4 = st.columns(4)
m1.metric("Area compared", common.fmt_km2(total), border=True)
m2.metric(
    "Changed",
    f"{changed.sum() / max(1, valid.sum()) * 100:.1f}%",
    common.fmt_km2(float((changed * grid.pixel_area_km2()).sum())),
    delta_color="off",
    delta_arrow="off",
    border=True,
)
land = net[net["class"] != "Unknown"]
gain, loss = land.sort_values("change").iloc[-1], land.sort_values("change").iloc[0]
m3.metric("Biggest gain", gain["class"], f"{gain.change:+.2f} km²", delta_color="off", border=True)
m4.metric("Biggest loss", loss["class"], f"{loss.change:+.2f} km²", delta_color="off", border=True)

t1, t2, t3 = st.tabs(["Imagery & classes", "Where it changed", "How much"], key="ch_tabs", on_change="rerun")
if t1.open:
    with t1:
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("**Imagery** — drag the handle")
            components.swipe(ra.scene.rgb, rb.scene.rgb, lab_a, lab_b)
        with right:
            st.markdown("**Land cover** — drag the handle")
            components.swipe(pal[a], pal[b], lab_a, lab_b)
        common.legend(palette)
if t2.open:
    with t2:
        show = st.segmented_control(
            "Highlight",
            ["any", "to_urban", "forest_loss", "water"],
            default="any",
            key="ch_highlight",
            format_func={
                "any": "Any change",
                "to_urban": "New built-up land",
                "forest_loss": "Forest loss",
                "water": "Water changes",
            }.get,
        )
        mask = changed.copy()
        if show == "to_urban":
            mask &= b == 0
        elif show == "forest_loss":
            mask &= a == 3
        elif show == "water":
            mask &= (a == 4) | (b == 4)
        rgba = maps.class_rgba(b, palette, valid=mask)
        lon_c, lat_c = grid.center_lonlat
        m = maps.base_map(center=(lat_c, lon_c), zoom=14 if size_km <= 3 else 13, labels=False)
        src_b = imagery.resolve(rb.spec, grid.bounds_lonlat)
        folium.TileLayer(
            src_b.url_template,
            name=f"After imagery · {rb.spec.label()}",
            attr=src_b.attribution,
            max_native_zoom=src_b.max_zoom,
            max_zoom=19,
            overlay=True,
        ).add_to(m)
        maps.add_raster(m, rgba, grid, name="Changed pixels (coloured by new class)", opacity=0.85)
        maps.add_box(m, grid)
        st_folium(maps.finish(m), height=520, use_container_width=True, returned_objects=[], key="ch_map")
        st.caption(
            "Coloured pixels changed class; the colour is the class they changed **to**. Unchanged areas show "
            "the imagery."
        )
        labels, nlab = ndimage.label(mask, structure=np.ones((3, 3)))
        if nlab:
            sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, nlab + 1))
            order = np.argsort(sizes)[::-1][:8]
            px_km2 = grid.ground_res**2 / 1e6
            rows = []
            for idx in order:
                comp = labels == (idx + 1)
                rr, cc = np.nonzero(comp)
                lo, la = grid.pixel_lonlat(rr.mean(), cc.mean())
                fa = np.bincount(a[comp], minlength=7).argmax()
                fb = np.bincount(b[comp], minlength=7).argmax()
                rows.append(
                    {
                        "from": CLASS_NAMES[fa],
                        "to": CLASS_NAMES[fb],
                        "area_ha": sizes[idx] * px_km2 * 100,
                        "lat": float(la),
                        "lon": float(lo),
                    }
                )
            st.markdown("**Largest change patches**")
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                column_config={
                    "area_ha": st.column_config.NumberColumn("Area (ha)", format="%.2f"),
                    "lat": st.column_config.NumberColumn(format="%.5f"),
                    "lon": st.column_config.NumberColumn(format="%.5f"),
                },
            )
if t3.open:
    with t3:
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            fig = charts.net_change(net, title="Net change by class (km²)")
            common.chart_and_table(fig, net.round(4), key="ch_net")
        with right:
            fig = charts.sankey(trans, title="Where each class went", min_value=total * 0.002)
            tbl = trans.round(4).reset_index().rename(columns={"index": "before \\ after"})
            common.chart_and_table(fig, tbl, key="ch_sankey")
st.caption(
    "Change maps inherit the model's errors from both dates, and differences in sensor, season or "
    "shadows can look like change. Treat small scattered changes with caution; large coherent patches are "
    "the reliable signal."
)
