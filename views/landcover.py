"""Land-cover map: run the U-Net on any place in Nepal, from any imagery source and date."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from shapely import contains_xy
from streamlit_folium import st_folium

from monitor_nepal import boundaries, imagery
from monitor_nepal.analysis import pipeline, stats
from monitor_nepal.config import CLASS_KEYS, CLASS_NAMES, MODEL_GSD_M, PALETTES
from monitor_nepal.export import bundle
from monitor_nepal.geo import Grid
from monitor_nepal.http import FetchError
from monitor_nepal.imagery import reference
from monitor_nepal.model import MODELS
from monitor_nepal.ui import charts, common, data, maps, widgets

common.header(
    "Land-cover map",
    "Classify every pixel of any place in Nepal with the paper's U-Net — at the 5.5 m scale it was trained on — "
    "from sub-metre Esri imagery, the Wayback archive back to 2014, Sentinel-2 since 2015 or Landsat since 1982. "
    "Whole districts and provinces are estimated statistically from random sample windows.",
    eyebrow="Monitor",
)
settings = common.settings()

MAX_KM = {"esri": 6.0, "wayback": 6.0, "eox": 12.0, "s2": 12.0, "landsat": 15.0, "mss": 15.0}

# ---------------------------------------------------------------------------------------------------------------
ctl_area, ctl_img = st.columns([1.35, 1], gap="large")
with ctl_area:
    mode = st.segmented_control(
        "Area",
        ["place", "click", "coords", "admin"],
        default="place",
        key="lc_mode",
        format_func={
            "place": "Famous places",
            "click": "Click on a map",
            "coords": "Coordinates",
            "admin": "District / municipality",
        }.get,
    )
    center, region = None, None
    if mode == "place":
        place = st.selectbox("Place", widgets.PLACES, format_func=lambda p: f"{p.name} — {p.why}", key="lc_place")
        center, default_km = (place.lon, place.lat), place.km
    elif mode == "click":
        st.caption("Click anywhere in Nepal; the analysis window is centred on your click.")
        pick = maps.base_map(center=(28.2, 84.2), zoom=7, basemap="imagery")
        maps.add_boundaries(pick, 1, name="Provinces")
        last = st.session_state.get("lc_clicked")
        if last:
            import folium

            folium.Marker([last[1], last[0]]).add_to(pick)
        out = st_folium(
            maps.finish(pick),
            height=330,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key="lc_pick_map",
        )
        if out and out.get("last_clicked"):
            st.session_state["lc_clicked"] = (out["last_clicked"]["lng"], out["last_clicked"]["lat"])
        center, default_km = st.session_state.get("lc_clicked"), 3.0
        if center:
            st.caption(f"Selected {center[1]:.4f}° N, {center[0]:.4f}° E")
    elif mode == "coords":
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", 26.3, 30.5, 27.7172, step=0.001, format="%.4f", key="lc_lat")
        lon = c2.number_input("Longitude", 80.0, 88.3, 85.3240, step=0.001, format="%.4f", key="lc_lon")
        center, default_km = (lon, lat), 3.0
    else:
        region = widgets.region_picker("lc_region", label="Administrative unit", default="NP0439101")
        default_km = 0

with ctl_img:
    spec = widgets.imagery_picker("lc_img", default="esri")
    common.model_source_warning(settings["model"], spec.kind)
    max_km = MAX_KM[spec.kind]
    if mode != "admin":
        size_km = st.slider(
            "Window size",
            1.0,
            max_km,
            float(min(max(default_km, 1.0), max_km)),
            0.5,
            format="%.1f km",
            key="lc_size",
            help=f"Square window. High-resolution sources are capped at {MAX_KM['esri']:.0f} km to "
            "keep tile downloads reasonable.",
        )

# decide map vs survey for administrative units
survey_mode = False
if mode == "admin" and region is not None:
    lon0, lat0, lon1, lat1 = boundaries.geometry(region).bounds
    width_km = (lon1 - lon0) * 111.32 * np.cos(np.radians((lat0 + lat1) / 2))
    height_km = (lat1 - lat0) * 110.57
    survey_mode = max(width_km, height_km) > max_km
    if survey_mode:
        n = st.slider(
            "Sample windows",
            20,
            200,
            40 if region.level >= 2 else 60,
            10,
            key="lc_n",
            help="Each window is 1.2 × 1.2 km (one model tile). More windows → narrower confidence intervals.",
        )
        st.caption(
            f"{region.short} spans {width_km:.0f} × {height_km:.0f} km — larger than a wall-to-wall window — so it "
            f"is estimated from {n} random sample windows with 95% confidence intervals."
        )

run = st.button(
    "Classify",
    type="primary",
    icon=":material/play_arrow:",
    key="lc_run",
    disabled=(mode in ("click",) and center is None),
)

# ---------------------------------------------------------------------------------------------------------------
if run:
    prog = st.progress(0.0, text="Starting…")

    def report(f: float, text: str) -> None:
        prog.progress(min(1.0, max(0.0, f)), text=text)

    try:
        if survey_mode:
            res = data.survey(region.code, spec, settings["model"], n, 0, settings["tta"], _progress=report)
            st.session_state["lc_result"] = ("survey", res)
        else:
            if mode == "admin":
                lon0, lat0, lon1, lat1 = boundaries.geometry(region).bounds
                grid = Grid.from_lonlat_bbox((lon0, lat0, lon1, lat1), MODEL_GSD_M)
            else:
                grid = Grid.around(center[0], center[1], size_km * 1000, MODEL_GSD_M)
            res = data.classify_area(
                grid, spec, settings["model"], settings["tta"], pipeline.PAPER_MIN_OBJECT_PX, _progress=report
            )
            mask = None
            if mode == "admin":
                rows, cols = np.mgrid[0 : grid.height, 0 : grid.width]
                lon, lat = grid.pixel_lonlat(rows, cols)
                mask = contains_xy(boundaries.geometry(region), lon, lat)
            st.session_state["lc_result"] = ("map", res, mask, region)
        prog.empty()
    except FetchError as exc:
        prog.empty()
        st.error(str(exc), icon=":material/cloud_off:")

result = st.session_state.get("lc_result")
if not result:
    st.info(
        "Pick an area and an imagery source, then press **Classify**. Tip: start with *Kathmandu core* on the "
        "latest Esri imagery, then switch to a 2014 Wayback release to see the city grow.",
        icon=":material/info:",
    )
    st.stop()

palette = settings["palette"]
pal = np.array(PALETTES[palette], np.uint8)

# ---------------------------------------------------------------------------------------------------------------
if result[0] == "map":
    _, res, mask, reg = result
    grid, scene = res.grid, res.scene
    valid = scene.valid if mask is None else (scene.valid & mask)
    comp = stats.composition(res.classes, grid, valid)
    lon_c, lat_c = grid.center_lonlat

    # capture date for high-resolution sources
    capture = None
    try:
        if res.spec.kind == "esri":
            capture = data.current_capture(lon_c, lat_c)
        elif res.spec.kind == "wayback":
            rel = next(r for r in data.wayback_releases() if r.number == res.spec.release)
            capture = data.capture_info(lon_c, lat_c, rel.metadata_url)
    except (FetchError, StopIteration):
        capture = None

    land = comp[comp.key != "unknown"]
    top = land.sort_values("share", ascending=False).iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Area analysed",
        common.fmt_km2(float(comp.km2.sum())),
        border=True,
        help=f"{grid.width} × {grid.height} pixels at {grid.ground_res:.2f} m.",
    )
    m2.metric(
        "Dominant cover",
        top["class"],
        f"{top.share:.0f}% of the area",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    conf = float(res.confidence[valid].astype(np.float32).mean()) if valid.any() else float("nan")
    m3.metric(
        "Mean model confidence",
        f"{conf * 100:.0f}%",
        border=True,
        help="Average of the highest class probability per pixel (not an accuracy).",
    )
    when = capture.text if capture else (scene.acquired or "—")
    m4.metric(
        "Imagery",
        res.spec.short_label(),
        when,
        delta_color="off",
        delta_arrow="off",
        border=True,
        help=f"{res.spec.label()} · {when}",
    )

    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.65, 0.05, key="lc_opacity")
    src = imagery.resolve(res.spec, grid.bounds_lonlat)
    m = maps.base_map(center=(lat_c, lon_c), zoom=14 if grid.ground_size_km[0] < 4 else 13, labels=False)
    import folium

    if res.spec.kind != "esri":  # the basemap already is the latest Esri imagery
        folium.TileLayer(
            src.url_template,
            name=f"Analysed imagery · {res.spec.label()}",
            attr=src.attribution,
            max_native_zoom=src.max_zoom,
            max_zoom=19,
            overlay=True,
            show=True,
        ).add_to(m)
    maps.add_raster(
        m, maps.class_rgba(res.classes, palette, valid=valid), grid, name="Land cover (U-Net)", opacity=opacity
    )
    maps.add_box(m, grid)
    if reg is not None:
        maps.add_region(m, reg)
    st_folium(maps.finish(m), height=560, use_container_width=True, returned_objects=[], key="lc_map")
    common.legend(palette)

    t_side, t_comp, t_ref, t_dl = st.tabs(
        ["Side by side", "Composition", "Check against a reference map", "Download"], key="lc_tabs", on_change="rerun"
    )
    if t_side.open:
        with t_side:
            a, b, c = st.columns(3)
            a.image(
                scene.rgb, caption=f"Input imagery at {grid.ground_res:.2f} m (as the model sees it)", width="stretch"
            )
            b.image(
                np.where(valid[..., None], pal[res.classes], 0).astype(np.uint8),
                caption="Land cover (post-processed)",
                width="stretch",
            )
            conf_img = (res.confidence.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
            c.image(conf_img, caption="Confidence (brighter = more certain)", width="stretch", clamp=True)
            changed = float((res.raw != res.classes)[valid].mean() * 100) if valid.any() else 0
            st.caption(
                f"Post-processing removed regions smaller than {pipeline.PAPER_MIN_OBJECT_PX} px "
                f"(≈{pipeline.PAPER_MIN_OBJECT_PX * grid.ground_res**2 / 1e4:.2f} ha) and gave them the surrounding "
                f"class: {changed:.1f}% of pixels changed. Model: {MODELS[res.model_key].label}."
            )
    if t_comp.open:
        with t_comp:
            fig = charts.composition_bar(comp.share.values, title="Share of the analysed area")
            tbl = comp[comp.key != "unknown"][["class", "km2", "share"]]
            common.chart_and_table(
                fig,
                tbl.round(3),
                key="lc_comp",
                column_config={
                    "km2": st.column_config.NumberColumn("Area (km²)", format="%.3f"),
                    "share": st.column_config.NumberColumn("Share (%)", format="%.1f"),
                },
            )
    if t_ref.open:
        with t_ref:
            st.caption(
                "Compare this map with Impact Observatory's independent 10 m annual land-cover map (2017 onwards), "
                "translated into the paper's seven classes. Disagreement is not automatically the U-Net's error — "
                "the reference is itself a model (~75–85% accurate) — but large gaps are informative."
            )
            year = st.select_slider(
                "Reference year", list(reference.years()), value=reference.years()[-1], key="lc_ref_year"
            )
            if st.button("Compare", key="lc_ref_go", icon=":material/fact_check:"):
                try:
                    with st.spinner("Fetching the reference map…"):
                        rgrid = grid.with_resolution(max(10.0, grid.ground_res))
                        _, ref_dg = reference.fetch_io(rgrid, year)
                        from monitor_nepal.imagery.base import resample_array

                        ref_on_grid = resample_array(ref_dg, rgrid, grid, nearest=True)
                        ref_on_grid[~valid] = 255
                        st.session_state["lc_ref"] = (year, ref_on_grid)
                except FetchError as exc:
                    st.error(str(exc))
            if "lc_ref" in st.session_state and st.session_state["lc_ref"][1].shape == res.classes.shape:
                year, ref_on_grid = st.session_state["lc_ref"]
                agr = stats.agreement(res.classes, ref_on_grid)
                if agr.get("n"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pixel agreement", f"{agr['overall'] * 100:.0f}%", border=True)
                    c2.metric(
                        "Cohen's kappa",
                        f"{agr['kappa']:.2f}",
                        border=True,
                        help="Agreement beyond chance: 0 = chance, 1 = perfect.",
                    )
                    c3.metric("Reference year", str(year), border=True)
                    x, y = st.columns(2)
                    x.image(
                        pal[np.where(ref_on_grid == 255, 6, ref_on_grid)],
                        caption=f"Reference {year} (IO, 10 m)",
                        width="stretch",
                    )
                    y.image(pal[res.classes], caption="U-Net", width="stretch")
                    cm = pd.DataFrame(agr["confusion"][:6, :6], index=CLASS_NAMES[:6], columns=CLASS_NAMES[:6])
                    share = cm.div(cm.sum(1).replace(0, np.nan), axis=0) * 100
                    fig = charts.heatmap(
                        share.round(1), title="Where does the reference class end up? (% of row)", unit="%", height=330
                    )
                    fig.update_xaxes(title="U-Net says")
                    fig.update_yaxes(title="Reference says")
                    common.chart_and_table(fig, agr["per_class"].round(3), key="lc_ref_cm")
                else:
                    st.warning("The reference has no data here.")
    if t_dl.open:
        with t_dl:
            name = f"monitor_nepal_{res.spec.kind}_{lat_c:.3f}N_{lon_c:.3f}E"
            meta = {
                "model": MODELS[res.model_key].label,
                "imagery": res.spec.label(),
                "acquired": when,
                "center_lonlat": [lon_c, lat_c],
                "paper": "https://doi.org/10.3126/joeis.v5i2.97825",
            }
            classes_out = np.where(valid, res.classes, 6).astype(np.uint8)
            st.download_button(
                "Download GeoTIFF + PNG + CSV (zip)",
                bundle(
                    name,
                    classes=classes_out,
                    rgb=scene.rgb,
                    grid=grid,
                    composition_csv=comp.to_csv(index=False),
                    meta=meta,
                    palette=palette,
                ),
                file_name=f"{name}.zip",
                mime="application/zip",
                icon=":material/download:",
                type="primary",
            )
            st.caption(
                "GeoTIFFs are in Web Mercator (EPSG:3857) with the class colours embedded; they open directly in "
                "QGIS, ArcGIS or Google Earth Engine. Class ids: "
                + ", ".join(f"{i} = {n}" for i, n in enumerate(CLASS_NAMES))
                + "."
            )

# ---------------------------------------------------------------------------------------------------------------
else:
    _, res = result
    est = res.estimate
    ok = res.samples[res.samples.error.isna()]
    st.subheader(f"{res.region.label}", anchor=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Region area", common.fmt_km2(res.region.area_km2), border=True)
    c2.metric(
        "Sample windows used",
        f"{len(ok)} of {len(res.samples)}",
        border=True,
        help="Windows without imagery are dropped.",
    )
    c3.metric("Imagery", res.spec.short_label(), border=True, help=res.spec.label())
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        fig = charts.composition_ci(est, title="Estimated land cover with 95% confidence intervals")
        common.chart_and_table(
            fig, est[["class", "share", "low", "high", "km2", "km2_low", "km2_high"]].round(2), key="lc_survey"
        )
    with right:
        m = maps.base_map(center=(res.region.lat, res.region.lon), zoom=8 if res.region.level <= 2 else 11)
        maps.add_region(m, res.region)
        import folium

        for r in ok.itertuples():
            k = CLASS_KEYS.index(r.dominant)
            col = "#%02x%02x%02x" % tuple(PALETTES[palette][k])
            folium.CircleMarker(
                (r.lat, r.lon),
                radius=6,
                color="#ffffff",
                weight=1.5,
                fill=True,
                fill_color=col,
                fill_opacity=0.95,
                tooltip=f"Window {r.id}: mostly {CLASS_NAMES[k]} ({getattr(r, r.dominant) * 100:.0f}%)",
            ).add_to(m)
        st_folium(maps.finish(m), height=430, use_container_width=True, returned_objects=[], key="lc_survey_map")
        common.legend(palette, include_unknown=False)
    if res.thumbnails:
        st.markdown("**A look inside some sample windows** (imagery and the model's classes)")
        cols = st.columns(6)
        for i, (sid, (rgb, cls)) in enumerate(list(res.thumbnails.items())[:12]):
            with cols[i % 6]:
                st.image(np.concatenate([rgb, pal[cls]], 0), caption=f"Window {sid}", width="stretch")
    st.caption(
        "How this works: windows are placed uniformly at random inside the region, each classified at 5.5 m. "
        "The class share is the mean over windows; the interval is mean ± t·SE. This is the standard "
        "sample-based area estimation used in national forest inventories."
    )
