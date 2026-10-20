"""Model lab: the paper's Streamlit demo (Sec. 3.6), faithfully reproduced and upgraded."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from monitor_nepal.config import CLASS_NAMES, MODEL_GSD_M, PALETTES, PAPER
from monitor_nepal.export import png_bytes
from monitor_nepal.geo import Grid
from monitor_nepal.http import FetchError
from monitor_nepal.imagery import esri
from monitor_nepal.model import MODELS, available_models
from monitor_nepal.postprocess import clean, majority_filter, paper_median, paper_morphology
from monitor_nepal.resample import cv2_linear
from monitor_nepal.ui import charts, common, data, widgets

common.header(
    "Model lab",
    f"This is the demo described in Section 3.6 of <i>{PAPER.title}</i> — upload a satellite image and the U-Net "
    "labels every pixel — rebuilt with what we learned while scaling it to the whole country: native-resolution "
    "inference, test-time augmentation, a fixed post-processing step, confidence maps and area statistics. "
    "The exact published behaviour is still one click away.",
    eyebrow="Research · paper demo",
)
settings = common.settings()
pal = np.array(PALETTES[settings["palette"]], np.uint8)

MAX_MODEL_PIXELS = 3000 * 3000  # model-grid pixels per image (≈16 × 16 km at 5.46 m)

GSD_PRESETS = {
    "unknown": "I don't know (use the paper's 224 × 224 resize)",
    "0.3": "0.3 m — very high resolution (WorldView-3, Google Earth zoom 19)",
    "0.5": "0.5 m — DeepGlobe / Maxar / Google Earth zoom 18",
    "1.0": "1 m — Esri / Google zoom 17",
    "2.0": "2 m — PlanetScope-like",
    "3.0": "3 m",
    "custom": "Custom…",
}


def _to_rgb(file) -> np.ndarray:
    im = Image.open(io.BytesIO(file.getvalue()))
    if im.mode in ("I;16", "I", "F") or (im.mode not in ("RGB", "RGBA", "L", "P")):
        arr = np.asarray(im).astype(np.float32)
        lo, hi = np.percentile(arr, (1, 99))
        arr = np.clip((arr - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
    return np.asarray(im.convert("RGB"))


# ---- inputs -----------------------------------------------------------------------------------------------
src = st.segmented_control(
    "Image source",
    ["upload", "nepal"],
    default="upload",
    key="ml_src",
    format_func={"upload": "Upload your own", "nepal": "Fetch a sample from Nepal"}.get,
)
images: list[tuple[str, np.ndarray, float | None]] = []
if src == "upload":
    files = st.file_uploader(
        "RGB satellite image(s)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True,
        key="ml_files",
        help="Any size. For best results tell the app the pixel size below.",
    )
    c1, c2 = st.columns([1.4, 1])
    preset = c1.selectbox("Pixel size of your image", list(GSD_PRESETS), format_func=GSD_PRESETS.get, key="ml_gsd")
    gsd = None
    if preset == "custom":
        gsd = c2.number_input("Metres per pixel", 0.05, 30.0, 0.5, 0.05, key="ml_gsd_custom")
    elif preset != "unknown":
        gsd = float(preset)
    for f in files or []:
        try:
            images.append((f.name, _to_rgb(f), gsd))
        except Exception as exc:  # unreadable file
            st.error(f"Could not read {f.name}: {exc}")
else:
    c1, c2 = st.columns([1.4, 1])
    place = c1.selectbox("Place", widgets.PLACES, format_func=lambda p: p.name, key="ml_place")
    km = c2.select_slider(
        "Tile size",
        [0.6, 1.2, 2.4],
        value=1.2,
        format_func=lambda v: f"{v} km",
        key="ml_km",
        help="1.2 km matches one DeepGlobe tile (2448 px at 0.5 m).",
    )
    if st.button("Fetch imagery", icon=":material/satellite_alt:", key="ml_fetch"):
        try:
            with st.spinner("Fetching Esri World Imagery at 1 m…"):
                g = Grid.around(place.lon, place.lat, km * 1000, 1.0)
                scene = esri.current_source().fetch(g)
                info = data.current_capture(place.lon, place.lat)
                st.session_state["ml_sample"] = (place.name, scene.rgb, 1.0, info.text)
        except FetchError as exc:
            st.error(str(exc))
    if "ml_sample" in st.session_state:
        name, rgb, gsd_s, info = st.session_state["ml_sample"]
        images.append((name, rgb, gsd_s))
        st.caption(f"{name}: Esri World Imagery, {info}.")

c1, c2 = st.columns(2)
with c1:
    feed = st.radio(
        "How the image is fed to the network",
        ["paper", "native"],
        key="ml_feed",
        format_func={
            "paper": "As in the paper — resize the whole image to 224 × 224",
            "native": "Native scale — resample to 5.46 m/pixel and slide over it",
        }.get,
        help="The network learned objects at 5.46 m per pixel (DeepGlobe 0.5 m tiles resized 2448→224). "
        "Squashing a large or very detailed image into 224 px changes the size of everything in it.",
    )
with c2:
    post = st.radio(
        "Post-processing",
        ["paper", "paper_median", "improved", "none"],
        key="ml_post",
        format_func={
            "paper": "Paper: morphological cleanup",
            "paper_median": "Paper: morphology + median",
            "improved": "Improved: nearest-class cleanup + majority filter",
            "none": "None (raw network output)",
        }.get,
    )

if not images:
    st.info(
        "Upload an image or fetch a sample to begin. Try the same image with both feeding options — on large "
        "images the native scale is usually far better.",
        icon=":material/upload:",
    )
    st.stop()

def segment(model_key: str, rgb: np.ndarray, gsd: float | None, use_native: bool):
    """Run one model on one image with the chosen feeding and post-processing."""
    net = data.model(model_key)
    h, w = rgb.shape[:2]
    if use_native:
        scale = gsd / MODEL_GSD_M
        small = cv2_linear(rgb, max(16, int(round(w * scale))), max(16, int(round(h * scale))))
        proba = net.predict_proba(small, tta=settings["tta"])
        px_area_m2 = MODEL_GSD_M**2
    else:
        small = cv2_linear(rgb, 224, 224)  # exactly cv2.resize(img, (224, 224)) as in the published app
        proba = net.predict_window(small.astype(np.float32) / 255.0, tta=settings["tta"])
        px_area_m2 = (w * gsd / 224) * (h * gsd / 224) if gsd else None
    raw = proba.argmax(-1).astype(np.uint8)
    if post == "paper":
        out = paper_morphology(raw)
    elif post == "paper_median":
        out = paper_median(paper_morphology(raw))
    elif post == "improved":
        out = majority_filter(clean(raw, 64), 3)
    else:
        out = raw
    return raw, out, proba.max(-1), px_area_m2


others = [m.key for m in available_models() if m.key != settings["model"]]
compare = bool(others) and st.toggle(
    f"Compare with the {MODELS[others[0]].short} model" if others else "Compare models", key="ml_compare",
    help="Run both networks on the same image to see what the Nepal adaptation changes.")
st.caption(
    f"Model: **{MODELS[settings['model']].label}** · test-time augmentation "
    f"{'on (4 views)' if settings['tta'] > 1 else 'off'}."
)

summary = []
for name, rgb, gsd in images:
    st.divider()
    st.subheader(name, anchor=False)
    h, w = rgb.shape[:2]
    if feed == "native" and gsd is None:
        st.warning("Native scale needs the pixel size — falling back to the paper's 224 × 224 resize.")
    use_native = feed == "native" and gsd is not None
    if use_native and (w * gsd / MODEL_GSD_M) * (h * gsd / MODEL_GSD_M) > MAX_MODEL_PIXELS:
        side = int(np.sqrt(MAX_MODEL_PIXELS) * MODEL_GSD_M / 1000)
        st.warning(f"At {gsd} m per pixel this image covers {w * gsd / 1000:.1f} × {h * gsd / 1000:.1f} km — "
                   f"too large to segment here (limit ≈ {side} × {side} km). Crop it, or use the Land-cover "
                   "map page for large areas.")
        continue
    with st.spinner("Segmenting…"):
        raw, out, conf, px_area_m2 = segment(settings["model"], rgb, gsd, use_native)

    shown = np.asarray(Image.fromarray(rgb).resize((out.shape[1], out.shape[0]), Image.Resampling.LANCZOS))
    overlay = (0.55 * shown + 0.45 * pal[out]).astype(np.uint8)
    preview = rgb if max(h, w) <= 1400 else np.asarray(Image.fromarray(rgb).reduce(int(np.ceil(max(h, w) / 1400))))
    a, b, c, d = st.columns(4)
    a.image(preview, caption=f"Input · {w}×{h} px", width="stretch")
    b.image(overlay, caption="Overlay", width="stretch")
    c.image(pal[out], caption=f"{MODELS[settings['model']].short} · {out.shape[1]}×{out.shape[0]} px",
            width="stretch")
    d.image((conf * 255).astype(np.uint8), caption="Confidence", width="stretch", clamp=True)
    common.legend(settings["palette"])

    counts = np.bincount(out.ravel(), minlength=7)[:7]
    share = counts / counts.sum() * 100
    comp = pd.DataFrame({"class": CLASS_NAMES, "pixels": counts, "share": share})
    if px_area_m2:
        comp["area_ha"] = counts * px_area_m2 / 1e4
    if compare:
        with st.spinner(f"Running the {MODELS[others[0]].short} model…"):
            _, out2, conf2, _ = segment(others[0], rgb, gsd, use_native)
        share2 = np.bincount(out2.ravel(), minlength=7)[:7] / out2.size * 100
        x, y, z = st.columns(3)
        x.image(pal[out], caption=MODELS[settings["model"]].short, width="stretch")
        y.image(pal[out2], caption=MODELS[others[0]].short, width="stretch")
        agree = (out == out2).mean() * 100
        z.metric("Pixels where the two models agree", f"{agree:.0f}%", border=True)
        comp[f"share_{others[0]}"] = share2
    left, right = st.columns([1.3, 1])
    with left:
        fig = charts.composition_bar(share, include_unknown=True)
        common.chart_and_table(fig, comp.round(2), key=f"ml_comp_{name}")
    with right:
        changed = float((out != raw).mean() * 100)
        msgs = [f"Post-processing changed **{changed:.1f}%** of pixels."]
        if post in ("paper", "paper_median"):
            fake_urban = float(((out == 0) & (raw != 0)).mean() * 100)
            if fake_urban > 0.05:
                msgs.append(
                    f"⚠️ **{fake_urban:.1f}%** of pixels became *Urban land* only because the published "
                    "cleanup assigns removed specks to class 0. The *Improved* option fixes this."
                )
        msgs.append(f"Mean confidence {conf.mean() * 100:.0f}%.")
        st.markdown("\n\n".join(msgs))
        st.download_button(
            "Land-cover PNG",
            png_bytes(pal[out]),
            file_name=f"{name}_landcover.png",
            mime="image/png",
            key=f"ml_dl1_{name}",
            icon=":material/download:",
        )
        st.download_button(
            "Overlay PNG",
            png_bytes(overlay),
            file_name=f"{name}_overlay.png",
            mime="image/png",
            key=f"ml_dl2_{name}",
            icon=":material/download:",
        )
    summary.append({"image": name, **{CLASS_NAMES[i]: round(float(share[i]), 1) for i in range(7)}})

if len(summary) > 1:
    st.divider()
    st.subheader("All images", anchor=False)
    st.dataframe(pd.DataFrame(summary), hide_index=True)

with st.expander("What changed since the published demo, and why", icon=":material/lightbulb:"):
    st.markdown("""
- **Resizing matters more than expected.** The training notebook shrank each 2448-px DeepGlobe tile to 224 px with
  `cv2.resize` (bilinear, no antialiasing). That keeps a sparse, sharp sample of the 0.5 m texture — and the network
  relies on it: on DeepGlobe tiles, forest IoU is 0.58 with OpenCV's resize but **0.00** with an antialiased resize.
  This app re-implements OpenCV's sampling exactly (max difference 1 grey level).
- **Scale matters.** The network sees the world at 5.46 m per pixel. Uploading a city-sized screenshot and
  squashing it to 224 px shrinks every building and field; the *native scale* option keeps them the right size.
- **A post-processing quirk.** The published cleanup never re-assigns the specks it removes, so they fall back to
  class 0 — *Urban land*. The *Improved* option gives each removed speck the class of the nearest surviving region,
  then applies a proper multi-class majority filter. The paper options here reproduce `app_alpha.py` bit-for-bit
  (verified against scikit-image 0.21).
- **Exact weights.** The model is the paper's `deepglobe_unet_jay.keras`, exported to ONNX with dynamic image size
  (max difference 2.4 × 10⁻⁶ in the softmax, identical predictions), so the app no longer needs TensorFlow.
""")
