"""Methods, data and the paper behind the app."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitor_nepal.config import CLASSES, MODEL_GSD_M, MODELS_DIR, PAPER, PAPER_IOU, PAPER_MEAN_IOU
from monitor_nepal.model import MODELS
from monitor_nepal.ui import charts, common

common.header(
    "Methods, data & paper",
    "How Monitor Nepal turns one research model into a national monitoring system — and how far to trust "
    "each number it shows.",
    eyebrow="Research",
)

tab_paper, tab_model, tab_nepal, tab_stats, tab_data, tab_limits = st.tabs(
    ["The paper", "The model", "Adapting it to Nepal", "Statistics", "Data sources", "Limitations"], key="me_tabs"
)

with tab_paper:
    st.subheader(PAPER.title, anchor=False)
    st.markdown(
        f"**{PAPER.authors}** · {PAPER.affiliation}  \n*{PAPER.journal}* {PAPER.volume}({PAPER.issue}): "
        f"{PAPER.pages}, {PAPER.year} · DOI [{PAPER.doi}](https://doi.org/{PAPER.doi})"
    )
    with st.container(horizontal=True):
        st.link_button("Read the paper", PAPER.url, icon=":material/description:")
        st.link_button("PDF", PAPER.pdf_url, icon=":material/picture_as_pdf:")
        st.link_button("Code & data", PAPER.code_url, icon=":material/code:")
    st.markdown("""
The paper designs a U-Net from scratch for pixel-level land-cover classification, trains it on the DeepGlobe Land
Cover dataset with a composite loss (focal Tversky + weighted categorical cross-entropy) to cope with class imbalance,
cleans the output with morphological post-processing, and deploys it as a Streamlit app. It closes by naming the next
step: fine-tuning for Nepal, held back by the lack of labelled Nepalese imagery. **This app is that next step** — the
same network, made to monitor the whole country.""")
    st.markdown("**Cite**")
    st.code(PAPER.citation, language=None, wrap_lines=True)
    st.code(PAPER.bibtex, language="bibtex")

with tab_model:
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        st.markdown(f"""
**Architecture.** Encoder 32-64-128-256 filters, bottleneck 512 with dropout 0.5, decoder with transposed convolutions
and skip connections, each block *Conv3×3+ReLU → BatchNorm → Conv3×3+ReLU → BatchNorm*, softmax over 7 classes.
**7,772,071 parameters.** Input 224 × 224 RGB scaled to [0, 1].

**Training (paper).** 803 DeepGlobe tiles (2448 px at 0.5 m) resized to 224 px — so the network sees
**{MODEL_GSD_M:.2f} m per pixel** — with flips, brightness/contrast, shift-scale-rotate and elastic augmentation;
loss = 0.4 × weighted cross-entropy + 0.6 × focal Tversky (α 0.7, β 0.3, γ 0.75); Adam; early stopping.

**Runtime here.** The Keras weights are exported to ONNX with dynamic image size (largest softmax difference
2.4 × 10⁻⁶; identical class maps), so the app runs on ONNX Runtime without TensorFlow. Large areas are processed in
512-px windows with 64-px overlaps blended by a linear ramp, so no seams appear.""")
    with b:
        iou = pd.DataFrame({"class": [c.name for c in CLASSES], "IoU (paper)": [PAPER_IOU[c.key] for c in CLASSES]})
        st.dataframe(
            iou,
            hide_index=True,
            column_config={"IoU (paper)": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)},
        )
        st.caption(
            f"Mean IoU {PAPER_MEAN_IOU:.3f} as reported (Table 3). In the training notebook these figures are "
            "computed over all 803 tiles (training + validation split); the validation mean IoU logged during "
            "training was about 0.60."
        )
    st.markdown("#### A finding that matters for every user of this model")
    st.markdown("""
Reproducing the paper's numbers on DeepGlobe tiles revealed that the network depends on **how the image is shrunk**.
The notebook used `cv2.resize` (bilinear without antialiasing), which keeps a sharp, sparse sample of the 0.5 m
texture. On 40 DeepGlobe tiles:""")
    st.dataframe(
        pd.DataFrame(
            {
                "Resize used to make the 224-px input": [
                    "cv2 bilinear (as in training)",
                    "cv2 area (antialiased)",
                    "Pillow bilinear (antialiased)",
                ],
                "Forest IoU": [0.583, 0.000, 0.000],
                "Water IoU": [0.867, 0.716, 0.648],
                "Mean IoU (6 land classes)": [0.623, 0.459, 0.387],
            }
        ),
        hide_index=True,
    )
    st.markdown("""
The same test shows that ~1 m source imagery works as well as 0.5 m (mean IoU 0.628), 2 m loses most forest (0.35) and
3 m loses it entirely. So the app fetches Esri imagery at about 1 m and samples it down to 5.46 m exactly the way
training did (a NumPy re-implementation of OpenCV's sampling, max difference one grey level). It also fixes a small
post-processing quirk: the published cleanup assigns every removed speck to class 0 (*Urban*); here removed specks
take the class of the nearest surviving region.""")

with tab_nepal:
    st.markdown("""
#### From DeepGlobe to Nepal
Even with the training-faithful pipeline, the original network struggles in Nepal: tested on 70 random sites against
an independent 10 m map it agreed on about a third of pixels, under-calling forest and mistaking shaded mountain forest
for water; on Sentinel-2 mosaics it fails outright. This is the domain gap the paper anticipates (Sec. 4.4).

**The missing dataset, built automatically** (`scripts/build_nepal_dataset.py`). Sites in all 77 districts plus
targeted urban and river sites; for each, 224 × 224 patches at 5.46 m from four sources — Esri World Imagery,
Sentinel-2 cloudless annual mosaics, Sentinel-2 seasonal mosaics and Landsat 8/9 mosaics — paired with *weak labels*
from the Impact Observatory 10 m land-cover map of the same year, translated to the seven DeepGlobe classes. Eleven
whole districts (Ilam, Sunsari, Dhanusa, Lalitpur, Rasuwa, Kaski, Mustang, Rupandehi, Jumla, Surkhet, Kailali) are
held out, so validation is spatially independent.

**Consistent years.** The annual Sentinel-2 cloudless mosaics were processed differently from year to year (national
mean red ranges from 63 to 95), which a network reads as land-cover change. Before classification each year's colours
are mapped onto the 2018–2025 average with a per-channel gain and offset measured at random sites across Nepal
(relative radiometric normalisation); local contrasts are untouched. The 2017 mosaic is blank over Nepal and is not used.

**Fine-tuning** (`scripts/finetune_nepal.py`). Same architecture, initialised from the paper's weights, trained with
the paper's own composite loss (unlabelled pixels ignored), flips/rotations and colour jitter. The result is exported
both to ONNX (used here) and back to a `.keras` file loadable with Keras.""")
    log_path, eval_path = MODELS_DIR / "finetune_log.json", MODELS_DIR / "evaluation.json"
    if log_path.exists():
        tlog = json.loads(log_path.read_text())
        done, planned = len(tlog.get("epochs", [])), tlog.get("epochs_planned") or len(tlog.get("epochs", []))
        best_ep = tlog.get("best_epoch")
        best_val = tlog["epochs"][best_ep - 1]["val"]["all"]["miou"] if best_ep else None
        base_val = tlog["baseline"]["all"]["miou"]
        how = tlog.get("summary") or f"after epoch {best_ep}"
        st.info(f"**Training:** validation mean IoU on the held-out districts (against the weak labels) "
                f"{base_val:.3f} with the paper's weights → {best_val:.3f} — {how}." +
                (f" This is a preliminary checkpoint ({done} of {planned} planned epochs); the remaining epochs can "
                 "be run with `scripts/finetune_nepal.py` (it resumes automatically)." if done < planned else ""),
                icon=":material/model_training:")
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        st.markdown("#### Evaluation on held-out districts")
        st.caption(ev.get("description", ""))
        rows = []
        for r in ev.get("results", []):
            rows.append(
                {
                    "model": r["model"],
                    "imagery": r["source"],
                    "reference": r["reference"],
                    "agreement": r["overall"],
                    "kappa": r["kappa"],
                    "mean IoU": r["miou"],
                    "pixels": r["n"],
                }
            )
        if rows:
            res = pd.DataFrame(rows)
            ref_name = st.segmented_control(
                "Scored against", sorted(res.reference.unique()), default="ESA WorldCover 2021", key="me_ref",
                help="ESA WorldCover is independent of the fine-tuning labels (Impact Observatory), so it is the "
                     "fairer test of the Nepal-adapted model.") or "ESA WorldCover 2021"
            sub = res[res.reference == ref_name]
            sources = list(dict.fromkeys(sub.imagery))
            fig = go.Figure()
            muted, accent = "#b6b4ac" if not common.is_dark() else "#5f5e59", charts.single_color()
            for model_name, color in (("Original (paper)", muted), ("Nepal-adapted", accent)):
                m = sub[sub.model == model_name].set_index("imagery").reindex(sources)
                fig.add_bar(x=sources, y=m["agreement"] * 100, name=model_name, marker_color=color, width=0.32,
                            text=[f"{v:.0f}%" if pd.notna(v) else "" for v in m["agreement"] * 100],
                            textposition="outside", cliponaxis=False,
                            hovertemplate=f"{model_name} · %{{x}}: %{{y:.1f}}% agreement<extra></extra>")
            if ev.get("inter_reference"):
                ceiling = ev["inter_reference"]["overall"] * 100
                fig.add_hline(y=ceiling, line_width=1, line_color=charts.INK_MUTED,
                              annotation_text=f"The two reference maps agree on {ceiling:.0f}%",
                              annotation_position="top left", annotation=dict(font=dict(size=11, color=charts.INK_MUTED)))
            charts.style(fig, 340, hovermode="closest", title=f"Pixel agreement with {ref_name}, held-out districts")
            fig.update_layout(barmode="group", bargap=0.35)
            fig.update_yaxes(ticksuffix="%", range=[0, 100])
            common.chart_and_table(fig, sub, key="me_eval")
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                column_config={
                    "agreement": st.column_config.NumberColumn(format="%.3f"),
                    "kappa": st.column_config.NumberColumn(format="%.3f"),
                    "mean IoU": st.column_config.NumberColumn(format="%.3f"),
                    "pixels": st.column_config.NumberColumn(format="%d"),
                },
            )
        if ev.get("inter_reference"):
            st.caption(
                "For scale — the two reference maps agree with each other on "
                f"{ev['inter_reference']['overall'] * 100:.0f}% of the same pixels, the practical ceiling "
                "for agreement with either."
            )
    elif log_path.exists():
        st.json(json.loads(log_path.read_text()).get("best_epoch"))
    else:
        st.info("The Nepal-adapted model has not been trained in this deployment yet.")
    st.markdown("**Models available here**")
    for m in MODELS.values():
        st.markdown(f"- **{m.label}** — {m.description} {'' if m.available else '*(not installed)*'}")

with tab_stats:
    st.markdown(r"""
#### Wall-to-wall maps
For areas up to a few kilometres across, every pixel is classified. Areas use the true ground size of each pixel
row, $(r \cos\varphi)^2$, because Web Mercator stretches distances by $1/\cos\varphi$.

#### Sample-based estimates for large regions
Districts, provinces and Nepal are estimated from $n$ windows of 1.22 × 1.22 km placed uniformly at random inside the
region. With $p_{ik}$ the share of class $k$ in window $i$:

$$\hat p_k = \frac1n\sum_i p_{ik},\qquad \widehat{SE}(\hat p_k)=\frac{s_k}{\sqrt n},\qquad
\text{95\% CI}=\hat p_k \pm t_{0.975,\,n-1}\,\widehat{SE},\qquad \widehat{A}_k=\hat p_k\,A_{\text{region}}$$

This is the design-based estimator used in national forest inventories; it is unbiased for the *model's* map
regardless of where errors fall, and its interval reflects sampling only (not classification error).

#### Change over time
The national atlas observes the **same** sites every year, so change is estimated from paired differences
$d_{ik}=p_{ik}^{(t_2)}-p_{ik}^{(t_1)}$, whose standard error is far smaller than comparing two independent samples.
A $p$-value from the paired $t$ test is reported with each change.

#### Agreement with a reference map
Overall agreement, Cohen's $\kappa$, and per-class precision, recall, F1 and IoU against the Impact Observatory 10 m
map (2017–2025) or ESA WorldCover (2020–2021), with unlabelled and cloud pixels excluded.""")

with tab_data:
    src = pd.DataFrame(
        [
            (
                "Esri World Imagery + Wayback",
                "Sub-metre imagery; 197 archived versions since Feb 2014",
                "2014–now",
                "0.3–1 m",
                "Esri terms of use",
            ),
            (
                "Sentinel-2 cloudless (EOX)",
                "Annual cloud-free mosaics (2017 is blank over Nepal)",
                "2016, 2018–2025",
                "10 m",
                "CC BY 4.0 (2016) / CC BY-NC-SA 4.0",
            ),
            (
                "Sentinel-2 L2A (Planetary Computer)",
                "Every pass, 5-day revisit",
                "2015–now",
                "10 m",
                "Copernicus open licence",
            ),
            (
                "Landsat 1–9 (Planetary Computer)",
                "MSS false colour, TM/ETM+/OLI true colour",
                "1972–now",
                "30–60 m",
                "Public domain",
            ),
            ("NASA GIBS", "Daily MODIS/VIIRS true colour", "2000–now", "250–375 m", "NASA open data"),
            ("Impact Observatory / Esri LULC", "Reference land cover, 9 classes", "2017–2025", "10 m", "CC BY 4.0"),
            ("ESA WorldCover", "Reference land cover, 11 classes", "2020–2021", "10 m", "CC BY 4.0"),
            ("NASA FIRMS", "Active fires (VIIRS, MODIS)", "2000–now", "375 m / 1 km", "NASA open data"),
            ("USGS ComCat", "Earthquakes", "1900–now", "events", "Public domain"),
        ("GloFAS v4 via Open-Meteo", "Modelled river discharge and 7-day forecast", "1984–now", "0.05°", "CC BY 4.0"),
            (
                "Open-Meteo (ERA5, CAMS)",
                "Weather, climate reanalysis, air quality",
                "1940–now",
                "~10–25 km",
                "CC BY 4.0",
            ),
            ("World Bank CCKP", "National climate (CRU TS 4.08, ERA5)", "1901–2023", "national", "CC BY 4.0"),
            ("World Bank WDI", "Population, urbanisation, forest & farm land", "1960–now", "national", "CC BY 4.0"),
            (
                "HYDE 3.3 via Our World in Data",
                "Population and land use reconstructions",
                "10,000 BCE–2023",
                "national",
                "CC BY 4.0",
            ),
            (
                "OCHA COD-AB (Survey Dept. of Nepal)",
                "Province, district and local-unit boundaries",
                "2024",
                "~100 m simplified",
                "CC BY-IGO",
            ),
        ],
        columns=["Source", "What", "Period", "Resolution", "Licence"],
    )
    st.dataframe(src, hide_index=True)

with tab_limits:
    st.markdown("""
- **Seven classes, defined by DeepGlobe.** There is no snow class: glaciers and permanent snow are *Barren land*.
  Shrubland falls between *Rangeland* and *Forest*.
- **Weak labels.** The Nepal-adapted model learned from another model's map (Impact Observatory, roughly 75–85 %
  accurate), so it inherits some of that map's habits. Its evaluation therefore also uses ESA WorldCover.
- **Resolution.** The network works at 5.46 m. Sentinel-2 (10 m) and especially Landsat (30 m) are upsampled to that
  scale; decade-scale Landsat results are experimental and the 1970s (MSS, no blue band) are shown as imagery only.
- **Imagery dates.** Esri basemap pixels can be several years old in remote areas — the app always shows the real
  capture date it gets from the imagery metadata.
- **Clouds and shadows.** Mosaics use the least-cloudy scenes, but clouds, haze and deep terrain shadow still cause
  errors, especially in the monsoon.
- **Statistics vs truth.** Confidence intervals describe sampling uncertainty of the model's map, not its
  classification error. Use the reference comparison to judge accuracy for your area.
- **Before 1972** numbers are reconstructions (HYDE), not measurements; they are labelled as such.""")
