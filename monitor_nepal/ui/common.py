"""Shared Streamlit helpers: theme awareness, headers, legends, formatting and settings."""
from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from ..config import CLASSES, PALETTE_LABELS, PALETTES, PAPER, chart_colors
from ..model import MODELS, available_models, default_model_key

NPT = timezone(timedelta(hours=5, minutes=45))

CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
  .mn-eyebrow {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; opacity: .7; margin-bottom: .1rem;}
  .mn-sub {opacity: .78; margin-top: -.4rem; margin-bottom: .6rem; max-width: 62rem;}
  .mn-legend {display:flex; flex-wrap:wrap; gap:.35rem .9rem; font-size:.86rem; align-items:center; margin:.2rem 0 .4rem;}
  .mn-legend span.sw {display:inline-block; width:.85rem; height:.85rem; border-radius:3px; margin-right:.35rem;
                      vertical-align:-2px; box-shadow: 0 0 0 1px rgba(127,127,127,.35) inset;}
  .mn-stripe {height:4px; border-radius:2px; margin:.15rem 0 .9rem;
              background: linear-gradient(90deg, #dc143c 0 62%, #003893 62% 100%); width: 7.5rem;}
  .mn-note {font-size:.85rem; opacity:.75;}
</style>
"""


def setup_page() -> None:
    st.html(CSS)


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # older runtimes / bare mode
        return False


def class_colors() -> list[str]:
    return chart_colors(is_dark())


def surface() -> str:
    return "#121211" if is_dark() else "#fbfbf9"


def header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    if eyebrow:
        st.html(f'<div class="mn-eyebrow">{html.escape(eyebrow)}</div>')
    st.title(title, anchor=False)
    st.html('<div class="mn-stripe"></div>')
    if subtitle:
        st.html(f'<div class="mn-sub">{subtitle}</div>')


def legend(palette: str | None = None, include_unknown: bool = True) -> None:
    palette = palette or settings()["palette"]
    cols = PALETTES[palette]
    items = []
    for c, rgb in zip(CLASSES, cols):
        if c.key == "unknown" and not include_unknown:
            continue
        items.append(f'<span title="{html.escape(c.definition)}"><span class="sw" '
                     f'style="background:rgb{tuple(rgb)}"></span>{html.escape(c.name)}</span>')
    st.html(f'<div class="mn-legend">{"".join(items)}</div>')


def note(text: str) -> None:
    st.html(f'<div class="mn-note">{text}</div>')


# ---- formatting ---------------------------------------------------------------------------
def fmt_num(v: float | None, digits: int = 0) -> str:
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div * 10 or (abs(v) >= div and digits == 0 and div >= 1e6):
            return f"{v / div:.1f}{suf}"
    return f"{v:,.{digits}f}"


def fmt_km2(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f} km²"
    if v >= 10:
        return f"{v:,.1f} km²"
    return f"{v:,.2f} km²"


def now_npt() -> datetime:
    return datetime.now(NPT)


def ago(ts: pd.Timestamp | datetime) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    secs = (pd.Timestamp.now(tz="UTC") - ts).total_seconds()
    if secs < 3600:
        return f"{max(1, int(secs // 60))} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    days = int(secs // 86400)
    return "1 day ago" if days == 1 else f"{days} days ago"


STATUS_BADGE = {"good": ("green", ":material/check_circle:"), "warning": ("orange", ":material/warning:"),
                "serious": ("orange", ":material/error:"), "critical": ("red", ":material/report:"),
                "neutral": ("gray", ":material/help:")}


def status_badge(label: str, status: str) -> None:
    color, icon = STATUS_BADGE.get(status, STATUS_BADGE["neutral"])
    st.badge(label, icon=icon, color=color)


# ---- settings shared by all pages -------------------------------------------------------------
def settings() -> dict:
    s = st.session_state.setdefault("mn_settings", {})
    s.setdefault("model", default_model_key())
    s.setdefault("tta", 1)
    s.setdefault("palette", "cartographic")
    if s["model"] not in MODELS or not MODELS[s["model"]].available:
        s["model"] = default_model_key()
    return s


def sidebar_settings() -> dict:
    s = settings()
    with st.sidebar:
        st.markdown("**Model**")
        models = available_models()
        s["model"] = st.radio("Model", [m.key for m in models], index=[m.key for m in models].index(s["model"]),
                              format_func=lambda k: MODELS[k].short, label_visibility="collapsed",
                              help="The Nepal-adapted network is the paper's U-Net fine-tuned for Nepal. "
                                   "The original is exactly the published model.")
        s["tta"] = 4 if st.toggle("Test-time augmentation", value=s["tta"] > 1,
                                  help="Average 4 flipped/rotated predictions: smoother maps, 4x slower.") else 1
        s["palette"] = st.selectbox("Map colours", list(PALETTES), index=list(PALETTES).index(s["palette"]),
                                    format_func=PALETTE_LABELS.get)
        st.divider()
        st.caption(f"Model from **{PAPER.short_authors} ({PAPER.year})**, *{PAPER.journal}* "
                   f"{PAPER.volume}({PAPER.issue}). [Paper]({PAPER.url}) · [Code]({PAPER.code_url})")
    return s


LOW_RES_KINDS = {"eox", "s2", "landsat", "mss"}


def preliminary_note(model_key: str) -> None:
    """Flag results from a fine-tuned model whose training run has not finished yet."""
    import json

    from ..config import MODELS_DIR

    path = MODELS_DIR / "finetune_log.json"
    if model_key != "nepal" or not path.exists():
        return
    log = json.loads(path.read_text())
    done, planned = len(log.get("epochs", [])), log.get("epochs_planned") or 0
    if planned and done < planned:
        st.caption(f":material/construction: **Preliminary model** — the Nepal adaptation has completed {done} of "
                   f"{planned} planned training epochs. Year-to-year figures can still contain model artefacts (for "
                   "example too much *urban land* on recent mosaics); read them next to the independent reference "
                   "map shown in grey.")


def model_badge(model_key: str) -> None:
    """Which network produced the figures below, how well it scores, and when it was last updated."""
    import json

    from ..config import MODELS_DIR, PAPER_MEAN_IOU

    path = MODELS_DIR / "finetune_log.json"
    if model_key == "nepal" and path.exists():
        log = json.loads(path.read_text())
        best = log["epochs"][log["best_epoch"] - 1]["val"]["all"]["miou"]
        label = (f"Nepal-adapted U-Net · validation mIoU {best:.3f} on held-out districts "
                 f"(paper weights: {log['baseline']['all']['miou']:.3f})")
        if log.get("updated"):
            label += f" · updated {datetime.fromisoformat(log['updated']):%-d %b %Y}"
        st.badge(label, icon=":material/verified:", color="green")
    elif model_key == "paper":
        st.badge(f"Original U-Net from the paper · DeepGlobe test mIoU {PAPER_MEAN_IOU:.3f}", icon=":material/article:",
                 color="gray")


def model_source_warning(model_key: str, *kinds: str) -> None:
    """The original DeepGlobe network was trained on sub-metre imagery; on 10-30 m imagery it fails."""
    if model_key == "paper" and any(k in LOW_RES_KINDS for k in kinds):
        st.warning("The **original** model was trained on sub-metre DeepGlobe imagery and is not reliable on "
                   "Sentinel-2 or Landsat (on held-out Nepal districts its mean IoU on Sentinel-2 mosaics is "
                   "about 0.04 — it can call most of Nepal *water*). Switch to the **Nepal-adapted** model in "
                   "the sidebar, or use Esri imagery.", icon=":material/warning:")


def chart_and_table(fig, df: pd.DataFrame, *, key: str, height: int | None = None,
                    column_config: dict | None = None) -> None:
    """Every chart ships with a table twin (accessibility + exact values)."""
    tab_chart, tab_table = st.tabs(["Chart", "Table"], key=f"tabs_{key}")
    with tab_chart:
        st.plotly_chart(fig, key=f"chart_{key}", config={"displaylogo": False, "responsive": True})
    with tab_table:
        st.dataframe(df, hide_index=True, column_config=column_config, height=height or "auto")
        st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=f"{key}.csv",
                           mime="text/csv", key=f"dl_{key}", icon=":material/download:")
