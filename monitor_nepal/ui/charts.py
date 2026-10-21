"""Plotly charts with one consistent look.

Rules followed throughout: colour follows the land-cover class (never its rank); one y-axis per chart
(two measures -> two charts); thin marks with a 2 px surface gap between stacked segments; area fills
are 10 % washes; selective direct labels; hover readouts list every series; every chart is rendered
next to a table view by ``common.chart_and_table``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import CLASS_NAMES, CLASSES
from .common import class_colors, is_dark, surface

INK_MUTED = "#898781"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SINGLE = "#2a78d6"  # slot-1 hue for single-series charts


def _single() -> str:
    return "#3987e5" if is_dark() else SINGLE


def single_color() -> str:
    """Slot-1 hue, stepped for the current theme."""
    return _single()


def fire_color() -> str:
    return "#e66767" if is_dark() else "#e34948"


def quake_color() -> str:
    return "#9085e9" if is_dark() else "#6250d6"


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _ink_on(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.55 else "#ffffff"


def style(fig: go.Figure, height: int = 320, *, legend: bool = True, hovermode: str | bool = "x unified",
          title: str | None = None) -> go.Figure:
    top = (74 if legend else 44) if title else (34 if legend else 12)
    fig.update_layout(height=height, margin=dict(l=6, r=10, t=top, b=6),
                      title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode=hovermode,
                      showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left",
                                                     title=None, font=dict(size=12), traceorder="normal"),
                      bargap=0.35)
    fig.update_xaxes(showgrid=False, ticks="outside", ticklen=4)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig


# ---- land-cover composition --------------------------------------------------------------------------
def composition_bar(shares: pd.Series | np.ndarray, *, title: str | None = None, include_unknown: bool = False,
                    height: int = 150) -> go.Figure:
    """One 100 % stacked horizontal bar of class shares (%)."""
    shares = np.asarray(shares, float)
    colors = class_colors()
    fig = go.Figure()
    for c, v, col in zip(CLASSES, shares, colors):
        if c.key == "unknown" and not include_unknown:
            continue
        fig.add_bar(y=[""], x=[v], orientation="h", name=c.name, marker=dict(color=col, line=dict(color=surface(), width=2)),
                    text=[f"{v:.0f}%" if v >= 7 else ""], textposition="inside", insidetextanchor="middle",
                    textfont=dict(color=_ink_on(col), size=12), hovertemplate=f"{c.name}: %{{x:.1f}}%<extra></extra>")
    fig.update_layout(barmode="stack")
    style(fig, height, hovermode="closest", title=title)
    if title:  # six class names wrap to two legend rows: leave room for them under the title
        fig.update_layout(height=height + 30, margin_t=104)
    fig.update_xaxes(range=[0, max(100, shares.sum())], ticksuffix="%", showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_traces(width=0.42)
    return fig


def composition_ci(est: pd.DataFrame, *, title: str | None = None, value: str = "share") -> go.Figure:
    """Class shares with 95 % confidence intervals (survey estimates)."""
    est = est[est["key"] != "unknown"]
    colors = dict(zip(CLASS_NAMES, class_colors()))
    fig = go.Figure()
    fig.add_bar(y=est["class"], x=est[value], orientation="h", marker=dict(color=[colors[c] for c in est["class"]]),
                error_x=dict(type="data", symmetric=False, array=est["high"] - est[value],
                             arrayminus=est[value] - est["low"], thickness=1.5, width=5, color=INK_MUTED),
                customdata=np.stack([est["low"], est["high"], est["km2"]], axis=1), width=0.5,
                hovertemplate="%{y}: %{x:.1f}% (95% CI %{customdata[0]:.1f}–%{customdata[1]:.1f}%)<br>"
                              "≈ %{customdata[2]:,.0f} km²<extra></extra>", showlegend=False)
    style(fig, 300, legend=False, hovermode="closest", title=title)
    fig.update_xaxes(ticksuffix="%", rangemode="tozero", showgrid=True, gridwidth=1)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def class_trends(df: pd.DataFrame, *, x: str = "year", ref: pd.DataFrame | None = None, title: str | None = None,
                 classes: list[str] | None = None) -> go.Figure:
    """Small multiples - one panel per class - of share over time, with 95 % bands when available.

    ``df`` columns: x, key, share, [low, high].  ``ref`` (same shape) is drawn in muted grey for comparison.
    """
    keys = classes or [c.key for c in CLASSES if c.key != "unknown"]
    names = {c.key: c.name for c in CLASSES}
    colors = {c.key: col for c, col in zip(CLASSES, class_colors())}
    cols = 3
    rows = int(np.ceil(len(keys) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[names[k] for k in keys], shared_xaxes=True,
                        vertical_spacing=0.16, horizontal_spacing=0.07)
    for i, k in enumerate(keys):
        r, c = i // cols + 1, i % cols + 1
        d = df[df["key"] == k].sort_values(x)
        if {"low", "high"} <= set(d.columns) and d["low"].notna().any():
            fig.add_trace(go.Scatter(x=list(d[x]) + list(d[x])[::-1], y=list(d["high"]) + list(d["low"])[::-1],
                                     fill="toself", mode="lines", fillcolor=rgba(colors[k], 0.12), line=dict(width=0),
                                     hoverinfo="skip", showlegend=False), row=r, col=c)
        if ref is not None:
            rr = ref[ref["key"] == k].sort_values(x)
            fig.add_trace(go.Scatter(x=rr[x], y=rr["share"], mode="lines+markers", name="Reference map (IO 10 m)",
                                     line=dict(color=INK_MUTED, width=2), marker=dict(size=6),
                                     legendgroup="ref", showlegend=i == 0,
                                     hovertemplate="Reference: %{y:.1f}%<extra></extra>"), row=r, col=c)
        fig.add_trace(go.Scatter(x=d[x], y=d["share"], mode="lines+markers", name="This model",
                                 line=dict(color=colors[k], width=2), marker=dict(size=8, line=dict(color=surface(), width=2)),
                                 legendgroup="model", showlegend=i == 0,
                                 hovertemplate=f"{names[k]}: %{{y:.1f}}%<extra></extra>"), row=r, col=c)
        fig.update_yaxes(ticksuffix="%", rangemode="tozero", row=r, col=c)
    style(fig, 240 * rows + 70, title=title)
    fig.update_layout(hovermode="x", margin=dict(t=110 if title else 70),
                      legend=dict(y=1.0, yref="container", yanchor="top", x=0.5, xanchor="center"))
    if title:
        fig.update_layout(title=dict(y=0.985, yref="container", yanchor="top"))
    return fig


def stacked_columns(df: pd.DataFrame, *, x: str = "year", title: str | None = None, height: int = 330) -> go.Figure:
    """Part-to-whole over time: columns = periods, segments = classes (2 px surface gaps)."""
    colors = {c.key: col for c, col in zip(CLASSES, class_colors())}
    fig = go.Figure()
    for c in CLASSES:
        if c.key == "unknown":
            continue
        d = df[df["key"] == c.key].sort_values(x)
        fig.add_bar(x=d[x].astype(str), y=d["share"], name=c.name,
                    marker=dict(color=colors[c.key], line=dict(color=surface(), width=2)),
                    hovertemplate=f"{c.name}: %{{y:.1f}}%<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.45)
    style(fig, height, title=title)
    fig.update_yaxes(ticksuffix="%", range=[0, 100.5])
    return fig


def net_change(df: pd.DataFrame, *, value: str = "change", unit: str = "km²", title: str | None = None) -> go.Figure:
    """Gains and losses per class against a zero baseline."""
    colors = dict(zip(CLASS_NAMES, class_colors()))
    d = df[df["class"] != "Unknown"]
    fig = go.Figure(go.Bar(x=d[value], y=d["class"], orientation="h", width=0.5,
                           marker=dict(color=[colors[c] for c in d["class"]]),
                           text=[f"{v:+,.2f}" for v in d[value]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:+,.2f} " + unit + "<extra></extra>"))
    style(fig, 280, legend=False, hovermode="closest", title=title)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor=INK_MUTED, showgrid=True, ticksuffix=f" {unit}")
    return fig


def sankey(matrix: pd.DataFrame, *, title: str | None = None, min_value: float = 0.0) -> go.Figure:
    names = list(matrix.index)
    colors = class_colors()
    src, dst, val, lc = [], [], [], []
    for i in range(len(names)):
        for j in range(len(names)):
            v = float(matrix.iloc[i, j])
            if v > min_value:
                src.append(i)
                dst.append(len(names) + j)
                val.append(v)
                lc.append(rgba(colors[i], 0.35 if i != j else 0.12))
    fig = go.Figure(go.Sankey(
        arrangement="snap", valueformat=",.3f", valuesuffix=" km²",
        node=dict(label=[f"{n} (before)" for n in names] + [f"{n} (after)" for n in names],
                  color=colors + colors, pad=14, thickness=14, line=dict(width=0)),
        link=dict(source=src, target=dst, value=val, color=lc)))
    style(fig, 420, legend=False, hovermode="closest", title=title)
    return fig


# ---- generic series -------------------------------------------------------------------------------------
def line(df: pd.DataFrame, x: str, y: str, *, name: str, unit: str = "", color: str | None = None,
         title: str | None = None, height: int = 300, band: tuple[str, str] | None = None, markers: bool = False,
         area: bool = False) -> go.Figure:
    """``markers`` are drawn only on sparse series (<= 40 points); dense series read better as a plain line."""
    color = color or _single()
    markers = markers and len(df) <= 40
    fig = go.Figure()
    if band:
        lo, hi = band
        fig.add_trace(go.Scatter(x=list(df[x]) + list(df[x])[::-1], y=list(df[hi]) + list(df[lo])[::-1],
                                 fill="toself", mode="lines", fillcolor=rgba(color, 0.12), line=dict(width=0), hoverinfo="skip",
                                 showlegend=False, name="range"))
    fig.add_trace(go.Scatter(x=df[x], y=df[y], name=name, mode="lines+markers" if markers else "lines", connectgaps=True,
                             line=dict(color=color, width=2), fill="tozeroy" if area else None,
                             fillcolor=rgba(color, 0.10) if area else None,
                             marker=dict(size=8, line=dict(color=surface(), width=2)),
                             hovertemplate=f"{name}: %{{y:,.2f}}{(' ' + unit) if unit else ''}<extra></extra>"))
    style(fig, height, legend=False, title=title)
    if unit:
        fig.update_yaxes(title=dict(text=unit, font=dict(size=12)))
    return fig


def bars(x, y, *, name: str, unit: str = "", title: str | None = None, height: int = 280, horizontal: bool = False,
         color: str | None = None) -> go.Figure:
    color = color or _single()
    tmpl = f"%{{{'y' if horizontal else 'x'}}}: %{{{'x' if horizontal else 'y'}:,.0f}} {unit}<extra></extra>"
    fig = go.Figure(go.Bar(x=y if horizontal else x, y=x if horizontal else y, orientation="h" if horizontal else "v",
                           marker=dict(color=color), name=name, hovertemplate=tmpl))
    style(fig, height, legend=False, hovermode="closest", title=title)
    peak = float(np.nanmax(np.asarray(y, float))) if len(y) else 0.0
    value_axis = fig.update_xaxes if horizontal else fig.update_yaxes
    value_axis(rangemode="tozero", **({"range": [0, 1]} if peak <= 0 else {}))
    if horizontal:
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=True)
    return fig


def heatmap(z: pd.DataFrame, *, title: str | None = None, unit: str = "", height: int = 360) -> go.Figure:
    scale = [[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)]
    fig = go.Figure(go.Heatmap(z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index],
                               colorscale=scale, xgap=2, ygap=2,
                               colorbar=dict(thickness=10, outlinewidth=0, title=dict(text=unit, side="right")),
                               hovertemplate="%{y} · %{x}: %{z:,.0f} " + unit + "<extra></extra>"))
    style(fig, height, legend=False, hovermode="closest", title=title)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(type="category")
    return fig


def river_hydrographs(recent: pd.DataFrame, clim: pd.DataFrame | None, rivers, *, cols: int = 3) -> go.Figure:
    """Small multiples: observed flow (solid), forecast (dashed, it *is* a projection) and the normal range."""
    rows = int(np.ceil(len(rivers) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"{r.name} at {r.gauge}" for r in rivers],
                        vertical_spacing=0.16, horizontal_spacing=0.06)
    color = _single()
    for i, r in enumerate(rivers):
        rr, cc = i // cols + 1, i % cols + 1
        d = recent[recent.river == r.key].sort_values("date")
        if d.empty:
            continue
        if clim is not None:
            c = clim[clim.river == r.key].set_index("doy")
            doy = d.date.dt.dayofyear.clip(upper=365)
            lo, mid, hi = c.p10.reindex(doy).values, c.p50.reindex(doy).values, c.p90.reindex(doy).values
            fig.add_trace(go.Scatter(x=list(d.date) + list(d.date)[::-1], y=list(hi) + list(lo)[::-1], fill="toself", mode="lines",
                                     fillcolor=rgba(INK_MUTED, 0.14), line=dict(width=0), hoverinfo="skip",
                                     name="Normal range 2004–2023 (10th–90th pct.)", legendgroup="band",
                                     showlegend=i == 0), row=rr, col=cc)
            fig.add_trace(go.Scatter(x=d.date, y=mid, mode="lines", line=dict(color=INK_MUTED, width=1),
                                     name="Median 2004–2023", legendgroup="median", showlegend=i == 0,
                                     hovertemplate="Median: %{y:,.0f} m³/s<extra></extra>"), row=rr, col=cc)
        obs, fc = d[~d.forecast], d[d.forecast]
        fig.add_trace(go.Scatter(x=obs.date, y=obs.discharge, mode="lines", line=dict(color=color, width=2),
                                 name="Modelled flow", legendgroup="obs", showlegend=i == 0,
                                 hovertemplate="%{x|%d %b}: %{y:,.0f} m³/s<extra></extra>"), row=rr, col=cc)
        if not fc.empty:
            link = pd.concat([obs.tail(1), fc])
            fig.add_trace(go.Scatter(x=link.date, y=link.discharge, mode="lines",
                                     line=dict(color=color, width=2, dash="dot"), name="Forecast", legendgroup="fc",
                                     showlegend=i == 0, hovertemplate="%{x|%d %b}: %{y:,.0f} m³/s (forecast)<extra></extra>"),
                          row=rr, col=cc)
        fig.update_yaxes(rangemode="tozero", row=rr, col=cc)
    style(fig, 250 * rows + 50)
    fig.update_layout(hovermode="x")
    return fig


def add_events(fig: go.Figure, events, *, y_frac: float = 1.0, lines: bool = False) -> go.Figure:
    """Historical events as a rug of diamonds along the top edge, labelled on hover."""
    xs, texts = [], []
    for e in events:
        if lines:
            fig.add_vline(x=e.year, line_width=1, line_color=rgba("#898781", 0.45))
        xs.append(e.year)
        texts.append(f"<b>{e.year} · {e.title}</b><br>{e.detail}")
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=[y_frac] * len(xs), yaxis="y2", mode="markers", name="Events",
                                 marker=dict(size=9, color=INK_MUTED, symbol="diamond",
                                             line=dict(color=surface(), width=2)),
                                 hovertext=texts, hoverinfo="text", showlegend=False))
        fig.update_layout(yaxis2=dict(overlaying="y", range=[0, 1.05], visible=False, fixedrange=True))
    return fig


def add_era_bands(fig: go.Figure, bands: list[tuple[float, float, str]]) -> go.Figure:
    """Shaded spans naming where each part of a long series comes from (provenance), labelled at the bottom."""
    shade = rgba("#898781", 0.09)
    for i, (x0, x1, label) in enumerate(bands):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=shade if i % 2 == 0 else "rgba(0,0,0,0)", line_width=0, layer="below",
                      annotation_text=label, annotation_position="bottom left",
                      annotation=dict(font=dict(size=11, color=INK_MUTED), yshift=-2))
    return fig
