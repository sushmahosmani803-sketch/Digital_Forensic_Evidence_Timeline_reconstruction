"""Plotly figure builders for the dashboard (kept separate so they can be tested)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import FLAG_THRESHOLD, RISK_ORDER

NAVY = "#0E1A2B"
PANEL = "#15243A"
PAPER = "#E6EBF2"
MUTED = "#8EA3BF"
GRID = "rgba(142,163,191,0.14)"
TAPE = "#F2C230"
CRITICAL = "#FF5D5D"
SIGNAL = "#5AB0FF"
LOW = "#3E5778"

RISK_COLORS = {"Critical": CRITICAL, "High": TAPE, "Medium": SIGNAL, "Low": LOW}
RISK_SCALE = [[0.0, LOW], [0.35, SIGNAL], [0.6, TAPE], [0.8, "#FF8A5B"], [1.0, CRITICAL]]
SOURCE_ORDER = ["logon", "device", "file", "email", "http", "unknown"]
FONT = "Archivo, 'Segoe UI', system-ui, sans-serif"


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=PAPER, size=13),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=TAPE, font=dict(family=FONT, color=PAPER)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def _flag_series(df: pd.DataFrame) -> pd.Series:
    if "flagged" in df.columns:
        return df["flagged"].astype(bool)
    return df["risk_score"] >= FLAG_THRESHOLD


def density_strip(df: pd.DataFrame, freq: str = "W-MON") -> go.Figure:
    """Hero strip: events per period with flagged events drawn on top."""
    d = df.assign(flag=_flag_series(df)).set_index("timestamp")
    total = d["user"].resample(freq).size()
    flagged = d["flag"].resample(freq).sum()
    fig = go.Figure()
    fig.add_bar(x=total.index, y=total.values, name="All events", marker_color="rgba(142,163,191,0.35)",
                hovertemplate="%{x|%d %b %Y}<br>%{y} events<extra></extra>")
    fig.add_bar(x=flagged.index, y=flagged.values, name="Flagged", marker_color=CRITICAL,
                hovertemplate="%{x|%d %b %Y}<br>%{y} flagged<extra></extra>")
    fig.update_layout(barmode="overlay", bargap=0.15)
    style(fig, height=150, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=4, b=0))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(showgrid=False)
    return fig


def activity_over_time(df: pd.DataFrame, freq: str = "W-MON") -> go.Figure:
    d = df.assign(flag=_flag_series(df)).set_index("timestamp")
    flagged = d["flag"].resample(freq).sum()
    normal = d["user"].resample(freq).size() - flagged
    fig = go.Figure()
    fig.add_scatter(x=normal.index, y=normal.values, name="Normal", mode="lines", stackgroup="a",
                    line=dict(color=SIGNAL, width=1), fillcolor="rgba(90,176,255,0.25)")
    fig.add_scatter(x=flagged.index, y=flagged.values, name="Flagged", mode="lines", stackgroup="a",
                    line=dict(color=CRITICAL, width=1.5), fillcolor="rgba(255,93,93,0.45)")
    return style(fig, 300)


def hour_weekday_heatmap(df: pd.DataFrame) -> go.Figure:
    d = df.assign(flag=_flag_series(df))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = (
        d[d["flag"]]
        .groupby([d["timestamp"].dt.dayofweek, d["timestamp"].dt.hour])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values, x=[f"{h:02d}" for h in range(24)], y=days,
            colorscale=[[0, PANEL], [0.25, LOW], [0.6, TAPE], [1, CRITICAL]],
            hovertemplate="%{y} %{x}:00<br>%{z} flagged events<extra></extra>",
            colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=MUTED)), xgap=2, ygap=2,
        )
    )
    style(fig, 300, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, title=dict(text="Hour of day", font=dict(color=MUTED)))
    return fig


def source_breakdown(df: pd.DataFrame) -> go.Figure:
    d = df.assign(flag=_flag_series(df))
    g = d.groupby("source_type")["flag"].agg(["size", "sum"])
    g["normal"] = g["size"] - g["sum"]
    g = g.reindex([s for s in SOURCE_ORDER if s in g.index] + [s for s in g.index if s not in SOURCE_ORDER])
    fig = go.Figure()
    fig.add_bar(y=g.index, x=g["normal"], name="Normal", orientation="h", marker_color=LOW)
    fig.add_bar(y=g.index, x=g["sum"], name="Flagged", orientation="h", marker_color=CRITICAL)
    fig.update_layout(barmode="stack")
    style(fig, 300)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def risk_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for level in reversed(RISK_ORDER):
        part = df[df["risk_level"] == level]
        if len(part):
            fig.add_histogram(x=part["risk_score"], name=level, marker_color=RISK_COLORS[level],
                              xbins=dict(start=0, end=1.0001, size=0.025))
    fig.update_layout(barmode="stack", bargap=0.05)
    fig.add_vline(x=FLAG_THRESHOLD, line=dict(color=PAPER, dash="dot", width=1))
    style(fig, 300)
    fig.update_xaxes(range=[0, 1], title=dict(text="Risk score", font=dict(color=MUTED)))
    return fig


def timeline_scatter(events: pd.DataFrame) -> go.Figure:
    """Reconstructed timeline: one lane per evidence source, coloured by risk."""
    lanes = [s for s in SOURCE_ORDER if s in set(events["source_type"])] + sorted(
        set(events["source_type"]) - set(SOURCE_ORDER)
    )
    act = events["display_activity"] if "display_activity" in events else events["activity"]
    custom = np.column_stack([act.astype(str), events["pc"].astype(str), events.get("event_id", pd.Series([""] * len(events))).astype(str),
                              events["risk_level"].astype(str)])
    fig = go.Figure()
    fig.add_scatter(
        x=events["timestamp"], y=events["source_type"], mode="markers",
        marker=dict(size=9 + 12 * events["risk_score"], color=events["risk_score"], colorscale=RISK_SCALE, cmin=0, cmax=1,
                    line=dict(width=0.5, color=NAVY), opacity=0.9,
                    colorbar=dict(title=dict(text="Risk", font=dict(color=MUTED)), thickness=10, outlinewidth=0, tickfont=dict(color=MUTED))),
        customdata=custom,
        hovertemplate="<b>%{customdata[0]}</b><br>%{x|%d %b %Y %H:%M:%S}<br>%{customdata[1]} · %{customdata[3]} risk<br>%{customdata[2]}<extra></extra>",
        showlegend=False,
    )
    style(fig, 360, legend=False)
    fig.update_yaxes(categoryorder="array", categoryarray=lanes, showgrid=True)
    fig.update_xaxes(rangeslider=dict(visible=True, bgcolor=PANEL, thickness=0.08))
    return fig


def user_bubble(profiles: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for level in reversed(RISK_ORDER):
        p = profiles[profiles["risk_level"] == level]
        if p.empty:
            continue
        fig.add_scatter(
            x=p["events"], y=p["mean_risk"], mode="markers", name=level, text=p["user"],
            marker=dict(size=10 + 30 * np.sqrt(p["flag_rate"].clip(0, 1)), color=RISK_COLORS[level], opacity=0.85,
                        line=dict(width=1, color=NAVY)),
            hovertemplate="<b>%{text}</b><br>%{x} events<br>mean risk %{y:.2f}<extra></extra>",
        )
    style(fig, 360)
    fig.update_xaxes(type="log", title=dict(text="Events (log scale)", font=dict(color=MUTED)))
    fig.update_yaxes(range=[0, 1], title=dict(text="Mean risk", font=dict(color=MUTED)))
    return fig


def metric_comparison(models: dict, keys: list[str]) -> go.Figure:
    metrics = ["accuracy", "precision", "recall", "f1"]
    palette = [LOW, SIGNAL, TAPE, CRITICAL, PAPER]
    fig = go.Figure()
    for i, k in enumerate(keys):
        m = models[k]
        fig.add_bar(x=[x.title() if x != "f1" else "F1" for x in metrics], y=[m[x] for x in metrics], name=m["name"],
                    marker_color=palette[i % len(palette)], hovertemplate="%{x}: %{y:.3f}<extra>" + m["name"] + "</extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    style(fig, 340)
    fig.update_yaxes(range=[0, 1])
    return fig


def confusion_heatmap(cm) -> go.Figure:
    cm = np.asarray(cm)
    labels = ["Normal", "Malicious"]
    text = [[f"{v:,}" for v in row] for row in cm]
    fig = go.Figure(
        go.Heatmap(z=cm, x=labels, y=labels, text=text, texttemplate="%{text}", textfont=dict(size=18, color=PAPER),
                   colorscale=[[0, PANEL], [1, "#2D5C8F"]], showscale=False, xgap=3, ygap=3,
                   hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z}<extra></extra>")
    )
    style(fig, 280, legend=False)
    fig.update_yaxes(autorange="reversed", title=dict(text="Actual", font=dict(color=MUTED)), showgrid=False)
    fig.update_xaxes(title=dict(text="Predicted", font=dict(color=MUTED)), showgrid=False)
    return fig


def roc_curves(models: dict, keys: list[str]) -> go.Figure:
    palette = [LOW, SIGNAL, TAPE, CRITICAL, PAPER]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRID, dash="dot"), showlegend=False, hoverinfo="skip")
    for i, k in enumerate(keys):
        m = models[k]
        if "roc_curve" in m:
            fig.add_scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"], mode="lines",
                            name=f"{m['name']} (AUC {m.get('roc_auc', 0):.3f})", line=dict(color=palette[i % len(palette)], width=2))
    style(fig, 340)
    fig.update_xaxes(range=[0, 1], title=dict(text="False positive rate", font=dict(color=MUTED)))
    fig.update_yaxes(range=[0, 1.02], title=dict(text="True positive rate", font=dict(color=MUTED)))
    return fig


def importance_bars(importance: dict, top: int = 13) -> go.Figure:
    items = list(importance.items())[:top][::-1]
    names = [k.replace("_", " ") for k, _ in items]
    vals = [v for _, v in items]
    behavioural = {"gap prev sec", "gap next sec", "user day events", "foreign pc", "off hours", "is weekend", "minute of day"}
    colors = [TAPE if n in behavioural else LOW for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    style(fig, 380, legend=False)
    fig.update_yaxes(showgrid=False)
    return fig
