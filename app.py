"""AI-Based Digital Evidence Timeline Reconstruction - investigator dashboard.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import inspect
import json

import pandas as pd
import streamlit as st

from evidence import charts, config
from evidence.normalize import normalize_raw_csv
from evidence.report import build_investigation_report, load_report
from evidence.timeline import detect_gaps, reconstruct_timeline, summarize_sessions, user_risk_profiles

st.set_page_config(page_title="Evidence Timeline Reconstruction", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Compatibility helpers (Streamlit renamed use_container_width -> width)
# --------------------------------------------------------------------------- #
def _stretch(fn) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    width = params.get("width")
    if width is not None and isinstance(width.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def show_chart(fig, key: str | None = None):
    kwargs = _stretch(st.plotly_chart)
    if key:
        kwargs["key"] = key
    st.plotly_chart(fig, config={"displayModeBar": False}, **kwargs)


def show_table(df: pd.DataFrame, **kwargs):
    st.dataframe(df, hide_index=True, **_stretch(st.dataframe), **kwargs)


def esc(value) -> str:
    return html.escape(str(value))


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Archivo+Narrow:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --navy: #0E1A2B; --panel: #15243A; --panel-2: #1B2D47; --paper: #E6EBF2;
  --muted: #8EA3BF; --tape: #F2C230; --critical: #FF5D5D; --signal: #5AB0FF; --low: #3E5778;
  --line: rgba(142,163,191,.18);
}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family: 'Archivo', 'Segoe UI', system-ui, sans-serif;
}
.stApp { background: radial-gradient(1200px 600px at 85% -10%, #17304F 0%, var(--navy) 55%) fixed; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
h1, h2, h3 { font-family: 'Archivo Narrow', 'Archivo', sans-serif; letter-spacing: -.01em; color: var(--paper); }
h3 { font-size: 1.25rem; font-weight: 600; margin-top: .4rem; }
section[data-testid="stSidebar"] { background: #0B1524; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] h2 { font-size: 1.15rem; }
:focus-visible { outline: 2px solid var(--tape) !important; outline-offset: 2px; }

/* ---- Case tag hero ---- */
.case-tag { position: relative; background: var(--panel); border: 1px solid var(--line); border-radius: 14px 14px 4px 4px;
  padding: 0 0 .4rem 0; overflow: hidden; margin-bottom: .6rem; }
.tape { background: repeating-linear-gradient(-45deg, var(--tape) 0 22px, #1b1b1b 22px 30px);
  height: 10px; }
.tape-label { display: inline-block; margin: -1px 0 0 1.6rem; background: var(--tape); color: #14110a;
  font-family: 'Archivo Narrow', sans-serif; font-weight: 700; font-size: .85rem; padding: .18rem .7rem .26rem;
  border-radius: 0 0 6px 6px; }
.case-body { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr); gap: 2rem; padding: 1rem 1.6rem .2rem; }
.case-title { font-family: 'Archivo Narrow', sans-serif; font-weight: 700; font-size: clamp(1.8rem, 3.2vw, 2.7rem);
  line-height: 1.05; margin: .2rem 0 .5rem; color: var(--paper); }
.case-sub { color: var(--muted); font-size: 1rem; max-width: 62ch; line-height: 1.5; margin: 0; }
.case-facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .2rem 1.4rem; align-content: start;
  border-left: 1px dashed var(--line); padding-left: 1.4rem; }
.fact { padding: .45rem 0; border-bottom: 1px solid var(--line); }
.fact b { display: block; font-size: 1.45rem; font-weight: 600; color: var(--paper); font-variant-numeric: tabular-nums; }
.fact span { color: var(--muted); font-size: .85rem; }
.fact.alert b { color: var(--critical); }
.strip-caption { color: var(--muted); font-size: .85rem; padding: 0 1.6rem; margin-top: .4rem; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family: 'Archivo Narrow', sans-serif; font-size: 1.05rem; font-weight: 600;
  color: var(--muted); padding: .55rem 1rem; }
.stTabs [aria-selected="true"] { color: var(--paper) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--tape) !important; height: 3px; }

/* ---- Panels & notes ---- */
.note { background: var(--panel); border-left: 3px solid var(--signal); border-radius: 4px; padding: .7rem 1rem;
  color: var(--muted); font-size: .92rem; line-height: 1.5; margin: .4rem 0 1rem; }
.note.warn { border-left-color: var(--tape); }
.note b { color: var(--paper); }

/* ---- Event chain ---- */
.chain { position: relative; margin: .4rem 0 0 .5rem; padding-left: 1.4rem; border-left: 2px solid var(--line); }
.ev { position: relative; padding: .5rem .8rem .55rem; margin-bottom: .45rem; background: var(--panel);
  border-radius: 6px; border: 1px solid var(--line); }
.ev::before { content: ""; position: absolute; left: calc(-1.4rem - 7px); top: .85rem; width: 12px; height: 12px;
  border-radius: 50%; background: var(--pip, var(--low)); box-shadow: 0 0 0 3px var(--navy); }
.ev-top { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; flex-wrap: wrap; }
.ev-what { font-weight: 600; color: var(--paper); }
.ev-when, .ev-id { font-family: 'IBM Plex Mono', monospace; font-size: .8rem; color: var(--muted); }
.ev-meta { color: var(--muted); font-size: .85rem; margin-top: .15rem; }
.risk-chip { font-size: .78rem; font-weight: 600; padding: .08rem .5rem; border-radius: 999px; color: #10151d;
  background: var(--pip, var(--low)); }
.risk-chip.Low { color: var(--paper); }

/* ---- Model cards ---- */
.model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: .7rem; margin: .3rem 0 1rem; }
.model-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: .8rem 1rem; }
.model-card.best { border-color: var(--tape); }
.model-card h4 { margin: 0 0 .1rem; font-family: 'Archivo Narrow', sans-serif; font-size: 1.05rem; color: var(--paper); }
.model-card p { margin: 0 0 .5rem; color: var(--muted); font-size: .8rem; line-height: 1.35; min-height: 2.2em; }
.model-card .f1 { font-size: 1.9rem; font-weight: 600; color: var(--paper); font-variant-numeric: tabular-nums; }
.model-card .f1 small { font-size: .85rem; color: var(--muted); font-weight: 400; margin-left: .3rem; }
.model-card .row { display: flex; justify-content: space-between; color: var(--muted); font-size: .82rem;
  border-top: 1px solid var(--line); padding-top: .35rem; margin-top: .35rem; font-variant-numeric: tabular-nums; }

div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: .6rem .9rem; }
.stDownloadButton button, .stButton button { border-radius: 6px; font-weight: 600; }

@media (max-width: 900px) {
  .case-body { grid-template-columns: 1fr; gap: 1rem; }
  .case-facts { border-left: 0; padding-left: 0; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

RISK_HEX = charts.RISK_COLORS


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_outputs():
    out = config.OUTPUT_DIR
    if not (out / config.SCORED_FILE).exists() or not (out / config.METRICS_FILE).exists():
        from pipeline import run_pipeline

        run_pipeline(config.RAW_FILE, out, verbose=False)
    scored = pd.read_csv(out / config.SCORED_FILE, parse_dates=["timestamp"])
    if "display_activity" not in scored.columns:
        scored["display_activity"] = scored["activity"]
    scored["flagged"] = scored["flagged"].astype(bool)
    metrics = json.loads((out / config.METRICS_FILE).read_text())
    return scored, metrics


@st.cache_data(show_spinner=False)
def reference_features():
    from evidence.features import build_features

    return build_features(normalize_raw_csv(config.RAW_FILE))


with st.spinner("Loading evidence and model results..."):
    try:
        scored, metrics = load_outputs()
    except Exception as exc:  # pragma: no cover - shown to the user
        st.error(f"Could not load the evidence data: {exc}. Run `python pipeline.py` in the project folder, then reload.")
        st.stop()

models = metrics["models"]
summary = metrics["summary"]


# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## Investigation filters")
    min_d, max_d = scored["timestamp"].min().date(), scored["timestamp"].max().date()
    picked = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(picked, (list, tuple)) and len(picked) == 2:
        d_from, d_to = picked
    else:
        d_from, d_to = min_d, max_d

    sources = sorted(scored["source_type"].dropna().unique())
    pick_sources = st.multiselect("Evidence sources", sources, default=sources)
    pick_levels = st.multiselect("Risk levels", config.RISK_ORDER, default=config.RISK_ORDER)
    pick_users = st.multiselect("Users", sorted(scored["user"].unique()), placeholder="All users")
    min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)
    st.markdown(
        "<div class='note'>Events are flagged when the hybrid risk score is <b>0.60 or higher</b>. "
        "The score blends a Random Forest probability (70%) with an Isolation Forest anomaly score (30%).</div>",
        unsafe_allow_html=True,
    )

mask = (
    scored["timestamp"].dt.date.between(d_from, d_to)
    & scored["source_type"].isin(pick_sources)
    & scored["risk_level"].isin(pick_levels)
    & (scored["risk_score"] >= min_risk)
)
if pick_users:
    mask &= scored["user"].isin(pick_users)
view = scored[mask].copy()


# --------------------------------------------------------------------------- #
# Hero: case tag
# --------------------------------------------------------------------------- #
n_flag = int(view["flagged"].sum())
pct = (n_flag / len(view) * 100) if len(view) else 0.0
start = pd.Timestamp(summary["start"]).strftime("%d %b %Y")
end = pd.Timestamp(summary["end"]).strftime("%d %b %Y")
best_f1 = models["hybrid_risk"]["f1"]

st.markdown(
    f"""
<div class="case-tag">
  <div class="tape"></div>
  <span class="tape-label">Evidence file: CERT insider threat dataset</span>
  <div class="case-body">
    <div>
      <div class="case-title">Digital evidence timeline reconstruction</div>
      <p class="case-sub">{summary['events']:,} log events from {summary['users']} employees on {summary['pcs']} workstations,
      rebuilt into one chronological record between {start} and {end}. Machine learning ranks every event
      so investigators start with the most suspicious activity.</p>
    </div>
    <div class="case-facts">
      <div class="fact"><b>{len(view):,}</b><span>events in view</span></div>
      <div class="fact alert"><b>{n_flag:,}</b><span>flagged for review ({pct:.1f}%)</span></div>
      <div class="fact"><b>{view['user'].nunique() if len(view) else 0}</b><span>users involved</span></div>
      <div class="fact"><b>{best_f1:.3f}</b><span>F1 score of the risk model</span></div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if view.empty:
    st.warning("No events match these filters. Widen the date range or add more sources and risk levels in the sidebar.")
    st.stop()

show_chart(charts.density_strip(view), key="hero_strip")
st.markdown("<div class='strip-caption'>Weekly event volume across the investigation period. Red marks flagged events.</div>",
            unsafe_allow_html=True)

tab_over, tab_time, tab_people, tab_models, tab_locker = st.tabs(
    ["Overview", "Timeline reconstruction", "People of interest", "Model performance", "Evidence locker"]
)


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Activity over time")
        show_chart(charts.activity_over_time(view), key="ov_time")
    with c2:
        st.markdown("### Risk distribution")
        show_chart(charts.risk_histogram(view), key="ov_hist")
    c3, c4 = st.columns([3, 2])
    with c3:
        st.markdown("### When flagged activity happens")
        show_chart(charts.hour_weekday_heatmap(view), key="ov_heat")
    with c4:
        st.markdown("### Flagged events by source")
        show_chart(charts.source_breakdown(view), key="ov_src")

    off = view[view["flagged"]]
    if len(off):
        off_rate = off["off_hours"].mean() * 100
        top_act = off["display_activity"].value_counts().idxmax()
        st.markdown(
            f"<div class='note'><b>{off_rate:.0f}%</b> of flagged events happen outside 07:00–19:00. "
            f"The most common flagged activity is <b>{esc(top_act)}</b>.</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Timeline reconstruction
# --------------------------------------------------------------------------- #
def event_chain(events: pd.DataFrame, limit: int = 40) -> str:
    rows = []
    for _, e in events.head(limit).iterrows():
        level = str(e["risk_level"])
        rows.append(
            f"""<div class="ev" style="--pip:{RISK_HEX.get(level, '#3E5778')}">
  <div class="ev-top"><span class="ev-what">{esc(e['display_activity'])}</span>
  <span class="risk-chip {esc(level)}">{esc(level)} {e['risk_score']:.2f}</span></div>
  <div class="ev-meta"><span class="ev-when">{e['timestamp']:%a %d %b %Y, %H:%M:%S}</span> &nbsp; {esc(e['user'])} on {esc(e['pc'])}
  &nbsp; ({esc(e['source_type'])}) &nbsp; <span class="ev-id">{esc(e.get('event_id', ''))}</span></div>
</div>"""
        )
    return "<div class='chain'>" + "".join(rows) + "</div>"


with tab_time:
    profiles_view = user_risk_profiles(view)
    users_ranked = profiles_view["user"].tolist()
    left, right = st.columns([2, 1])
    with left:
        who = st.selectbox(
            "Reconstruct the timeline for",
            users_ranked,
            format_func=lambda u: f"{u}  ({profiles_view.set_index('user').loc[u, 'risk_level']} threat)",
        )
    with right:
        gap_min = st.number_input("New session after (minutes idle)", min_value=5, max_value=720, value=30, step=5)

    tl = reconstruct_timeline(view, user=who)
    sessions = summarize_sessions(tl, gap_minutes=gap_min)
    prof = profiles_view.set_index("user").loc[who]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Events", f"{int(prof['events']):,}")
    m2.metric("Flagged", f"{int(prof['flagged_events']):,}", f"{prof['flag_rate'] * 100:.0f}% of events", delta_color="off")
    m3.metric("Sessions", f"{len(sessions):,}")
    m4.metric("Workstations", int(prof["distinct_pcs"]))
    m5.metric("Peak risk", f"{prof['max_risk']:.2f}")

    st.markdown("### Reconstructed timeline")
    st.caption("Each lane is one evidence source. Larger, warmer dots are riskier. Drag the slider under the chart to zoom in.")
    show_chart(charts.timeline_scatter(tl), key="tl_scatter")

    a, b = st.columns([3, 2])
    with a:
        order = st.radio("Event chain order", ["Chronological", "Highest risk first"], horizontal=True)
        chain_src = tl if order == "Chronological" else tl.sort_values("risk_score", ascending=False)
        only_flag = st.checkbox("Only flagged events", value=True)
        if only_flag:
            chain_src = chain_src[chain_src["flagged"]]
        if chain_src.empty:
            st.info("This user has no flagged events in the current view. Clear the checkbox to see all of their events.")
        else:
            st.markdown(event_chain(chain_src), unsafe_allow_html=True)
            if len(chain_src) > 40:
                st.caption(f"Showing 40 of {len(chain_src):,} events. The full list is in the Evidence locker tab.")
    with b:
        st.markdown("### Sessions")
        show_table(
            sessions[["start", "events", "duration_min", "max_risk", "flagged", "pcs"]].sort_values("max_risk", ascending=False),
            column_config={
                "start": st.column_config.DatetimeColumn("Start", format="DD MMM YYYY HH:mm"),
                "events": "Events",
                "duration_min": st.column_config.NumberColumn("Minutes", format="%.0f"),
                "max_risk": st.column_config.ProgressColumn("Peak risk", min_value=0.0, max_value=1.0, format="%.2f"),
                "flagged": "Flagged",
                "pcs": "PCs",
            },
            height=330,
        )
        st.markdown("### Silent periods")
        gaps = detect_gaps(tl, min_gap_hours=72)
        if gaps.empty:
            st.caption("No inactivity longer than 3 days.")
        else:
            show_table(
                gaps.head(8)[["gap_start", "gap_end", "gap_hours"]],
                column_config={
                    "gap_start": st.column_config.DatetimeColumn("Last seen", format="DD MMM YYYY HH:mm"),
                    "gap_end": st.column_config.DatetimeColumn("Seen again", format="DD MMM YYYY HH:mm"),
                    "gap_hours": st.column_config.NumberColumn("Hours idle", format="%.0f"),
                },
            )


# --------------------------------------------------------------------------- #
# People of interest
# --------------------------------------------------------------------------- #
with tab_people:
    profiles = user_risk_profiles(view)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Threat ranking")
        show_table(
            profiles[["user", "risk_level", "threat_score", "events", "flagged_events", "flag_rate", "off_hours_ratio", "distinct_pcs", "last_seen"]],
            column_config={
                "user": "User",
                "risk_level": "Level",
                "threat_score": st.column_config.ProgressColumn("Threat score", min_value=0.0, max_value=1.0, format="%.2f"),
                "events": "Events",
                "flagged_events": "Flagged",
                "flag_rate": st.column_config.ProgressColumn("Flag rate", min_value=0.0, max_value=1.0, format="%.2f"),
                "off_hours_ratio": st.column_config.NumberColumn("Off-hours", format="%.2f"),
                "distinct_pcs": "PCs",
                "last_seen": st.column_config.DatetimeColumn("Last seen", format="DD MMM YYYY"),
            },
            height=440,
        )
    with c2:
        st.markdown("### Volume against risk")
        st.caption("Bubble size shows the share of a user's events that were flagged.")
        show_chart(charts.user_bubble(profiles), key="ppl_bubble")
    st.markdown(
        "<div class='note'>Threat score = 0.5 × peak event risk + 0.3 × average risk + 0.2 × flag rate. "
        "Pick a user in <b>Timeline reconstruction</b> to see their events in order.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #
with tab_models:
    order_keys = ["baseline_rf", "enhanced_rf", "enhanced_rf_chronological", "isolation_forest", "hybrid_risk"]
    order_keys = [k for k in order_keys if k in models]
    best = max(order_keys, key=lambda k: models[k]["f1"])
    cards = []
    for k in order_keys:
        m = models[k]
        cards.append(
            f"""<div class="model-card {'best' if k == best else ''}"><h4>{esc(m['name'])}</h4><p>{esc(m['description'])}</p>
<div class="f1">{m['f1']:.3f}<small>F1</small></div>
<div class="row"><span>Accuracy {m['accuracy']:.3f}</span><span>AUC {m.get('roc_auc', float('nan')):.3f}</span></div>
<div class="row"><span>Precision {m['precision']:.3f}</span><span>Recall {m['recall']:.3f}</span></div></div>"""
        )
    st.markdown("<div class='model-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

    base, enh = models.get("baseline_rf"), models.get("enhanced_rf")
    if base and enh:
        st.markdown(
            f"<div class='note'>Behavioural features raised F1 from <b>{base['f1']:.3f}</b> to <b>{enh['f1']:.3f}</b> "
            f"and cut false alarms from <b>{base['confusion_matrix'][0][1]}</b> to <b>{enh['confusion_matrix'][0][1]}</b> on the same test events.</div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Metric comparison")
        show_chart(charts.metric_comparison(models, order_keys), key="mdl_cmp")
    with c2:
        st.markdown("### ROC curves")
        show_chart(charts.roc_curves(models, order_keys), key="mdl_roc")

    c3, c4 = st.columns([2, 3])
    with c3:
        st.markdown("### Confusion matrix")
        pick = st.selectbox("Model", order_keys, index=order_keys.index(best), format_func=lambda k: models[k]["name"])
        show_chart(charts.confusion_heatmap(models[pick]["confusion_matrix"]), key="mdl_cm")
        st.caption(f"Evaluation: {models[pick]['split']}, {models[pick]['support']:,} events.")
    with c4:
        st.markdown("### What the model relies on")
        st.caption("Yellow bars are the new behavioural features.")
        show_chart(charts.importance_bars(models["enhanced_rf"]["feature_importance"]), key="mdl_imp")

    st.markdown(
        "<div class='note warn'><b>Leakage control.</b> In the raw export every label-1 row uses the 9-field layout and every "
        "label-0 row uses the 7-field layout. Record format, resource, case ID and event ID would reveal the answer, so they "
        "are never given to the models. The chronological test (train on the past, predict the future) is the strictest check: "
        "its high AUC shows the ranking still holds, while the lower F1 shows behaviour drifts over time.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Evidence locker
# --------------------------------------------------------------------------- #
with tab_locker:
    st.markdown("### Search the evidence")
    q = st.text_input("Search by user, PC, activity or event ID", placeholder="e.g. WDD0366, PC-0155, wikileaks")
    table = view
    if q:
        needle = q.strip().lower()
        hay = (view["user"] + " " + view["pc"] + " " + view["activity"] + " " + view["event_id"].astype(str)).str.lower()
        table = view[hay.str.contains(needle, regex=False)]
    st.caption(f"{len(table):,} matching events")
    cols = ["timestamp", "user", "pc", "source_type", "display_activity", "risk_level", "risk_score", "rf_probability", "anomaly_status", "event_id"]
    show_table(
        table.sort_values("timestamp")[cols],
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="DD MMM YYYY HH:mm:ss"),
            "user": "User", "pc": "PC", "source_type": "Source", "display_activity": "Activity", "risk_level": "Level",
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0.0, max_value=1.0, format="%.2f"),
            "rf_probability": st.column_config.NumberColumn("RF probability", format="%.2f"),
            "anomaly_status": "Isolation Forest", "event_id": "Event ID",
        },
        height=420,
    )
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download these events (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_evidence.csv", mime="text/csv", **_stretch(st.download_button))
    with d2:
        rep = build_investigation_report(table)
        st.download_button(f"Download investigation report ({len(rep):,} flagged)", rep.to_csv(index=False).encode("utf-8"),
                           file_name="anomaly_investigation_report.csv", mime="text/csv", **_stretch(st.download_button))

    st.markdown("### Analyse new evidence")
    st.caption("Upload a raw export in the same CERT layout. Events are scored by models trained on the case dataset. "
               "A previously generated report CSV can also be opened here.")
    up = st.file_uploader("Raw evidence or report CSV", type=["csv"])
    if up is not None:
        raw = up.getvalue()
        head = raw[:400].decode("utf-8-sig", errors="ignore").splitlines()[0] if raw else ""
        try:
            if "risk_score" in head or "anomaly_score" in head:
                rep_df = load_report(raw)
                st.success(f"Opened report with {len(rep_df):,} events.")
                show_table(rep_df, height=360)
            else:
                from pipeline import score_new_evidence

                with st.spinner("Normalising and scoring the uploaded evidence..."):
                    new = normalize_raw_csv(raw)
                    if new.empty:
                        raise ValueError("no rows matched the 7- or 9-field evidence layout")
                    new_scored = score_new_evidence(new, reference_features())
                n_new = int(new_scored["flagged"].sum())
                st.success(f"Scored {len(new_scored):,} events. {n_new:,} flagged for review"
                           + (f"; {new.attrs.get('skipped_rows', 0)} rows skipped." if new.attrs.get("skipped_rows") else "."))
                show_chart(charts.timeline_scatter(new_scored), key="up_tl")
                show_table(new_scored[cols].sort_values("risk_score", ascending=False), height=360,
                           column_config={"risk_score": st.column_config.ProgressColumn("Risk", min_value=0.0, max_value=1.0, format="%.2f")})
                st.download_button("Download scored upload (CSV)", new_scored.to_csv(index=False).encode("utf-8"),
                                   file_name="scored_upload.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not analyse this file: {exc}. Check that it is a CERT-style CSV with a header row.")

st.markdown(
    f"<div style='color:#8EA3BF;font-size:.8rem;margin-top:2rem;border-top:1px solid rgba(142,163,191,.18);padding-top:.6rem'>"
    f"Models generated {esc(summary.get('generated_at', ''))}. Regenerate with <code>python pipeline.py</code>.</div>",
    unsafe_allow_html=True,
)
