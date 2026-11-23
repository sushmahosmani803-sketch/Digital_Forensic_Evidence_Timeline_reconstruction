"""Timeline reconstruction utilities."""

from __future__ import annotations

import pandas as pd

from .config import FLAG_THRESHOLD
from .models import risk_level


def reconstruct_timeline(
    df: pd.DataFrame,
    user: str | None = None,
    case_id=None,
    pc: str | None = None,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Return events in strict chronological order, optionally filtered."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if user is not None:
        out = out[out["user"] == user]
    if case_id is not None:
        out = out[out["case_id"].astype(str) == str(case_id)]
    if pc is not None:
        out = out[out["pc"] == pc]
    if start is not None:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["timestamp"] <= pd.Timestamp(end)]
    out = out.sort_values(["timestamp", "user"], kind="mergesort").reset_index(drop=True)
    out.insert(0, "sequence", range(1, len(out) + 1))
    return out


def assign_sessions(df: pd.DataFrame, gap_minutes: float = 30) -> pd.DataFrame:
    """Group each user's events into sessions separated by > gap_minutes of inactivity."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values(["user", "timestamp"], kind="mergesort")
    gap = out.groupby("user")["timestamp"].diff()
    new_session = gap.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    out["session_no"] = new_session.groupby(out["user"]).cumsum().astype(int)
    out["session_id"] = out["user"].astype(str) + "-S" + out["session_no"].astype(str).str.zfill(3)
    return out.sort_index()


def summarize_sessions(df: pd.DataFrame, gap_minutes: float = 30) -> pd.DataFrame:
    s = assign_sessions(df, gap_minutes)
    agg = {
        "user": ("user", "first"),
        "start": ("timestamp", "min"),
        "end": ("timestamp", "max"),
        "events": ("timestamp", "size"),
        "pcs": ("pc", lambda x: ", ".join(sorted(set(map(str, x))))),
        "sources": ("source_type", lambda x: ", ".join(sorted(set(map(str, x))))),
    }
    if "risk_score" in s.columns:
        agg["max_risk"] = ("risk_score", "max")
        agg["flagged"] = ("risk_score", lambda x: int((x >= FLAG_THRESHOLD).sum()))
    out = s.groupby("session_id").agg(**agg).reset_index()
    out["duration_min"] = ((out["end"] - out["start"]).dt.total_seconds() / 60).round(1)
    return out.sort_values("start").reset_index(drop=True)


def detect_gaps(df: pd.DataFrame, min_gap_hours: float = 72) -> pd.DataFrame:
    """Find periods of inactivity per user longer than min_gap_hours."""
    d = df[["user", "timestamp"]].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna().sort_values(["user", "timestamp"], kind="mergesort")
    d["previous"] = d.groupby("user")["timestamp"].shift()
    d["gap_hours"] = (d["timestamp"] - d["previous"]).dt.total_seconds() / 3600
    gaps = d[d["gap_hours"] > min_gap_hours]
    gaps = gaps.rename(columns={"previous": "gap_start", "timestamp": "gap_end"})
    gaps = gaps[["user", "gap_start", "gap_end", "gap_hours"]].copy()
    gaps["gap_hours"] = gaps["gap_hours"].round(1)
    return gaps.sort_values("gap_hours", ascending=False).reset_index(drop=True)


def user_risk_profiles(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-user summary for the suspects leaderboard."""
    d = scored.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    if "flagged" not in d.columns:
        d["flagged"] = d["risk_score"] >= FLAG_THRESHOLD
    g = d.groupby("user")
    out = pd.DataFrame(
        {
            "events": g.size(),
            "flagged_events": g["flagged"].sum().astype(int),
            "mean_risk": g["risk_score"].mean(),
            "max_risk": g["risk_score"].max(),
            "off_hours_ratio": g["off_hours"].mean() if "off_hours" in d else 0.0,
            "distinct_pcs": g["pc"].nunique(),
            "first_seen": g["timestamp"].min(),
            "last_seen": g["timestamp"].max(),
        }
    )
    out["flag_rate"] = out["flagged_events"] / out["events"]
    # Threat score: weighted blend of how risky and how consistently risky a user is.
    out["threat_score"] = (0.5 * out["max_risk"] + 0.3 * out["mean_risk"] + 0.2 * out["flag_rate"]).clip(0, 1)
    out["risk_level"] = out["threat_score"].map(risk_level)
    out = out.reset_index().sort_values(["threat_score", "flagged_events"], ascending=False)
    return out.reset_index(drop=True)
