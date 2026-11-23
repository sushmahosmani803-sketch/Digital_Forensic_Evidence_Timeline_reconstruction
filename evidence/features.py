"""Leak-free time and behavioural feature engineering.

All features are computed from observable event information only
(who, where, what, when). Label-revealing columns are excluded; see
``config.LEAKAGE_COLUMNS``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LEAKAGE_COLUMNS, WORK_END_HOUR, WORK_START_HOUR

CATEGORICAL_FEATURES = ["user", "pc", "activity", "source_type"]
BASELINE_NUMERIC = ["hour", "day_of_week"]
NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "minute_of_day",
    "is_weekend",
    "off_hours",
    "gap_prev_sec",
    "gap_next_sec",
    "user_day_events",
    "foreign_pc",
]
ISOLATION_NUMERIC = [
    "hour",
    "day_of_week",
    "is_weekend",
    "off_hours",
    "log_gap_prev",
    "user_day_events",
    "foreign_pc",
]

assert not set(CATEGORICAL_FEATURES + NUMERIC_FEATURES) & set(LEAKAGE_COLUMNS)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar/clock features. Returns a new frame."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    out["timestamp"] = ts
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["off_hours"] = ((out["hour"] < WORK_START_HOUR) | (out["hour"] >= WORK_END_HOUR)).astype(int)
    return out


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-user sequence features while preserving the original row order.

    * gap_prev_sec / gap_next_sec - seconds to the user's previous / next event
    * user_day_events             - events by the same user on the same day
    * foreign_pc                  - 1 if the PC is not the user's most-used PC
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    order = out.sort_values(["user", "timestamp"], kind="mergesort")
    grp = order.groupby("user")["timestamp"]
    out["gap_prev_sec"] = grp.diff().dt.total_seconds()
    out["gap_next_sec"] = (-grp.diff(-1)).dt.total_seconds()

    day = out["timestamp"].dt.normalize()
    out["user_day_events"] = out.groupby([out["user"], day])["user"].transform("size").astype(int)

    usual_pc = out.groupby("user")["pc"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
    out["foreign_pc"] = (out["pc"] != out["user"].map(usual_pc)).astype(int)

    out["log_gap_prev"] = np.log1p(out["gap_prev_sec"].clip(lower=0))
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature table (time + sequence)."""
    return add_sequence_features(add_time_features(df))
