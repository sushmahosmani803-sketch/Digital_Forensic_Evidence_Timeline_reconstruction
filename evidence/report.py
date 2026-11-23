"""Investigation report generation and robust loading."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .config import FLAG_THRESHOLD

REPORT_COLUMNS = [
    "timestamp",
    "user",
    "pc",
    "activity",
    "source_type",
    "case_id",
    "risk_score",
    "risk_level",
    "rf_probability",
    "anomaly_score",
    "anomaly_status",
    "event_id",
]

KNOWN_FIRST_COLUMNS = ["timestamp", "record_number", "sequence"]


def _repair_header(text: str) -> str:
    """Remove junk accidentally prepended to the header line.

    The original report started with a pasted PowerShell fragment
    (``$files | Where-Object { Test-Path $_ }timestamp,user,...``), which
    broke ``df["timestamp"]`` in the old dashboard.
    """
    if not text:
        return text
    first, sep, rest = text.partition("\n")
    for col in KNOWN_FIRST_COLUMNS:
        pos = first.find(col + ",")
        if pos > 0:
            first = first[pos:]
            break
    return first + sep + rest


def load_report(source: str | Path | bytes) -> pd.DataFrame:
    """Load a report CSV, repairing a corrupted header if necessary."""
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        text = source.decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(_repair_header(text)))
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def build_investigation_report(scored: pd.DataFrame, threshold: float = FLAG_THRESHOLD) -> pd.DataFrame:
    """Flagged events, ordered by user then time (investigator-friendly)."""
    d = scored.copy()
    if "risk_score" in d.columns:
        mask = d["risk_score"] >= threshold
    else:
        mask = d["anomaly_prediction"] == -1
    d = d[mask]
    cols = [c for c in REPORT_COLUMNS if c in d.columns]
    d = d.sort_values(["user", "timestamp"], kind="mergesort")[cols]
    for c in ("risk_score", "rf_probability", "anomaly_score"):
        if c in d.columns:
            d[c] = d[c].round(4)
    return d.reset_index(drop=True)
