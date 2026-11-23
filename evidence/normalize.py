"""Normalisation of the raw CERT evidence export.

The raw file mixes two record layouts:

* standard (9 fields):
  record_number, source, event_id, timestamp, user, pc, activity, resource, "case;label"
* short (7 fields):
  record_number, event_id, timestamp, pc, activity, user, "case;label"

Both are mapped to one schema. For short rows the source type is inferred
from the activity text.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd

SCHEMA = [
    "record_number",
    "source_type",
    "event_id",
    "timestamp",
    "user",
    "pc",
    "activity",
    "resource",
    "case_id",
    "label",
    "original_case_label",
    "record_format",
]


def infer_source_type(activity: str) -> str:
    """Infer the log source from an activity description."""
    a = (activity or "").strip().lower()
    if not a:
        return "unknown"
    if "http" in a or a.startswith("www."):
        return "http"
    if "email" in a:
        return "email"
    if "file" in a:
        return "file"
    if a in {"logon", "logoff"}:
        return "logon"
    if a in {"connect", "disconnect"}:
        return "device"
    return "unknown"


def split_case_label(value: str) -> tuple[str, str]:
    """Split the combined "case;label" field, e.g. "17;1" -> ("17", "1")."""
    parts = (value or "").split(";")
    case_id = parts[0].strip() if parts else ""
    label = parts[1].strip() if len(parts) > 1 else ""
    return case_id, label


def display_activity(activity: str) -> str:
    """Return a short human-readable activity (URLs are reduced to their domain)."""
    a = (activity or "").strip()
    if a.lower().startswith(("http://", "https://")):
        host = urlparse(a).netloc or a
        return f"visit {host}"
    return a


def normalize_row(row: list[str]) -> dict | None:
    """Normalise one raw CSV row. Returns None for unsupported layouts."""
    if len(row) == 9:
        record_number, source, event_id, timestamp, user, pc, activity, resource, case_label = row
        record_format = "standard"
    elif len(row) == 7:
        record_number, event_id, timestamp, pc, activity, user, case_label = row
        source = infer_source_type(activity)
        resource = ""
        record_format = "short"
    else:
        return None

    case_id, label = split_case_label(case_label)
    return {
        "record_number": record_number.strip(),
        "source_type": (source or "").strip().lower() or infer_source_type(activity),
        "event_id": event_id.strip(),
        "timestamp": timestamp.strip(),
        "user": user.strip(),
        "pc": pc.strip(),
        "activity": activity.strip(),
        "resource": resource.strip(),
        "case_id": case_id,
        "label": label,
        "original_case_label": case_label,
        "record_format": record_format,
    }


def normalize_rows(rows: Iterable[list[str]]) -> tuple[pd.DataFrame, int]:
    """Normalise an iterable of raw rows. Returns (frame, skipped_count)."""
    records, skipped = [], 0
    for row in rows:
        if not row or all(not c.strip() for c in row):
            continue
        rec = normalize_row(row)
        if rec is None:
            skipped += 1
        else:
            records.append(rec)

    df = pd.DataFrame(records, columns=SCHEMA)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df["case_id"] = pd.to_numeric(df["case_id"], errors="coerce").astype("Int64")
    df["resource"] = df["resource"].replace("", pd.NA)
    df["display_activity"] = df["activity"].map(display_activity)
    return df, skipped


def normalize_raw_csv(source: str | Path | io.IOBase | bytes) -> pd.DataFrame:
    """Read and normalise a raw evidence CSV (path, bytes or file object)."""
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8-sig", newline="") as fh:
            text = fh.read()
    elif isinstance(source, bytes):
        text = source.decode("utf-8-sig")
    else:
        data = source.read()
        text = data.decode("utf-8-sig") if isinstance(data, bytes) else data

    reader = csv.reader(io.StringIO(text))
    next(reader, None)  # header
    df, skipped = normalize_rows(reader)
    df.attrs["skipped_rows"] = skipped
    return df
