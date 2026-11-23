"""Project paths and shared constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

RAW_FILE = DATA_DIR / "cert_preprocessed_ml_comparison.csv"

NORMALIZED_FILE = "normalized_evidence.csv"
SCORED_FILE = "scored_evidence.csv"
REPORT_FILE = "anomaly_investigation_report.csv"
USER_RISK_FILE = "user_risk_profiles.csv"
METRICS_FILE = "model_metrics.json"

RANDOM_STATE = 42

# Columns that reveal the label directly (the label is perfectly correlated
# with the record layout, and only labelled rows carry a resource value).
# They must NEVER be used as model features.
LEAKAGE_COLUMNS = [
    "label",
    "original_case_label",
    "record_format",
    "resource",
    "record_number",
    "event_id",
    "case_id",
]

# Risk-level bands applied to the hybrid risk score (0..1).
RISK_BANDS = [
    (0.80, "Critical"),
    (0.60, "High"),
    (0.35, "Medium"),
    (0.00, "Low"),
]
RISK_ORDER = ["Critical", "High", "Medium", "Low"]
FLAG_THRESHOLD = 0.60

# Business hours used for the off-hours feature.
WORK_START_HOUR = 7
WORK_END_HOUR = 19
