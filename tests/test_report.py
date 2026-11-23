import pandas as pd

from evidence.report import REPORT_COLUMNS, build_investigation_report, load_report

CORRUPTED = (
    "$files | Where-Object { Test-Path $_ }timestamp,user,pc,activity,source_type,case_id,anomaly_score,anomaly_status\n"
    "2010-01-18 20:31:14,AJF0370,PC-1609,Logon,logon,164,-0.0005,Potential anomaly\n"
    "2010-03-11 01:00:00,AJF0370,PC-1609,Logon,logon,164,-0.0196,Potential anomaly\n"
)


def test_load_report_repairs_corrupted_header(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text(CORRUPTED, encoding="utf-8")
    df = load_report(p)
    assert df.columns[0] == "timestamp"
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert len(df) == 2


def test_load_report_clean_file_and_bytes(tmp_path):
    clean = CORRUPTED.split("}", 1)[1]
    p = tmp_path / "c.csv"
    p.write_text(clean)
    assert load_report(p).columns[0] == "timestamp"
    assert len(load_report(clean.encode())) == 2


def test_original_corrupted_report_now_loads():
    """The report shipped with the original project had a broken header."""
    from evidence.config import ROOT

    legacy = ROOT / "legacy_scripts" / "original_anomaly_investigation_report.csv"
    if legacy.exists():
        df = load_report(legacy)
        assert "timestamp" in df.columns and len(df) > 0


def test_build_investigation_report(sample_df):
    d = sample_df.assign(
        risk_score=[0.9, 0.2, 0.7, 0.65, 0.1, 0.0, 0.61],
        rf_probability=0.5,
        anomaly_score=0.0,
        anomaly_status="Normal pattern",
        risk_level="High",
    )
    r = build_investigation_report(d, threshold=0.6)
    assert len(r) == 4
    assert (r["risk_score"] >= 0.6).all()
    assert list(r.columns) == [c for c in REPORT_COLUMNS if c in r.columns]
    # Sorted by user then time.
    assert r["user"].tolist() == sorted(r["user"].tolist())
    for _, grp in r.groupby("user"):
        assert grp["timestamp"].is_monotonic_increasing


def test_report_falls_back_to_isolation_prediction(sample_df):
    d = sample_df.assign(anomaly_prediction=[-1, 1, 1, -1, 1, 1, 1])
    assert len(build_investigation_report(d)) == 2
