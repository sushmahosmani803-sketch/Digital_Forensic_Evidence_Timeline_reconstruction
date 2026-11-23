import json

import pandas as pd
import pytest

from evidence import config
from evidence.normalize import normalize_raw_csv
from pipeline import run_pipeline, score_new_evidence

EXPECTED_FILES = [
    config.NORMALIZED_FILE,
    config.SCORED_FILE,
    config.REPORT_FILE,
    config.USER_RISK_FILE,
    config.METRICS_FILE,
]


@pytest.fixture(scope="module")
def pipeline_run(small_raw_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    payload = run_pipeline(small_raw_file, out, n_estimators=60, verbose=False)
    return out, payload


def test_pipeline_writes_all_outputs(pipeline_run):
    out, _ = pipeline_run
    for name in EXPECTED_FILES:
        assert (out / name).exists(), name
        assert (out / name).stat().st_size > 0


def test_pipeline_metrics_structure(pipeline_run):
    out, payload = pipeline_run
    on_disk = json.loads((out / config.METRICS_FILE).read_text())
    assert on_disk["summary"]["events"] == payload["summary"]["events"]
    for key in ["baseline_rf", "enhanced_rf", "enhanced_rf_chronological", "isolation_forest", "hybrid_risk"]:
        m = on_disk["models"][key]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert 0.0 <= m[metric] <= 1.0
        assert len(m["confusion_matrix"]) == 2
    assert "feature_importance" in on_disk["models"]["enhanced_rf"]


def test_pipeline_scored_events_consistent(pipeline_run, small_raw_file):
    out, payload = pipeline_run
    scored = pd.read_csv(out / config.SCORED_FILE, parse_dates=["timestamp"])
    raw_rows = len(small_raw_file.read_text().strip().splitlines()) - 1
    assert len(scored) == raw_rows == payload["summary"]["events"]
    assert scored["timestamp"].is_monotonic_increasing
    assert scored["risk_score"].between(0, 1).all()
    assert scored["rf_probability"].between(0, 1).all()
    report = pd.read_csv(out / config.REPORT_FILE)
    assert len(report) == int(scored["flagged"].sum()) == payload["summary"]["flagged"]


def test_score_new_evidence(full_features, raw_sample_file):
    new = normalize_raw_csv(raw_sample_file)
    scored = score_new_evidence(new, full_features, n_estimators=40)
    assert len(scored) == len(new)
    assert scored["risk_score"].between(0, 1).all()
    assert set(scored["risk_level"]) <= set(config.RISK_ORDER)
    assert scored["timestamp"].is_monotonic_increasing
