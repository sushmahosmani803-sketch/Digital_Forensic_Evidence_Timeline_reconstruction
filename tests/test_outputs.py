"""Checks on the pre-generated outputs that ship with the project and feed the dashboard."""

import json

import pandas as pd
import pytest

from evidence import config

OUT = config.OUTPUT_DIR

pytestmark = pytest.mark.skipif(not (OUT / config.SCORED_FILE).exists(), reason="run `python pipeline.py` first")


def test_shipped_scored_evidence():
    s = pd.read_csv(OUT / config.SCORED_FILE)
    assert len(s) == 4615
    assert s["risk_score"].between(0, 1).all()
    assert set(s["risk_level"]) <= set(config.RISK_ORDER)


def test_shipped_report_loads_and_matches():
    from evidence.report import load_report

    r = load_report(OUT / config.REPORT_FILE)
    s = pd.read_csv(OUT / config.SCORED_FILE)
    assert r.columns[0] == "timestamp"
    assert len(r) == int(s["flagged"].sum())


def test_shipped_metrics_meet_targets():
    m = json.loads((OUT / config.METRICS_FILE).read_text())["models"]
    assert m["enhanced_rf"]["f1"] >= 0.88
    assert m["enhanced_rf"]["f1"] > m["baseline_rf"]["f1"]
    assert m["hybrid_risk"]["roc_auc"] >= 0.97


def test_shipped_user_profiles():
    p = pd.read_csv(OUT / config.USER_RISK_FILE)
    assert p["user"].is_unique
    assert p["threat_score"].is_monotonic_decreasing
