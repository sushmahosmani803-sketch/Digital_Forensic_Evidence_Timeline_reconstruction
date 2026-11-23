import numpy as np
import pytest

from evidence.config import FLAG_THRESHOLD
from evidence.features import BASELINE_NUMERIC, CATEGORICAL_FEATURES
from evidence.models import (
    apply_risk,
    chronological_split,
    classification_metrics,
    combine_risk,
    evaluate_classifier,
    evaluate_isolation,
    feature_columns,
    feature_importance,
    iso_scores_to_risk,
    risk_level,
    score_isolation,
    stratified_split,
    train_isolation_forest,
    train_random_forest,
)


@pytest.fixture(scope="module")
def splits(full_features):
    return stratified_split(full_features)


@pytest.fixture(scope="module")
def enhanced(splits):
    train, _ = splits
    return train_random_forest(train, n_estimators=150)


def test_stratified_split_preserves_ratio(splits, full_features):
    train, test = splits
    assert len(train) + len(test) == len(full_features)
    assert abs(len(test) / len(full_features) - 0.2) < 0.01
    assert abs(train["label"].mean() - test["label"].mean()) < 0.01
    assert set(train.index).isdisjoint(test.index)


def test_chronological_split_has_no_time_overlap(full_features):
    train, test = chronological_split(full_features)
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(train) == int(len(full_features) * 0.8)


def test_model_never_uses_leaky_columns(enhanced):
    cols = feature_columns(enhanced)
    for leaky in ["label", "record_format", "resource", "original_case_label"]:
        assert leaky not in cols


def test_enhanced_rf_quality(enhanced, splits):
    _, test = splits
    m = evaluate_classifier(enhanced, test)
    assert m["f1"] >= 0.88
    assert m["accuracy"] >= 0.93
    assert m["roc_auc"] >= 0.97
    assert np.array(m["confusion_matrix"]).sum() == len(test)


def test_enhanced_beats_baseline(enhanced, splits):
    train, test = splits
    base = train_random_forest(train, CATEGORICAL_FEATURES, BASELINE_NUMERIC, n_estimators=150)
    assert evaluate_classifier(enhanced, test)["f1"] > evaluate_classifier(base, test)["f1"] + 0.08


def test_feature_importance_aggregates_to_original_features(enhanced):
    imp = feature_importance(enhanced)
    assert set(imp) == set(feature_columns(enhanced))
    assert sum(imp.values()) == pytest.approx(1.0, abs=1e-6)
    assert list(imp.values()) == sorted(imp.values(), reverse=True)


def test_isolation_forest_flags_about_contamination(full_features):
    iso = train_isolation_forest(full_features, contamination=0.05)
    scored = score_isolation(iso, full_features)
    rate = (scored["anomaly_prediction"] == -1).mean()
    assert 0.03 <= rate <= 0.07
    assert scored["iso_risk"].between(0, 1).all()
    assert set(scored["anomaly_status"]) == {"Normal pattern", "Potential anomaly"}
    m = evaluate_isolation(iso, full_features)
    assert 0 <= m["f1"] <= 1


def test_iso_scores_to_risk_inverts_and_scales():
    r = iso_scores_to_risk([0.2, 0.0, -0.2])
    assert r.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert iso_scores_to_risk([0.1, 0.1]).tolist() == [0.0, 0.0]
    assert iso_scores_to_risk([-1.0], reference=[0.0, 0.2]).tolist() == [1.0]  # clipped
    assert iso_scores_to_risk([]).size == 0


@pytest.mark.parametrize(
    "score,level", [(0.95, "Critical"), (0.80, "Critical"), (0.7, "High"), (0.6, "High"), (0.4, "Medium"), (0.1, "Low"), (0.0, "Low")]
)
def test_risk_level_bands(score, level):
    assert risk_level(score) == level


def test_combine_and_apply_risk(sample_df):
    assert combine_risk([1.0, 0.0], [1.0, 0.0]).tolist() == [1.0, 0.0]
    assert combine_risk([1.0], [0.0])[0] == pytest.approx(0.7)
    d = sample_df.assign(rf_probability=[1, 1, 0.9, 0.1, 0, 0, 0.5], iso_risk=[1, 0, 0.5, 0, 0, 1, 0.5])
    out = apply_risk(d)
    assert out["risk_score"].between(0, 1).all()
    assert (out["flagged"] == (out["risk_score"] >= FLAG_THRESHOLD)).all()
    assert out.loc[0, "risk_level"] == "Critical"


def test_classification_metrics_perfect_and_roc():
    m = classification_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert m["accuracy"] == m["f1"] == m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]
    assert len(m["roc_curve"]["fpr"]) == len(m["roc_curve"]["tpr"])
