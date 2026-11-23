"""End-to-end evidence pipeline.

    python pipeline.py                       # uses data/cert_preprocessed_ml_comparison.csv
    python pipeline.py --raw my.csv --out outputs

Steps: normalise -> engineer features -> evaluate models -> score every
event -> write the investigation report, user risk profiles and metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evidence import config
from evidence.features import BASELINE_NUMERIC, CATEGORICAL_FEATURES, build_features
from evidence.models import (
    apply_risk,
    chronological_split,
    classification_metrics,
    evaluate_classifier,
    evaluate_isolation,
    feature_columns,
    feature_importance,
    iso_scores_to_risk,
    out_of_fold_probability,
    score_isolation,
    stratified_split,
    train_isolation_forest,
    train_random_forest,
)
from evidence.normalize import normalize_raw_csv
from evidence.report import build_investigation_report
from evidence.timeline import user_risk_profiles


def evaluate_models(features: pd.DataFrame, n_estimators: int = 300) -> dict:
    train, test = stratified_split(features)
    baseline = train_random_forest(train, CATEGORICAL_FEATURES, BASELINE_NUMERIC, n_estimators=200)
    enhanced = train_random_forest(train, n_estimators=n_estimators)
    iso = train_isolation_forest(train)

    c_train, c_test = chronological_split(features)
    chrono = train_random_forest(c_train, n_estimators=n_estimators)

    return {
        "baseline_rf": {
            "name": "Baseline Random Forest",
            "description": "Original project features (user, pc, activity, source, hour, weekday)",
            "split": "stratified 80/20",
            **evaluate_classifier(baseline, test),
        },
        "enhanced_rf": {
            "name": "Enhanced Random Forest",
            "description": "Adds behavioural features: inter-event gaps, daily volume, off-hours, foreign PC",
            "split": "stratified 80/20",
            **evaluate_classifier(enhanced, test),
            "feature_importance": feature_importance(enhanced),
        },
        "enhanced_rf_chronological": {
            "name": "Enhanced RF (future period)",
            "description": "Trained on the earliest 80% of the timeline, tested on the latest 20%",
            "split": "chronological 80/20",
            **evaluate_classifier(chrono, c_test),
        },
        "isolation_forest": {
            "name": "Isolation Forest",
            "description": "Unsupervised: flags unusual behaviour without using labels",
            "split": "stratified 80/20",
            **evaluate_isolation(iso, test),
        },
    }


def score_all_events(features: pd.DataFrame, n_estimators: int = 300) -> pd.DataFrame:
    scored = features.copy()
    scored["rf_probability"] = out_of_fold_probability(features, n_estimators=n_estimators)
    scored["rf_probability"] = scored["rf_probability"].fillna(0.0)
    iso = train_isolation_forest(features)
    scored = score_isolation(iso, scored)
    scored = apply_risk(scored)
    return scored.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def score_new_evidence(new_raw: pd.DataFrame, reference_features: pd.DataFrame, n_estimators: int = 200) -> pd.DataFrame:
    """Score freshly uploaded (normalised) evidence with models trained on the reference data."""
    new_feat = build_features(new_raw)
    ref = reference_features[pd.to_numeric(reference_features["label"], errors="coerce").notna()].copy()
    ref["label"] = ref["label"].astype(int)
    rf = train_random_forest(ref, n_estimators=n_estimators)
    iso = train_isolation_forest(ref)
    out = new_feat.copy()
    out["rf_probability"] = rf.predict_proba(out[feature_columns(rf)])[:, 1]
    ref_scores = iso.decision_function(ref[feature_columns(iso)])
    out = score_isolation(iso, out)
    out["iso_risk"] = iso_scores_to_risk(out["anomaly_score"], reference=ref_scores)
    out = apply_risk(out)
    return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def run_pipeline(raw_path=config.RAW_FILE, out_dir=config.OUTPUT_DIR, n_estimators: int = 300, verbose: bool = True) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}")

    normalized = normalize_raw_csv(raw_path)
    log(f"Normalised {len(normalized):,} events ({normalized.attrs.get('skipped_rows', 0)} skipped)")
    normalized.drop(columns=["display_activity"]).to_csv(out_dir / config.NORMALIZED_FILE, index=False)

    features = build_features(normalized)
    log("Engineered time + behavioural features")

    metrics = evaluate_models(features, n_estimators=n_estimators)
    for key, m in metrics.items():
        log(f"{m['name']:<30} F1={m['f1']:.3f}  acc={m['accuracy']:.3f}  AUC={m.get('roc_auc', float('nan')):.3f}")

    scored = score_all_events(features, n_estimators=n_estimators)
    lab = scored[pd.to_numeric(scored["label"], errors="coerce").notna()]
    metrics["hybrid_risk"] = {
        "name": "Hybrid Risk Score",
        "description": "0.7 x out-of-fold RF probability + 0.3 x Isolation Forest risk, flag at >= 0.60",
        "split": "5-fold out-of-fold (every event)",
        **classification_metrics(lab["label"].astype(int), lab["flagged"].astype(int), lab["risk_score"]),
    }
    log(f"Scored all events; {int(scored['flagged'].sum())} flagged")

    keep = [c for c in scored.columns if c not in {"log_gap_prev"}]
    scored[keep].to_csv(out_dir / config.SCORED_FILE, index=False)

    report = build_investigation_report(scored)
    report.to_csv(out_dir / config.REPORT_FILE, index=False)

    profiles = user_risk_profiles(scored)
    profiles.to_csv(out_dir / config.USER_RISK_FILE, index=False)

    summary = {
        "events": int(len(scored)),
        "users": int(scored["user"].nunique()),
        "pcs": int(scored["pc"].nunique()),
        "cases": int(scored["case_id"].nunique()),
        "start": scored["timestamp"].min(),
        "end": scored["timestamp"].max(),
        "label_counts": {str(k): int(v) for k, v in lab["label"].value_counts().sort_index().items()},
        "flagged": int(scored["flagged"].sum()),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    payload = _jsonable({"summary": summary, "models": metrics})
    (out_dir / config.METRICS_FILE).write_text(json.dumps(payload, indent=2))
    log(f"Wrote outputs to {out_dir}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=str(config.RAW_FILE), help="raw evidence CSV")
    parser.add_argument("--out", default=str(config.OUTPUT_DIR), help="output folder")
    parser.add_argument("--trees", type=int, default=300, help="trees per Random Forest")
    args = parser.parse_args()
    run_pipeline(args.raw, args.out, n_estimators=args.trees)


if __name__ == "__main__":
    main()
