"""Model training, scoring and evaluation.

* Random Forest  - supervised classifier (label 1 = malicious / insider event)
* Isolation Forest - unsupervised detector of unusual behaviour
* Hybrid risk score - 0.7 * RF probability + 0.3 * Isolation risk
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import FLAG_THRESHOLD, RANDOM_STATE, RISK_BANDS
from .features import CATEGORICAL_FEATURES, ISOLATION_NUMERIC, NUMERIC_FEATURES

RF_WEIGHT = 0.7
ISO_WEIGHT = 0.3


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def labelled(df: pd.DataFrame) -> pd.DataFrame:
    out = df[pd.to_numeric(df["label"], errors="coerce").notna()].copy()
    out["label"] = out["label"].astype(int)
    return out


def stratified_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    data = labelled(df)
    return train_test_split(data, test_size=test_size, random_state=random_state, stratify=data["label"])


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.8):
    data = labelled(df).dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    cut = int(len(data) * train_fraction)
    return data.iloc[:cut].copy(), data.iloc[cut:].copy()


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ]
    )


def make_random_forest(categorical=None, numeric=None, n_estimators: int = 300) -> Pipeline:
    categorical = list(categorical or CATEGORICAL_FEATURES)
    numeric = list(numeric or NUMERIC_FEATURES)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(categorical, numeric)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_isolation_forest(contamination: float = 0.05, numeric=None) -> Pipeline:
    numeric = list(numeric or ISOLATION_NUMERIC)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(CATEGORICAL_FEATURES, numeric)),
            (
                "detector",
                IsolationForest(n_estimators=200, contamination=contamination, random_state=RANDOM_STATE),
            ),
        ]
    )


def feature_columns(model: Pipeline) -> list[str]:
    pre = model.named_steps["preprocessor"]
    cols = []
    for _, _, c in pre.transformers:
        if isinstance(c, list):
            cols.extend(c)
    return cols


def train_random_forest(train: pd.DataFrame, categorical=None, numeric=None, n_estimators: int = 300) -> Pipeline:
    model = make_random_forest(categorical, numeric, n_estimators)
    cols = feature_columns(model)
    model.fit(train[cols], train["label"].astype(int))
    return model


def train_isolation_forest(train: pd.DataFrame, contamination: float = 0.05) -> Pipeline:
    model = make_isolation_forest(contamination)
    model.fit(train[feature_columns(model)])
    return model


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def iso_scores_to_risk(scores, reference=None) -> np.ndarray:
    """Map Isolation Forest decision scores (lower = more anomalous) to 0..1 risk.

    ``reference`` (optional) supplies the score range to scale against, so new
    evidence is scored on the same scale as the training data.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return s
    ref = s if reference is None else np.asarray(reference, dtype=float)
    lo, hi = np.nanmin(ref), np.nanmax(ref)
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return np.clip((hi - s) / (hi - lo), 0.0, 1.0)


def risk_level(score: float) -> str:
    for threshold, name in RISK_BANDS:
        if score >= threshold:
            return name
    return RISK_BANDS[-1][1]


def score_isolation(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    X = out[feature_columns(model)]
    out["anomaly_score"] = model.decision_function(X)
    out["anomaly_prediction"] = model.predict(X)
    out["anomaly_status"] = out["anomaly_prediction"].map({1: "Normal pattern", -1: "Potential anomaly"})
    out["iso_risk"] = iso_scores_to_risk(out["anomaly_score"])
    return out


def combine_risk(rf_probability, iso_risk) -> np.ndarray:
    rf = np.asarray(rf_probability, dtype=float)
    iso = np.asarray(iso_risk, dtype=float)
    return np.clip(RF_WEIGHT * rf + ISO_WEIGHT * iso, 0.0, 1.0)


def apply_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_score"] = combine_risk(out["rf_probability"], out["iso_risk"])
    out["risk_level"] = out["risk_score"].map(risk_level)
    out["flagged"] = out["risk_score"] >= FLAG_THRESHOLD
    return out


def out_of_fold_probability(df: pd.DataFrame, folds: int = 5, n_estimators: int = 300) -> np.ndarray:
    """Honest per-event malicious probability: each event is scored by a model
    that never saw it during training (k-fold cross-validation)."""
    data = labelled(df)
    model = make_random_forest(n_estimators=n_estimators)
    cols = feature_columns(model)
    min_class = data["label"].value_counts().min()
    k = int(max(2, min(folds, min_class)))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(model, data[cols], data["label"], cv=cv, method="predict_proba")[:, 1]
    result = pd.Series(np.nan, index=df.index, dtype=float)
    result.loc[data.index] = proba
    return result.to_numpy()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "support": int(len(y_true)),
    }
    if y_score is not None and len(set(y_true)) == 2:
        y_score = np.asarray(y_score, dtype=float)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        idx = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 60)).astype(int))
        metrics["roc_curve"] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}
    return metrics


def evaluate_classifier(model: Pipeline, test: pd.DataFrame) -> dict:
    cols = feature_columns(model)
    pred = model.predict(test[cols])
    proba = model.predict_proba(test[cols])[:, 1]
    return classification_metrics(test["label"].astype(int), pred, proba)


def evaluate_isolation(model: Pipeline, test: pd.DataFrame) -> dict:
    scored = score_isolation(model, test)
    y_pred = (scored["anomaly_prediction"] == -1).astype(int)
    return classification_metrics(test["label"].astype(int), y_pred, scored["iso_risk"])


def feature_importance(model: Pipeline) -> dict:
    """RF importances aggregated back to the original (pre one-hot) features."""
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    names = pre.get_feature_names_out()
    agg: dict[str, float] = {}
    originals = feature_columns(model)
    for name, imp in zip(names, clf.feature_importances_):
        base = name.split("__", 1)[-1]
        match = next((o for o in sorted(originals, key=len, reverse=True) if base == o or base.startswith(o + "_")), base)
        agg[match] = agg.get(match, 0.0) + float(imp)
    return dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))
