
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

df = pd.read_csv("test_anomaly_results.csv")

# Candidate ground-truth labels:
# 1 = anomalous, 0 = normal.
# Verify this mapping against the dataset documentation.
y_true = pd.to_numeric(df["label"], errors="coerce")
y_pred = (df["anomaly_prediction"] == -1).astype(int)

if y_true.isna().any():
    raise ValueError("Some labels are missing or invalid.")

if not set(y_true.unique()).issubset({0, 1}):
    raise ValueError("Labels must be encoded as 0 and 1.")

print("\nPROVISIONAL TEST EVALUATION")
print("Assumption: label 1 = anomalous, label 0 = normal")
print("Test events:", len(df))
print("Actual label counts:")
print(y_true.value_counts().sort_index())

print("\nConfusion matrix:")
print("Rows = actual [normal, anomalous]")
print("Columns = predicted [normal, anomalous]")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nAccuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Precision:", round(precision_score(y_true, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_true, y_pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y_true, y_pred, zero_division=0), 4))

print("\nDetailed report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Anomalous"],
    zero_division=0
))

print("\nWARNING:")
print("These metrics are valid only if the label mapping is correct")
print("and the test data and feature processing are appropriate.")
