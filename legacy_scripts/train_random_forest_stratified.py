
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

train = pd.read_csv("evidence_train_stratified.csv")
test = pd.read_csv("evidence_test_stratified.csv")

# Prepare timestamps and time-based features
for df in [train, test]:
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

# Use only available event information
categorical = ["user", "pc", "activity", "source_type"]
numeric = ["hour", "day_of_week"]
features = categorical + numeric

y_train = train["label"].astype(int)
y_test = test["label"].astype(int)

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(train[features], y_train)
predictions = model.predict(test[features])

print("\nSTRATIFIED RANDOM FOREST EVALUATION")
print("Assumption: label 0 = normal, label 1 = anomalous")
print("Test events:", len(test))

print("\nConfusion matrix:")
print("Rows = actual [normal, anomalous]")
print("Columns = predicted [normal, anomalous]")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(
    y_test, predictions, zero_division=0
), 4))
print("Recall:", round(recall_score(
    y_test, predictions, zero_division=0
), 4))
print("F1-score:", round(f1_score(
    y_test, predictions, zero_division=0
), 4))

print("\nDetailed report:")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Normal", "Anomalous"],
    zero_division=0
))

results = test.copy()
results["predicted_label"] = predictions
results.to_csv("random_forest_stratified_results.csv", index=False)

print("\nSaved random_forest_stratified_results.csv")
