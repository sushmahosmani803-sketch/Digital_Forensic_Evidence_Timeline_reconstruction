
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

# 1. Load datasets
train = pd.read_csv("evidence_train.csv")
test = pd.read_csv("evidence_test.csv")

# 2. Reset indexes
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# 3. Prepare timestamps
train["timestamp"] = pd.to_datetime(train["timestamp"], errors="coerce")
test["timestamp"] = pd.to_datetime(test["timestamp"], errors="coerce")

# Calculate time since the previous event for each user.
# Use only earlier events to avoid future-data leakage.
train["_split"] = "train"
train["_row_id"] = range(len(train))
test["_split"] = "test"
test["_row_id"] = range(len(test))

combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values("timestamp", na_position="last")

combined["time_since_previous_event"] = (
    combined.groupby("user")["timestamp"]
    .diff()
    .dt.total_seconds()
)

train = combined[combined["_split"] == "train"].sort_values("_row_id").copy()
test = combined[combined["_split"] == "test"].sort_values("_row_id").copy()

for df in [train, test]:
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

# 4. Define features and labels
categorical = ["user", "pc", "activity", "source_type"]
numeric = ["hour", "day_of_week", "time_since_previous_event"]
features = categorical + numeric

y_train = pd.to_numeric(train["label"], errors="raise").astype(int)
y_test = pd.to_numeric(test["label"], errors="raise").astype(int)

if not set(y_train.unique()).issubset({0, 1}):
    raise ValueError("Labels must be 0 or 1.")

# 5. Preprocess data
preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric)
])

# 6. Build and train Random Forest
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

# 7. Predict test data
predictions = model.predict(test[features])

print("\nRANDOM FOREST TEST EVALUATION")
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

# 8. Save results
results = test.drop(columns=["_split", "_row_id"]).copy()
results["actual_label"] = y_test.to_numpy()
results["predicted_label"] = predictions
results["prediction_result"] = results["predicted_label"].map({
    0: "Normal",
    1: "Potential anomaly"
})
results.to_csv("random_forest_results.csv", index=False)

print("\nResults saved to random_forest_results.csv")
