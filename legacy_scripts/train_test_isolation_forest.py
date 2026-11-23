
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

# Load the chronological split
train = pd.read_csv("evidence_train.csv")
test = pd.read_csv("evidence_test.csv")

train["timestamp"] = pd.to_datetime(train["timestamp"])
test["timestamp"] = pd.to_datetime(test["timestamp"])

# Create time-based features without using labels
train["_split"] = "train"
test["_split"] = "test"

all_data = pd.concat([train, test], ignore_index=True)
all_data = all_data.sort_values(["user", "timestamp"]).copy()

all_data["hour"] = all_data["timestamp"].dt.hour
all_data["day_of_week"] = all_data["timestamp"].dt.dayofweek
all_data["time_since_previous_event"] = (
    all_data.groupby("user")["timestamp"]
    .diff()
    .dt.total_seconds()
)

train = all_data[all_data["_split"] == "train"].copy()
test = all_data[all_data["_split"] == "test"].copy()

categorical = ["user", "pc", "activity", "source_type"]
numeric = [
    "hour",
    "day_of_week",
    "time_since_previous_event"
]
features = categorical + numeric

# Fit preprocessing using training data only
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", SimpleImputer(strategy="median"), numeric)
])

# Build the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("detector", IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    ))
])

# Train only on the training period
model.fit(train[features])

# Evaluate on the later, unseen period
test["anomaly_score"] = model.decision_function(test[features])
test["anomaly_prediction"] = model.predict(test[features])

test["anomaly_status"] = test["anomaly_prediction"].map({
    1: "Normal pattern",
    -1: "Potential anomaly"
})

# Save results
test = test.sort_values("timestamp")
test.to_csv("test_anomaly_results.csv", index=False)

flagged = (test["anomaly_prediction"] == -1).sum()

print("\nTRAIN-TEST ISOLATION FOREST RESULTS")
print("-" * 40)
print("Training events:", len(train))
print("Unseen test events:", len(test))
print("Potential anomalies in test:", flagged)
print("Test events flagged (%):", round(flagged / len(test) * 100, 2))

print("\nAnomalies by activity:")
print(
    test[test["anomaly_prediction"] == -1]["activity"]
    .value_counts()
)

print("\nSaved: test_anomaly_results.csv")
print("Accuracy has not been calculated because reliable")
print("ground-truth anomaly labels are not yet verified.")
