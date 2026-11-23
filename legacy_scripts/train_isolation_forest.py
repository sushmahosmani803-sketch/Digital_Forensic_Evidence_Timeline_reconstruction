
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv("normalized_evidence_v3.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["user", "timestamp"]).copy()

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

df["time_since_previous_event"] = (
    df.groupby("user")["timestamp"]
    .diff()
    .dt.total_seconds()
)

features = [
    "user",
    "pc",
    "activity",
    "source_type",
    "hour",
    "day_of_week",
    "time_since_previous_event"
]

X = df[features]

categorical = ["user", "pc", "activity", "source_type"]
numeric = ["hour", "day_of_week", "time_since_previous_event"]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical),
    ("numeric", SimpleImputer(strategy="median"), numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("detector", IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    ))
])

model.fit(X)

df["anomaly_score"] = model.decision_function(X)
df["anomaly_prediction"] = model.predict(X)

df["anomaly_status"] = df["anomaly_prediction"].map({
    1: "Normal pattern",
    -1: "Potential anomaly"
})

output = "isolation_forest_results.csv"
df.to_csv(output, index=False)

print("Isolation Forest training completed.")
print("Total events:", len(df))
print("\nPredicted event patterns:")
print(df["anomaly_status"].value_counts())

print("\nPotential anomaly examples:")
print(
    df[df["anomaly_prediction"] == -1][
        ["timestamp", "user", "pc", "activity",
         "anomaly_score", "anomaly_status"]
    ].head(10).to_string(index=False)
)

print("\nSaved results to:", output)
