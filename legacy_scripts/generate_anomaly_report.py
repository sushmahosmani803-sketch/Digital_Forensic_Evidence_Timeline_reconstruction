
import pandas as pd

df = pd.read_csv("isolation_forest_results.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

anomalies = df[df["anomaly_prediction"] == -1].copy()

anomalies = anomalies.sort_values(
    ["user", "timestamp"]
)

columns = [
    "timestamp",
    "user",
    "pc",
    "activity",
    "source_type",
    "case_id",
    "anomaly_score",
    "anomaly_status"
]

anomalies[columns].to_csv(
    "anomaly_investigation_report.csv",
    index=False
)

print("Anomaly investigation report created.")
print("File: anomaly_investigation_report.csv")
print("Events included:", len(anomalies))

print("\nFirst 10 events in chronological order:")
print(
    anomalies[columns]
    .sort_values("timestamp")
    .head(10)
    .to_string(index=False)
)

