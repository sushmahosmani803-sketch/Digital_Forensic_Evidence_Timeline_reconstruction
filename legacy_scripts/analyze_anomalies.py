
import pandas as pd

df = pd.read_csv("isolation_forest_results.csv")

anomalies = df[df["anomaly_prediction"] == -1].copy()

print("TOTAL POTENTIAL ANOMALIES:", len(anomalies))

print("\nTOP USERS WITH FLAGGED EVENTS:")
print(anomalies["user"].value_counts().head(10))

print("\nFLAGGED ACTIVITIES:")
print(anomalies["activity"].value_counts())

print("\nFLAGGED EVENTS BY SOURCE:")
print(anomalies["source_type"].value_counts())

print("\nTOP COMPUTERS:")
print(anomalies["pc"].value_counts().head(10))

print("\nLOWEST ANOMALY SCORES:")
print(
    anomalies.sort_values("anomaly_score")[
        ["timestamp", "user", "pc", "activity",
         "source_type", "anomaly_score"]
    ].head(15).to_string(index=False)
)
