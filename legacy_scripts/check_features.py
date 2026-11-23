
import pandas as pd

df = pd.read_csv("normalized_evidence_v3.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(["user", "timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

df["previous_event_time"] = (
    df.groupby("user")["timestamp"].shift(1)
)

df["time_since_previous_event"] = (
    df["timestamp"] - df["previous_event_time"]
).dt.total_seconds()

print("FEATURE PREVIEW:")
print(
    df[[
        "user", "pc", "activity", "source_type",
        "hour", "day_of_week", "time_since_previous_event"
    ]].head(10).to_string(index=False)
)

print("\nMissing time intervals:",
      df["time_since_previous_event"].isna().sum())

print("\nUnique users:", df["user"].nunique())
print("Unique computers:", df["pc"].nunique())
