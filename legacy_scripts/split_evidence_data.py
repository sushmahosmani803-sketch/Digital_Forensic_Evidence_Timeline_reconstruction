
import pandas as pd
from pathlib import Path

# Load the normalized evidence dataset
file_path = Path("normalized_evidence_v3.csv")

if not file_path.exists():
    print("ERROR: normalized_evidence_v3.csv was not found.")
    print("Run this script from your mini_project folder.")
    raise SystemExit(1)

df = pd.read_csv(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Check the dataset
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Chronological 80:20 split
split_index = int(len(df) * 0.80)

train_data = df.iloc[:split_index].copy()
test_data = df.iloc[split_index:].copy()

# Save separate files
train_data.to_csv("evidence_train.csv", index=False)
test_data.to_csv("evidence_test.csv", index=False)

# Display results
print("\nCHRONOLOGICAL TRAIN-TEST SPLIT")
print("-" * 40)
print("Total events:", len(df))
print("Training events:", len(train_data))
print("Testing events:", len(test_data))

print("\nTRAINING PERIOD:")
print(train_data["timestamp"].min(), "to",
      train_data["timestamp"].max())

print("\nTESTING PERIOD:")
print(test_data["timestamp"].min(), "to",
      test_data["timestamp"].max())

print("\nTRAINING LABEL COUNTS:")
print(train_data["label"].value_counts().sort_index())

print("\nTESTING LABEL COUNTS:")
print(test_data["label"].value_counts().sort_index())

print("\nRECORD FORMAT AND LABEL CHECK:")
print(pd.crosstab(df["record_format"], df["label"]))

print("\nFiles created:")
print("evidence_train.csv")
print("evidence_test.csv")

print("\nIMPORTANT:")
print("The split is complete, but model evaluation is not.")
print("Verify label meanings before calculating accuracy.")
