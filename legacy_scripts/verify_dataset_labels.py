
import pandas as pd

df = pd.read_csv("normalized_evidence_v3.csv")

print("DATASET LABEL VERIFICATION")
print("\nTotal records:", len(df))

print("\nLabel counts:")
print(df["label"].value_counts().sort_index())

print("\nLabels by record format:")
print(pd.crosstab(df["record_format"], df["label"]))

print("\nLabels by activity:")
print(pd.crosstab(df["activity"], df["label"]))

print("\nLabels by source type:")
print(pd.crosstab(df["source_type"], df["label"]))

print("\nSample records:")
print(df[["timestamp", "activity", "source_type",
          "record_format", "label"]].head(10).to_string(index=False))
