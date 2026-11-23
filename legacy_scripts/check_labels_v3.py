
import pandas as pd

df = pd.read_csv("normalized_evidence_v3.csv")

print("LABELS BY CASE ID:")
print(pd.crosstab(df["case_id"], df["label"]).to_string())

print("\nLABELS BY ACTIVITY:")
print(pd.crosstab(df["activity"], df["label"]).to_string())

print("\nLABELS BY SOURCE TYPE:")
print(pd.crosstab(df["source_type"], df["label"]).to_string())

print("\nRECORD FORMAT COUNTS BY CASE:")
print(pd.crosstab(df["case_id"], df["record_format"]).to_string())
