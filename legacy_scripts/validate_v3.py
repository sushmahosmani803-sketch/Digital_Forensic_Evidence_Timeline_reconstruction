
import pandas as pd

df = pd.read_csv("normalized_evidence_v3.csv")

print("TOTAL ROWS:", len(df))
print("DUPLICATE EVENT IDs:", df["event_id"].duplicated().sum())
print("MISSING VALUES:")
print(df.isna().sum())

print("\nLABEL DISTRIBUTION:")
print(df["label"].value_counts().sort_index())

print("\nRECORD FORMAT VS LABEL:")
print(pd.crosstab(df["record_format"], df["label"]))

print("\nSAMPLE SHORT RECORDS:")
print(
    df[df["record_format"] == "short"][
        ["event_id", "timestamp", "user", "pc",
         "activity", "source_type", "case_id", "label"]
    ].head(10).to_string(index=False)
)

print("\nTIMELINE RANGE:")
print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())
