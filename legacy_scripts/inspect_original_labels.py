
import pandas as pd

file_path = "cert_preprocessed_ml_comparison.csv"

# Read the original CSV without changing its structure
df = pd.read_csv(file_path)

print("\nORIGINAL CSV COLUMNS")
print(df.columns.tolist())

print("\nDATASET SHAPE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nLAST COLUMN VALUES")
last_col = df.columns[-1]
print("Column name:", repr(last_col))
print(df[last_col].value_counts(dropna=False).head(20))

print("\nSAMPLE ORIGINAL RECORDS")
print(df.head(10).to_string(index=False))

print("\nNORMALIZED LABEL COUNTS")
normalized = pd.read_csv("normalized_evidence_v3.csv")
print(normalized["label"].value_counts(dropna=False))

print("\nLABEL COUNTS BY RECORD FORMAT")
print(pd.crosstab(
    normalized["record_format"],
    normalized["label"],
    margins=True
))

print("\nLABELS BY ACTIVITY")
print(pd.crosstab(
    normalized["activity"],
    normalized["label"]
))

print("\nIMPORTANT:")
print("These checks show patterns in the data,")
print("but do not establish what the labels mean.")
