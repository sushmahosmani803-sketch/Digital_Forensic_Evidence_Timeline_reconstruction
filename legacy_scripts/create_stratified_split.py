
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the normalized dataset
df = pd.read_csv("normalized_evidence_v3.csv")

# Remove rows without valid labels
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

# Create a stratified 80/20 split
train, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Save separate files for comparison
train.to_csv("evidence_train_stratified.csv", index=False)
test.to_csv("evidence_test_stratified.csv", index=False)

print("STRATIFIED DATA SPLIT")
print("Total records:", len(df))
print("Training records:", len(train))
print("Testing records:", len(test))

print("\nTraining label distribution:")
print(train["label"].value_counts().sort_index())

print("\nTesting label distribution:")
print(test["label"].value_counts().sort_index())

print("\nSaved stratified datasets successfully.")
