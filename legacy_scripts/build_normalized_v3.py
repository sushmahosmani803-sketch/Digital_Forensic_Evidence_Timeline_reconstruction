
import csv
import pandas as pd

INPUT_FILE = "cert_preprocessed_ml_comparison.csv"
OUTPUT_FILE = "normalized_evidence_v3.csv"

records = []

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        if len(row) == 9:
            record_number, source, event_id, timestamp, user, pc, activity, resource, case_label = row

        elif len(row) == 7:
            record_number, event_id, timestamp, pc, activity, user, case_label = row

            # Infer source type from the observed activity.
            activity_lower = activity.lower()

            if "http" in activity_lower:
                source = "http"
            elif "email" in activity_lower:
                source = "email"
            elif "file" in activity_lower:
                source = "file"
            elif activity_lower in ["logon", "logoff"]:
                source = "logon"
            elif activity_lower in ["connect", "disconnect"]:
                source = "device"
            else:
                source = "unknown"

            resource = ""

        else:
            print("Skipping unexpected row with", len(row), "fields")
            continue

        # Separate case ID and label without changing the original value.
        parts = case_label.split(";")
        case_id = parts[0] if len(parts) > 0 else ""
        label = parts[1] if len(parts) > 1 else ""

        records.append({
            "record_number": record_number,
            "source_type": source,
            "event_id": event_id,
            "timestamp": timestamp,
            "user": user,
            "pc": pc,
            "activity": activity,
            "resource": resource,
            "case_id": case_id,
            "label": label,
            "original_case_label": case_label,
            "record_format": "standard" if len(row) == 9 else "short"
        })

df = pd.DataFrame(records)

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce", format="mixed"
)
df["label"] = pd.to_numeric(df["label"], errors="coerce")

df.to_csv(OUTPUT_FILE, index=False)

print("\nNormalization completed.")
print("Output file:", OUTPUT_FILE)
print("Total records:", len(df))
print("\nRecord formats:")
print(df["record_format"].value_counts())
print("\nSource types:")
print(df["source_type"].value_counts())
print("\nMissing timestamps:", df["timestamp"].isna().sum())
print("Missing users:", df["user"].isna().sum())
print("\nLabels:")
print(df["label"].value_counts(dropna=False).sort_index())
