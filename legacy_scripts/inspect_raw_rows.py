
import csv
from collections import Counter

filename = "cert_preprocessed_ml_comparison.csv"
last_values = Counter()

with open(filename, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file)
    next(reader, None)

    for row in reader:
        if row:
            last_values[row[-1]] += 1

print("Most common final-field values:")
for value, count in last_values.most_common(15):
    print(f"{value}: {count}")
