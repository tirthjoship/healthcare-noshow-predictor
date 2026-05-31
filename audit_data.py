import os
import urllib.request
import zipfile

import pandas as pd

# URLs
url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
zip_path = "diabetes.zip"
extract_dir = "diabetes_data"

# Download and extract if not exists
if not os.path.exists(extract_dir):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

csv_file = os.path.join(extract_dir, "diabetic_data.csv")
print("Loading data...")
df = pd.read_csv(csv_file, na_values="?")

print("Dataset shape:", df.shape)

# Readmission rates
# The target variable is 'readmitted': '<30', '>30', 'NO'
# We typically want <30 as 1 and others as 0 for readmission risk
readmitted_30 = (df["readmitted"] == "<30").mean()
print(
    f"Overall <30 day readmission rate: {readmitted_30:.4f} ({readmitted_30 * 100:.2f}%)"
)

# Specific medication risk
# Let's check some common ones like insulin, metformin
medicines = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

print("\n--- Medication readmission (<30 days) rates ---")
df["is_readmitted"] = df["readmitted"] == "<30"

for med in [
    "insulin",
    "metformin",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
]:
    if med in df.columns:
        rates = df.groupby(med)["is_readmitted"].agg(["mean", "count"])
        print(f"\nMedication: {med}")
        print(rates)

print("\n--- Age group readmission (<30 days) rates ---")
if "age" in df.columns:
    age_rates = df.groupby("age")["is_readmitted"].agg(["mean", "count"])
    print(age_rates)
