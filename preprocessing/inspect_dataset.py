import pandas as pd

# Load Excel dataset
file_path = "dataset/drug_repurposing.xlsx"

df = pd.read_excel(file_path)

print("=" * 60)
print("DRUG REPURPOSING AI - DATASET INSPECTION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["Repurposed_Category"].value_counts())

# Columns excluded from ML
identifier_columns = [
    "Record_#",
    "Drug_Name",
    "DrugBank_ID",
    "Repurposed_Use"
]

print("\nColumns excluded from ML:")

for column in identifier_columns:
    print("-", column)

# Create ML dataframe without excluded columns
ml_df = df.drop(columns=identifier_columns)

print("\nML Dataset Shape:")
print(ml_df.shape)

print("\nML Dataset Columns:")

for column in ml_df.columns:
    print("-", column)