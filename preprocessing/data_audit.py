import pandas as pd
import numpy as np

# ============================================================
# DRUG REPURPOSING AI - DATA QUALITY AUDIT
# ============================================================

DATA_PATH = "dataset/drug_repurposing.xlsx"

df = pd.read_excel(DATA_PATH)

print("=" * 70)
print("DRUG REPURPOSING AI - DATA QUALITY & LEAKAGE AUDIT")
print("=" * 70)


# ============================================================
# 1. BASIC INFORMATION
# ============================================================

print("\n1. DATASET INFORMATION")
print("-" * 70)

print("Rows    :", len(df))
print("Columns :", len(df.columns))

print("\nData types:")
print(df.dtypes)


# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n2. MISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()

missing_percentage = (
    missing / len(df) * 100
)

missing_table = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": missing_percentage.round(2)
})

print(
    missing_table[
        missing_table["Missing_Count"] > 0
    ]
)


# ============================================================
# 3. DUPLICATE ROWS
# ============================================================

print("\n3. DUPLICATE ROWS")
print("-" * 70)

duplicate_rows = df.duplicated().sum()

print(
    "Complete duplicate rows:",
    duplicate_rows
)


# ============================================================
# 4. DUPLICATE RECORD NUMBERS
# ============================================================

print("\n4. DUPLICATE RECORD NUMBERS")
print("-" * 70)

print(
    "Unique Record_#:",
    df["Record_#"].nunique()
)

print(
    "Duplicate Record_#:",
    df["Record_#"].duplicated().sum()
)


# ============================================================
# 5. DUPLICATE DRUG NAMES
# ============================================================

print("\n5. DRUG NAME UNIQUENESS")
print("-" * 70)

print(
    "Unique Drug_Name:",
    df["Drug_Name"].nunique()
)

print(
    "Duplicate Drug_Name records:",
    df["Drug_Name"].duplicated().sum()
)


# ============================================================
# 6. DRUGBANK ID UNIQUENESS
# ============================================================

print("\n6. DRUGBANK ID UNIQUENESS")
print("-" * 70)

print(
    "Unique DrugBank_ID:",
    df["DrugBank_ID"].nunique()
)

print(
    "Duplicate DrugBank_ID records:",
    df["DrugBank_ID"].duplicated().sum()
)


# ============================================================
# 7. DUPLICATE DRUGS WITH DIFFERENT TARGETS
# ============================================================

print("\n7. DRUGS WITH MULTIPLE TARGET CATEGORIES")
print("-" * 70)

drug_target_counts = (
    df.groupby("Drug_Name")["Repurposed_Category"]
    .nunique()
)

multiple_targets = (
    drug_target_counts[
        drug_target_counts > 1
    ]
)

print(
    "Drugs appearing in multiple categories:",
    len(multiple_targets)
)

if len(multiple_targets) > 0:

    print("\nExamples:")

    print(
        multiple_targets.head(20)
    )


# ============================================================
# 8. TARGET DISTRIBUTION
# ============================================================

print("\n8. TARGET DISTRIBUTION")
print("-" * 70)

target_distribution = (
    df["Repurposed_Category"]
    .value_counts()
)

target_percentage = (
    df["Repurposed_Category"]
    .value_counts(normalize=True) * 100
)

target_table = pd.DataFrame({
    "Count": target_distribution,
    "Percentage": target_percentage.round(2)
})

print(target_table)


# ============================================================
# 9. UNIQUE CATEGORICAL VALUES
# ============================================================

print("\n9. CATEGORICAL FEATURE CARDINALITY")
print("-" * 70)

categorical_columns = [
    "Drug_Class",
    "Target_Protein_UniProt",
    "Current_Use",
    "Repurposed_Use",
    "Repurposed_Category"
]

for column in categorical_columns:

    print(
        f"{column}: "
        f"{df[column].nunique(dropna=True)} "
        f"unique values"
    )


# ============================================================
# 10. CURRENT USE VS TARGET
# ============================================================

print("\n10. CURRENT USE → REPURPOSED CATEGORY")
print("-" * 70)

current_use_target = pd.crosstab(
    df["Current_Use"],
    df["Repurposed_Category"]
)

print(
    "Current_Use categories:",
    len(current_use_target)
)

print(
    "Current_Use × Target table shape:",
    current_use_target.shape
)


# ============================================================
# 11. REPURPOSED USE VS TARGET
# ============================================================

print("\n11. REPURPOSED USE → TARGET LEAKAGE CHECK")
print("-" * 70)

repurposed_use_target = pd.crosstab(
    df["Repurposed_Use"],
    df["Repurposed_Category"]
)

print(
    "Repurposed_Use unique values:",
    df["Repurposed_Use"].nunique(
        dropna=True
    )
)

print(
    "Repurposed_Use × Target table shape:",
    repurposed_use_target.shape
)

# Check whether Repurposed_Use is strongly associated
# with exactly one target category

if len(repurposed_use_target) > 0:

    dominant_counts = (
        repurposed_use_target.max(axis=1)
    )

    total_counts = (
        repurposed_use_target.sum(axis=1)
    )

    dominance = (
        dominant_counts / total_counts
    )

    print(
        "\nAverage target dominance of "
        "Repurposed_Use values:",
        round(dominance.mean() * 100, 2),
        "%"
    )

    print(
        "Maximum target dominance:",
        round(dominance.max() * 100, 2),
        "%"
    )


# ============================================================
# 12. TARGET PROTEIN CARDINALITY
# ============================================================

print("\n12. TARGET PROTEIN ANALYSIS")
print("-" * 70)

protein_counts = (
    df["Target_Protein_UniProt"]
    .value_counts(dropna=True)
)

print(
    "Unique proteins:",
    len(protein_counts)
)

print(
    "\nMost common proteins:"
)

print(
    protein_counts.head(10)
)


# ============================================================
# 13. DRUG CLASS DISTRIBUTION
# ============================================================

print("\n13. DRUG CLASS ANALYSIS")
print("-" * 70)

drug_class_counts = (
    df["Drug_Class"]
    .value_counts(dropna=True)
)

print(
    "Unique drug classes:",
    len(drug_class_counts)
)

print(
    "\nMost common drug classes:"
)

print(
    drug_class_counts.head(10)
)


# ============================================================
# 14. NUMERICAL FEATURES
# ============================================================

print("\n14. NUMERICAL FEATURE STATISTICS")
print("-" * 70)

numeric_columns = [
    "Molecular_Weight",
    "LogP",
    "Hydrogen_Bond_Donors",
    "Hydrogen_Bond_Acceptors",
    "TPSA",
    "Rotatable_Bonds"
]

print(
    df[numeric_columns].describe().round(3)
)


# ============================================================
# 15. EXTREME VALUES
# ============================================================

print("\n15. EXTREME VALUE CHECK")
print("-" * 70)

for column in numeric_columns:

    print(
        f"\n{column}"
    )

    print(
        "Minimum:",
        df[column].min()
    )

    print(
        "Maximum:",
        df[column].max()
    )

    print(
        "Unique:",
        df[column].nunique()
    )


# ============================================================
# 16. CORRELATION BETWEEN NUMERICAL FEATURES
# ============================================================

print("\n16. NUMERICAL CORRELATION")
print("-" * 70)

correlation = (
    df[numeric_columns]
    .corr()
    .round(3)
)

print(correlation)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print(
    "\nTotal records:",
    len(df)
)

print(
    "Duplicate rows:",
    duplicate_rows
)

print(
    "Unique drugs:",
    df["Drug_Name"].nunique()
)

print(
    "Unique DrugBank IDs:",
    df["DrugBank_ID"].nunique()
)

print(
    "Unique Drug Classes:",
    df["Drug_Class"].nunique(
        dropna=True
    )
)

print(
    "Unique Target Proteins:",
    df["Target_Protein_UniProt"].nunique(
        dropna=True
    )
)

print(
    "Unique Current Uses:",
    df["Current_Use"].nunique(
        dropna=True
    )
)

print(
    "Unique Repurposed Uses:",
    df["Repurposed_Use"].nunique(
        dropna=True
    )
)

print(
    "Target Classes:",
    df["Repurposed_Category"].nunique()
)

print("\nAudit completed successfully.")