import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "dataset/drug_repurposing.xlsx"

df = pd.read_excel(DATA_PATH)

print("=" * 60)
print("DRUG REPURPOSING AI - PREPROCESSING")
print("=" * 60)

print(f"\nOriginal dataset shape: {df.shape}")


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

target_column = "Repurposed_Category"

feature_columns = [
    "Molecular_Weight",
    "LogP",
    "Hydrogen_Bond_Donors",
    "Hydrogen_Bond_Acceptors",
    "TPSA",
    "Rotatable_Bonds",
    "Drug_Class",
    "Target_Protein_UniProt",
    "Current_Use"
]

X = df[feature_columns].copy()
y = df[target_column].copy()


# ============================================================
# 3. GROUP VERY RARE TARGET CLASSES
# ============================================================

class_counts = y.value_counts()

rare_classes = class_counts[class_counts < 5].index

print("\nRare classes being grouped:")
for category in rare_classes:
    print(f"  {category}: {class_counts[category]} records")

y = y.replace(rare_classes, "Rare")

print("\nTarget distribution after grouping:")
print(y.value_counts())


# ============================================================
# 4. DEFINE FEATURE TYPES
# ============================================================

numeric_features = [
    "Molecular_Weight",
    "LogP",
    "Hydrogen_Bond_Donors",
    "Hydrogen_Bond_Acceptors",
    "TPSA",
    "Rotatable_Bonds"
]

categorical_features = [
    "Drug_Class",
    "Target_Protein_UniProt",
    "Current_Use"
]


# ============================================================
# 5. NUMERICAL PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# ============================================================
# 6. CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


# ============================================================
# 7. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 8. ENCODE TARGET
# ============================================================

target_encoder = LabelEncoder()

y_encoded = target_encoder.fit_transform(y)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ============================================================
# 10. FIT PREPROCESSOR ONLY ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING RESULTS")
print("=" * 60)

print(f"\nTraining records: {X_train.shape[0]}")
print(f"Testing records:  {X_test.shape[0]}")

print(f"\nProcessed training shape: {X_train_processed.shape}")
print(f"Processed testing shape:  {X_test_processed.shape}")

print("\nTarget classes:")

for index, class_name in enumerate(target_encoder.classes_):
    print(f"{index}: {class_name}")

print("\nPreprocessing completed successfully.")