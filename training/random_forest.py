import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# DRUG REPURPOSING AI - RANDOM FOREST
# ============================================================

print("=" * 70)
print("DRUG REPURPOSING AI - RANDOM FOREST")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "dataset/drug_repurposing.xlsx"

df = pd.read_excel(DATA_PATH)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# 2. TARGET VARIABLE
# ============================================================

y = df["Repurposed_Category"].copy()


# ============================================================
# 3. GROUP RARE CLASSES
# ============================================================

class_counts = y.value_counts()

rare_classes = class_counts[class_counts < 5].index

print("\nRare classes being grouped:")

for cls in rare_classes:
    print(
        f"{cls}: {class_counts[cls]} records"
    )


y = y.replace(
    rare_classes,
    "Rare"
)


print("\nTarget distribution after grouping:")
print(y.value_counts())


# ============================================================
# 4. INPUT FEATURES
# ============================================================

features = [
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

X = df[features].copy()


# IMPORTANT:
# Repurposed_Category = TARGET
# Repurposed_Use      = NOT USED because it creates leakage


# ============================================================
# 5. TARGET ENCODING
# ============================================================

target_encoder = LabelEncoder()

y_encoded = target_encoder.fit_transform(y)


print("\nTarget classes:")

for i, class_name in enumerate(
    target_encoder.classes_
):
    print(
        f"{i}: {class_name}"
    )


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. FEATURE TYPES
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
# 8. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ============================================================
# 9. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
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
# 10. COMBINE PREPROCESSORS
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 11. PREPROCESS DATA
# ============================================================

print("\nPreprocessing...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# 12. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

model.fit(
    X_train_processed,
    y_train
)

print(
    "Random Forest training completed."
)


# ============================================================
# 14. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_processed
)


# ============================================================
# 15. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 16. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy        : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision       : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall          : "
    f"{recall * 100:.2f}%"
)

print(
    f"Weighted F1     : "
    f"{weighted_f1 * 100:.2f}%"
)

print(
    f"Macro F1        : "
    f"{macro_f1 * 100:.2f}%"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 18. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    "evaluation",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 19. SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "Algorithm": "Random Forest",
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "Weighted_F1": weighted_f1 * 100,
        "Macro_F1": macro_f1 * 100
    }
])


results.to_csv(
    "evaluation/random_forest_results.csv",
    index=False
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


cm_df = pd.DataFrame(
    cm,
    index=target_encoder.classes_,
    columns=target_encoder.classes_
)


cm_df.to_csv(
    "evaluation/random_forest_confusion_matrix.csv"
)


# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

feature_names = (
    preprocessor
    .get_feature_names_out()
)


feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nTop 20 Important Features:")

print(
    feature_importance.head(20)
)


feature_importance.to_csv(
    "evaluation/random_forest_feature_importance.csv",
    index=False
)


# ============================================================
# 22. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/random_forest_model.pkl"
)


# ============================================================
# 23. SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    "models/random_forest_preprocessor.pkl"
)


# ============================================================
# 24. SAVE TARGET ENCODER
# ============================================================

joblib.dump(
    target_encoder,
    "models/random_forest_target_encoder.pkl"
)


# ============================================================
# 25. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles saved:")

print(
    "evaluation/random_forest_results.csv"
)

print(
    "evaluation/random_forest_confusion_matrix.csv"
)

print(
    "evaluation/random_forest_feature_importance.csv"
)

print(
    "models/random_forest_model.pkl"
)

print(
    "models/random_forest_preprocessor.pkl"
)

print(
    "models/random_forest_target_encoder.pkl"
)

print("\nReady for model comparison.")