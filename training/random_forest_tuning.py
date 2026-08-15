import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
# DRUG REPURPOSING AI - RANDOM FOREST TUNING
# ============================================================

print("=" * 70)
print("DRUG REPURPOSING AI - RANDOM FOREST TUNING")
print("=" * 70)

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "dataset/drug_repurposing.xlsx"

df = pd.read_excel(DATA_PATH)

print("\nDataset shape:", df.shape)

# ============================================================
# 2. TARGET
# ============================================================

y = df["Repurposed_Category"].copy()

# ============================================================
# 3. GROUP RARE CLASSES
# ============================================================

class_counts = y.value_counts()

rare_classes = class_counts[class_counts < 5].index

print("\nRare classes being grouped:")

for cls in rare_classes:
    print(f"{cls}: {class_counts[cls]} records")

y = y.replace(rare_classes, "Rare")

print("\nTarget distribution after grouping:")
print(y.value_counts())

# ============================================================
# 4. FEATURES
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

# ============================================================
# 5. TARGET ENCODING
# ============================================================

target_encoder = LabelEncoder()

y_encoded = target_encoder.fit_transform(y)

print("\nTarget classes:")

for i, class_name in enumerate(target_encoder.classes_):
    print(f"{i}: {class_name}")

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
# 8. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
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
# 9. RANDOM FOREST PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# ============================================================
# 10. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {

    "classifier__n_estimators": [
        100,
        200,
        300,
        500
    ],

    "classifier__max_depth": [
        None,
        5,
        10,
        15,
        20,
        30
    ],

    "classifier__min_samples_split": [
        2,
        5,
        10
    ],

    "classifier__min_samples_leaf": [
        1,
        2,
        4
    ],

    "classifier__max_features": [
        "sqrt",
        "log2",
        None
    ],

    "classifier__class_weight": [
        None,
        "balanced",
        "balanced_subsample"
    ]
}

# ============================================================
# 11. RANDOMIZED SEARCH
# ============================================================

print("\nStarting Random Forest hyperparameter tuning...")
print("Cross-validation: 3-fold")
print("Selection metric: Macro F1")
print("Random search iterations: 30")

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="f1_macro",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

# ============================================================
# 12. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST RANDOM FOREST PARAMETERS")
print("=" * 70)

print(
    f"\nBest CV Macro F1: "
    f"{search.best_score_ * 100:.2f}%"
)

print("\nBest Parameters:")

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")

# ============================================================
# 13. BEST MODEL
# ============================================================

best_model = search.best_estimator_

print("\nEvaluating best Random Forest model...")

y_pred = best_model.predict(X_test)

# ============================================================
# 14. METRICS
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
# 15. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST TUNED RESULTS")
print("=" * 70)

print(
    f"\nAccuracy       : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision      : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall         : "
    f"{recall * 100:.2f}%"
)

print(
    f"Weighted F1    : "
    f"{weighted_f1 * 100:.2f}%"
)

print(
    f"Macro F1       : "
    f"{macro_f1 * 100:.2f}%"
)

# ============================================================
# 16. CLASSIFICATION REPORT
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
# 17. CREATE DIRECTORIES
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
# 18. SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "Algorithm": "Random Forest Tuned",
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "Weighted_F1": weighted_f1 * 100,
        "Macro_F1": macro_f1 * 100,
        "CV_Macro_F1": search.best_score_ * 100
    }
])

results.to_csv(
    "evaluation/random_forest_tuned_results.csv",
    index=False
)

# ============================================================
# 19. CONFUSION MATRIX
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
    "evaluation/random_forest_tuned_confusion_matrix.csv"
)

# ============================================================
# 20. SAVE BEST PARAMETERS
# ============================================================

best_params_df = pd.DataFrame(
    [
        {
            "Parameter": key,
            "Value": str(value)
        }
        for key, value in search.best_params_.items()
    ]
)

best_params_df.to_csv(
    "evaluation/random_forest_best_parameters.csv",
    index=False
)

# ============================================================
# 21. SAVE ALL TUNING RESULTS
# ============================================================

tuning_results = pd.DataFrame(
    search.cv_results_
)

tuning_results.to_csv(
    "evaluation/random_forest_tuning_results.csv",
    index=False
)

# ============================================================
# 22. SAVE TUNED PIPELINE
# ============================================================

joblib.dump(
    best_model,
    "models/random_forest_tuned_pipeline.pkl"
)

# ============================================================
# 23. SAVE TARGET ENCODER
# ============================================================

joblib.dump(
    target_encoder,
    "models/random_forest_tuned_target_encoder.pkl"
)

# ============================================================
# 24. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST TUNING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print(
    "evaluation/random_forest_tuned_results.csv"
)

print(
    "evaluation/random_forest_tuned_confusion_matrix.csv"
)

print(
    "evaluation/random_forest_best_parameters.csv"
)

print(
    "evaluation/random_forest_tuning_results.csv"
)

print(
    "models/random_forest_tuned_pipeline.pkl"
)

print(
    "models/random_forest_tuned_target_encoder.pkl"
)

print("\nBest Random Forest model saved successfully.")