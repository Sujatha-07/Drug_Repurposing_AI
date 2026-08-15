import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "dataset/drug_repurposing.xlsx"

df = pd.read_excel(DATA_PATH)

print("=" * 70)
print("DRUG REPURPOSING AI - DECISION TREE TUNING")
print("=" * 70)

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

y = y.replace(
    rare_classes,
    "Rare"
)

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
# 9. DECISION TREE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42
            )
        )
    ]
)


# ============================================================
# 10. PARAMETER GRID
# ============================================================

param_grid = {

    "classifier__criterion": [
        "gini",
        "entropy",
        "log_loss"
    ],

    "classifier__max_depth": [
        None,
        3,
        5,
        7,
        10,
        15,
        20
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

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# ============================================================
# 11. GRID SEARCH
# ============================================================

print("\nStarting Decision Tree hyperparameter tuning...")

print("Cross-validation: 3-fold")
print("Selection metric: Macro F1")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 12. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST DECISION TREE PARAMETERS")
print("=" * 70)

print(
    "\nBest CV Macro F1: "
    f"{grid_search.best_score_ * 100:.2f}%"
)

print("\nBest Parameters:")

for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")


# ============================================================
# 13. BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_


# ============================================================
# 14. TEST PREDICTION
# ============================================================

print("\nEvaluating best Decision Tree model...")

y_pred = best_model.predict(X_test)


# ============================================================
# 15. METRICS
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
# 16. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TUNED DECISION TREE RESULTS")
print("=" * 70)

print(
    f"\nAccuracy       : {accuracy * 100:.2f}%"
)

print(
    f"Precision      : {precision * 100:.2f}%"
)

print(
    f"Recall         : {recall * 100:.2f}%"
)

print(
    f"Weighted F1    : {weighted_f1 * 100:.2f}%"
)

print(
    f"Macro F1       : {macro_f1 * 100:.2f}%"
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
# 19. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{

    "Algorithm": "Decision Tree Tuned",

    "Accuracy": accuracy * 100,

    "Precision": precision * 100,

    "Recall": recall * 100,

    "Weighted_F1": weighted_f1 * 100,

    "Macro_F1": macro_f1 * 100

}])

results.to_csv(
    "evaluation/decision_tree_tuned_results.csv",
    index=False
)


# ============================================================
# 20. SAVE CONFUSION MATRIX
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
    "evaluation/decision_tree_tuned_confusion_matrix.csv"
)


# ============================================================
# 21. SAVE BEST PARAMETERS
# ============================================================

best_parameters = pd.DataFrame(
    [
        {
            "Parameter": parameter,
            "Value": str(value)
        }
        for parameter, value
        in grid_search.best_params_.items()
    ]
)

best_parameters.to_csv(
    "evaluation/decision_tree_best_parameters.csv",
    index=False
)


# ============================================================
# 22. SAVE CV RESULTS
# ============================================================

cv_results = pd.DataFrame(
    grid_search.cv_results_
)

cv_results.to_csv(
    "evaluation/decision_tree_tuning_results.csv",
    index=False
)


# ============================================================
# 23. SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/decision_tree_tuned_pipeline.pkl"
)

joblib.dump(
    target_encoder,
    "models/decision_tree_tuned_target_encoder.pkl"
)


# ============================================================
# 24. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE TUNING COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(
    "evaluation/decision_tree_tuned_results.csv"
)

print(
    "evaluation/decision_tree_tuned_confusion_matrix.csv"
)

print(
    "evaluation/decision_tree_best_parameters.csv"
)

print(
    "evaluation/decision_tree_tuning_results.csv"
)

print(
    "models/decision_tree_tuned_pipeline.pkl"
)

print(
    "models/decision_tree_tuned_target_encoder.pkl"
)

print("\nBest Decision Tree model saved successfully.")