import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "dataset/drug_repurposing.xlsx"

df = pd.read_excel(DATA_PATH)

print("=" * 70)
print("DRUG REPURPOSING AI - LOGISTIC REGRESSION TUNING")
print("=" * 70)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. TARGET
# ============================================================

y = df["Repurposed_Category"].copy()


# ============================================================
# 3. GROUP VERY RARE CLASSES
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

for i, cls in enumerate(target_encoder.classes_):
    print(f"{i}: {cls}")


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
        ),
        (
            "scaler",
            StandardScaler()
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
# 9. LOGISTIC REGRESSION PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 10. PARAMETER SEARCH SPACE
# ============================================================

param_grid = {
    'classifier__C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    'classifier__solver': ['lbfgs'],
    'classifier__class_weight': [None, 'balanced']
}

# ============================================================
# 11. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ============================================================
# 12. RANDOMIZED SEARCH
# ============================================================

search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_grid,

    n_iter=20,

    scoring="f1_macro",

    cv=cv,

    verbose=2,

    random_state=42,

    n_jobs=-1,

    return_train_score=True
)


# ============================================================
# 13. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING LOGISTIC REGRESSION HYPERPARAMETER SEARCH")
print("=" * 70)

print("\n20 parameter combinations")
print("3-fold cross-validation")
print("Selection metric: Macro F1")

search.fit(
    X_train,
    y_train
)


# ============================================================
# 14. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST LOGISTIC REGRESSION PARAMETERS")
print("=" * 70)

print(
    f"\nBest CV Macro F1: "
    f"{search.best_score_ * 100:.2f}%"
)

print("\nBest Parameters:")

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")


# ============================================================
# 15. BEST MODEL
# ============================================================

best_model = search.best_estimator_


# ============================================================
# 16. FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

y_pred = best_model.predict(X_test)


# ============================================================
# 17. METRICS
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
# 18. DISPLAY
# ============================================================

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
# 19. CLASSIFICATION REPORT
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
# 20. SAVE DIRECTORIES
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
# 21. SAVE MAIN RESULTS
# ============================================================

results = pd.DataFrame([{

    "Algorithm":
        "Logistic Regression Tuned",

    "CV_Macro_F1":
        search.best_score_ * 100,

    "Test_Accuracy":
        accuracy * 100,

    "Precision":
        precision * 100,

    "Recall":
        recall * 100,

    "Weighted_F1":
        weighted_f1 * 100,

    "Macro_F1":
        macro_f1 * 100

}])

results.to_csv(
    "evaluation/logistic_regression_tuned_results.csv",
    index=False
)


# ============================================================
# 22. SAVE SEARCH RESULTS
# ============================================================

cv_results = pd.DataFrame(
    search.cv_results_
)

cv_results.to_csv(
    "evaluation/logistic_regression_tuning_results.csv",
    index=False
)


# ============================================================
# 23. CONFUSION MATRIX
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
    "evaluation/logistic_regression_tuned_confusion_matrix.csv"
)


# ============================================================
# 24. SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/logistic_regression_tuned_pipeline.pkl"
)

joblib.dump(
    target_encoder,
    "models/logistic_regression_tuned_target_encoder.pkl"
)


# ============================================================
# 25. SAVE BEST PARAMETERS
# ============================================================

best_parameters_df = pd.DataFrame(
    [search.best_params_]
)

best_parameters_df.to_csv(
    "evaluation/logistic_regression_best_parameters.csv",
    index=False
)


# ============================================================
# 26. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION TUNING COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(
    "evaluation/logistic_regression_tuned_results.csv"
)

print(
    "evaluation/logistic_regression_tuning_results.csv"
)

print(
    "evaluation/logistic_regression_tuned_confusion_matrix.csv"
)

print(
    "evaluation/logistic_regression_best_parameters.csv"
)

print(
    "models/logistic_regression_tuned_pipeline.pkl"
)

print(
    "models/logistic_regression_tuned_target_encoder.pkl"
)

print("\nBest Logistic Regression model saved successfully.")