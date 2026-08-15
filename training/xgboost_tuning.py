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
    LabelEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "dataset/drug_repurposing.xlsx"

df = pd.read_excel(DATA_PATH)

print("=" * 70)
print("DRUG REPURPOSING AI - XGBOOST HYPERPARAMETER TUNING")
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

for i, cls in enumerate(
    target_encoder.classes_
):
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
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

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
# 9. XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    objective="multi:softprob",

    num_class=len(
        target_encoder.classes_
    ),

    eval_metric="mlogloss",

    random_state=42,

    tree_method="hist",

    n_jobs=1
)


# ============================================================
# 10. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# ============================================================
# 11. PARAMETER SEARCH SPACE
# ============================================================

param_grid = {

    "classifier__n_estimators": [
        100,
        200,
        300,
        500
    ],

    "classifier__learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1,
        0.2
    ],

    "classifier__max_depth": [
        3,
        4,
        5,
        6,
        8
    ],

    "classifier__min_child_weight": [
        1,
        3,
        5,
        7
    ],

    "classifier__subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "classifier__colsample_bytree": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "classifier__gamma": [
        0,
        0.1,
        0.3,
        0.5
    ],

    "classifier__reg_lambda": [
        0.5,
        1.0,
        2.0,
        5.0
    ]
}


# ============================================================
# 12. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ============================================================
# 13. RANDOMIZED SEARCH
# ============================================================

search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_grid,

    n_iter=20,

    scoring={
        "accuracy": "accuracy",

        "weighted_f1": "f1_weighted",

        "macro_f1": "f1_macro"
    },

    refit="macro_f1",

    cv=cv,

    verbose=2,

    random_state=42,

    n_jobs=-1,

    return_train_score=True
)


# ============================================================
# 14. START SEARCH
# ============================================================

print("\n" + "=" * 70)
print("STARTING XGBOOST HYPERPARAMETER SEARCH")
print("=" * 70)

print("\nParameter combinations: 20")
print("Cross-validation folds : 3")
print("Total model fits       : 60")

print(
    "\nModel selection metric: Macro F1"
)

search.fit(
    X_train,
    y_train
)


# ============================================================
# 15. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST XGBOOST PARAMETERS")
print("=" * 70)

print(
    f"\nBest CV Macro F1: "
    f"{search.best_score_ * 100:.2f}%"
)

print("\nBest Parameters:")

for parameter, value in search.best_params_.items():

    clean_parameter = parameter.replace(
        "classifier__",
        ""
    )

    print(
        f"{clean_parameter}: {value}"
    )


# ============================================================
# 16. SEARCH RESULTS
# ============================================================

cv_results = pd.DataFrame(
    search.cv_results_
)

cv_results["mean_test_accuracy"] *= 100
cv_results["mean_test_weighted_f1"] *= 100
cv_results["mean_test_macro_f1"] *= 100

cv_results = cv_results.sort_values(
    by="mean_test_macro_f1",
    ascending=False
)


print("\n" + "=" * 70)
print("TOP XGBOOST CONFIGURATIONS")
print("=" * 70)

display_columns = [
    "mean_test_accuracy",
    "mean_test_weighted_f1",
    "mean_test_macro_f1"
]

print(
    cv_results[
        display_columns
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# 17. BEST MODEL
# ============================================================

best_model = search.best_estimator_


# ============================================================
# 18. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

y_pred = best_model.predict(
    X_test
)


# ============================================================
# 19. METRICS
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
# 20. DISPLAY FINAL METRICS
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
# 21. CLASSIFICATION REPORT
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
# 22. CREATE RESULT FILE
# ============================================================

os.makedirs(
    "evaluation",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)


results = pd.DataFrame([{

    "Algorithm":
        "XGBoost Tuned",

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
    "evaluation/xgboost_tuned_results.csv",
    index=False
)


# ============================================================
# 23. SAVE ALL SEARCH RESULTS
# ============================================================

cv_results.to_csv(
    "evaluation/xgboost_tuning_results.csv",
    index=False
)


# ============================================================
# 24. CONFUSION MATRIX
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
    "evaluation/xgboost_tuned_confusion_matrix.csv"
)


# ============================================================
# 25. SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/xgboost_tuned_pipeline.pkl"
)


# ============================================================
# 26. SAVE TARGET ENCODER
# ============================================================

joblib.dump(
    target_encoder,
    "models/xgboost_tuned_target_encoder.pkl"
)


# ============================================================
# 27. SAVE BEST PARAMETERS
# ============================================================

best_parameters = {
    key.replace(
        "classifier__",
        ""
    ): value

    for key, value
    in search.best_params_.items()
}

best_parameters_df = pd.DataFrame(
    [best_parameters]
)

best_parameters_df.to_csv(
    "evaluation/xgboost_best_parameters.csv",
    index=False
)


# ============================================================
# 28. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST TUNING COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(
    "evaluation/xgboost_tuned_results.csv"
)

print(
    "evaluation/xgboost_tuning_results.csv"
)

print(
    "evaluation/xgboost_tuned_confusion_matrix.csv"
)

print(
    "evaluation/xgboost_best_parameters.csv"
)

print(
    "models/xgboost_tuned_pipeline.pkl"
)

print(
    "models/xgboost_tuned_target_encoder.pkl"
)

print("\nBest XGBoost pipeline saved successfully.")