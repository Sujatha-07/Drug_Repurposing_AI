import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "dataset/drug_repurposing.xlsx"

df = pd.read_excel(DATA_PATH)

print("=" * 80)
print("DRUG REPURPOSING AI - FAIR MODEL COMPARISON")
print("=" * 80)

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
# 6. SAME TRAIN / TEST SPLIT FOR ALL MODELS
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
# 8. COMMON PREPROCESSING
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
# 9. PREPROCESS ONCE
# ============================================================

print("\nPreprocessing...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# 10. MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=5000,
        solver="lbfgs",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 11. TRAIN AND EVALUATE
# ============================================================

results = []

os.makedirs("evaluation", exist_ok=True)

for name, model in models.items():

    print("\n" + "=" * 80)
    print(f"TRAINING: {name}")
    print("=" * 80)

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(
        X_test_processed
    )

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

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "Weighted_F1": weighted_f1 * 100,
        "Macro_F1": macro_f1 * 100
    })

    print(f"\n{name} Results")

    print(
        f"Accuracy       : {accuracy * 100:.2f}%"
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
# 12. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n" + "=" * 80)
print("FINAL FAIR MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(index=False)
)


# ============================================================
# 13. BEST MODELS
# ============================================================

best_accuracy = results_df.loc[
    results_df["Accuracy"].idxmax()
]

best_precision = results_df.loc[
    results_df["Precision"].idxmax()
]

best_weighted_f1 = results_df.loc[
    results_df["Weighted_F1"].idxmax()
]

best_macro_f1 = results_df.loc[
    results_df["Macro_F1"].idxmax()
]

print("\n" + "=" * 80)
print("BEST MODELS")
print("=" * 80)

print(
    f"Best Accuracy     : "
    f"{best_accuracy['Algorithm']} "
    f"({best_accuracy['Accuracy']:.2f}%)"
)

print(
    f"Best Precision    : "
    f"{best_precision['Algorithm']} "
    f"({best_precision['Precision']:.2f}%)"
)

print(
    f"Best Weighted F1  : "
    f"{best_weighted_f1['Algorithm']} "
    f"({best_weighted_f1['Weighted_F1']:.2f}%)"
)

print(
    f"Best Macro F1     : "
    f"{best_macro_f1['Algorithm']} "
    f"({best_macro_f1['Macro_F1']:.2f}%)"
)


# ============================================================
# 14. SAVE COMPARISON
# ============================================================

output_path = (
    "evaluation/"
    "fair_model_comparison.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print(
    f"\nComparison saved to: {output_path}"
)

print("\nFAIR MODEL COMPARISON COMPLETED")