import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
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
print("DRUG REPURPOSING AI - DECISION TREE")
print("=" * 70)

print("\nOriginal dataset shape:", df.shape)


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

for i, class_name in enumerate(
    target_encoder.classes_
):
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
# 9. PREPROCESS
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
# 10. DECISION TREE MODEL
# ============================================================

model = DecisionTreeClassifier(

    criterion="gini",

    random_state=42

)


# ============================================================
# 11. TRAIN
# ============================================================

print("\nTraining Decision Tree...")

model.fit(
    X_train_processed,
    y_train
)

print("Decision Tree training completed.")


# ============================================================
# 12. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_processed
)


# ============================================================
# 13. METRICS
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
# 14. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE RESULTS")
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
# 15. CLASSIFICATION REPORT
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
# 16. CREATE DIRECTORIES
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
# 17. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{

    "Algorithm":
        "Decision Tree",

    "Accuracy":
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
    "evaluation/decision_tree_results.csv",
    index=False
)


# ============================================================
# 18. SAVE CONFUSION MATRIX
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
    "evaluation/decision_tree_confusion_matrix.csv"
)


# ============================================================
# 19. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/decision_tree_model.pkl"
)

joblib.dump(
    preprocessor,
    "models/decision_tree_preprocessor.pkl"
)

joblib.dump(
    target_encoder,
    "models/decision_tree_target_encoder.pkl"
)


# ============================================================
# 20. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE COMPLETED")
print("=" * 70)

print("\nFiles saved:")

print(
    "evaluation/decision_tree_results.csv"
)

print(
    "evaluation/decision_tree_confusion_matrix.csv"
)

print(
    "models/decision_tree_model.pkl"
)

print(
    "models/decision_tree_preprocessor.pkl"
)

print(
    "models/decision_tree_target_encoder.pkl"
)