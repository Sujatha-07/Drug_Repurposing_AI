# python -m training.xgboost_188_features

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "dataset/drug_repurposing.xlsx"

df = pd.read_excel(DATA_PATH)

# ============================================================
# CREATE NUM_TARGETS FEATURE
# ============================================================

def count_targets(value):
    if pd.isna(value) or str(value).strip() == "":
        return 0

    value = str(value)

    # Handle multiple target proteins separated by common delimiters
    targets = [
        x.strip()
        for x in value.replace("|", ";").replace(",", ";").split(";")
        if x.strip()
    ]

    return len(targets)


df["Num_Targets"] = df["Target_Protein_UniProt"].apply(count_targets)

print("\nNum_Targets feature created.")
print(df["Num_Targets"].describe())

print("=" * 70)
print("DRUG REPURPOSING AI - XGBOOST 188 FEATURE MODEL")
print("=" * 70)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. TARGET
# ============================================================

# IMPORTANT:
# Do NOT group Endocrine, Liver or Diabetes here.
# The earlier 58.55% experiment used 21 classes.

# ============================================================
# TARGET
# ============================================================

y = df["Repurposed_Category"].copy()

# ============================================================
# GROUP RARE CLASSES
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
# 3. TARGET ENCODING
# ============================================================

target_encoder = LabelEncoder()

y_encoded = target_encoder.fit_transform(y)

print("\nTarget classes:")

for i, cls in enumerate(target_encoder.classes_):
    print(f"{i}: {cls}")

print("\nNumber of classes:", len(target_encoder.classes_))


# ============================================================
# 4. FEATURES
# ============================================================

numeric_features = [
    "Molecular_Weight",
    "LogP",
    "Hydrogen_Bond_Donors",
    "Hydrogen_Bond_Acceptors",
    "TPSA",
    "Rotatable_Bonds",
    "Num_Targets"
]

categorical_feature = "Drug_Class"

text_feature = "Current_Use"


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    numeric_features
    + [categorical_feature]
    + [text_feature]
    + ["Repurposed_Category"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X = df[
    numeric_features
    + [categorical_feature]
    + [text_feature]
].copy()

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
# 7. NUMERIC FEATURES
# ============================================================

print("\nProcessing numeric features...")

numeric_imputer = SimpleImputer(
    strategy="median"
)

X_train_numeric = numeric_imputer.fit_transform(
    X_train[numeric_features]
)

X_test_numeric = numeric_imputer.transform(
    X_test[numeric_features]
)

X_train_numeric = csr_matrix(X_train_numeric)
X_test_numeric = csr_matrix(X_test_numeric)


# ============================================================
# 8. DRUG CLASS ONE-HOT ENCODING
# ============================================================

print("Processing Drug_Class...")

drug_class_imputer = SimpleImputer(
    strategy="most_frequent"
)

train_drug_class = drug_class_imputer.fit_transform(
    X_train[[categorical_feature]]
)

test_drug_class = drug_class_imputer.transform(
    X_test[[categorical_feature]]
)

onehot = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

X_train_drug_class = onehot.fit_transform(
    train_drug_class
)

X_test_drug_class = onehot.transform(
    test_drug_class
)

print(
    "Drug_Class features:",
    X_train_drug_class.shape[1]
)


# ============================================================
# 9. TF-IDF FEATURES
# ============================================================

print("Processing Current_Use using TF-IDF...")

train_text = (
    X_train[text_feature]
    .fillna("")
    .astype(str)
)

test_text = (
    X_test[text_feature]
    .fillna("")
    .astype(str)
)

tfidf = TfidfVectorizer(
    max_features=150,
    lowercase=True,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(
    train_text
)

X_test_tfidf = tfidf.transform(
    test_text
)

print(
    "TF-IDF features:",
    X_train_tfidf.shape[1]
)


# ============================================================
# 10. COMBINE FEATURES
# ============================================================

X_train_processed = hstack([
    X_train_numeric,
    X_train_drug_class,
    X_train_tfidf
]).tocsr()

X_test_processed = hstack([
    X_test_numeric,
    X_test_drug_class,
    X_test_tfidf
]).tocsr()


print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print(
    "Numeric features      :",
    X_train_numeric.shape[1]
)

print(
    "Drug_Class features   :",
    X_train_drug_class.shape[1]
)

print(
    "TF-IDF features       :",
    X_train_tfidf.shape[1]
)

print(
    "TOTAL FEATURES        :",
    X_train_processed.shape[1]
)

print(
    "Processed train shape :",
    X_train_processed.shape
)

print(
    "Processed test shape  :",
    X_test_processed.shape
)


# ============================================================
# 11. XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(target_encoder.classes_),

    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,

    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,

    gamma=0,
    reg_lambda=1,

    eval_metric="mlogloss",

    random_state=42,
    tree_method="hist",
    n_jobs=-1
)


# ============================================================
# 12. TRAIN
# ============================================================

model.fit(
    X_train_processed,
    y_train
)

print("XGBoost training completed.")


# ============================================================
# 13. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_processed
)


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
print("XGBOOST RESULTS")
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

results = pd.DataFrame([{

    "Algorithm": "XGBoost 188 Features",

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
    "evaluation/xgboost_188_feature_results.csv",
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
    "evaluation/xgboost_188_feature_confusion_matrix.csv"
)


# ============================================================
# 20. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/xgboost_188_feature_model.pkl"
)

joblib.dump(
    target_encoder,
    "models/xgboost_188_feature_target_encoder.pkl"
)

joblib.dump(
    numeric_imputer,
    "models/xgboost_188_feature_numeric_imputer.pkl"
)

joblib.dump(
    drug_class_imputer,
    "models/xgboost_188_feature_drug_class_imputer.pkl"
)

joblib.dump(
    onehot,
    "models/xgboost_188_feature_onehot.pkl"
)

joblib.dump(
    tfidf,
    "models/xgboost_188_feature_tfidf.pkl"
)


# ============================================================
# 21. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST 188 FEATURE MODEL COMPLETED")
print("=" * 70)

print("\nFiles saved:")

print(
    "evaluation/xgboost_188_feature_results.csv"
)

print(
    "evaluation/xgboost_188_feature_confusion_matrix.csv"
)

print(
    "models/xgboost_188_feature_model.pkl"
)

print(
    "models/xgboost_188_feature_target_encoder.pkl"
)

print(
    "models/xgboost_188_feature_numeric_imputer.pkl"
)

print(
    "models/xgboost_188_feature_drug_class_imputer.pkl"
)

print(
    "models/xgboost_188_feature_onehot.pkl"
)

print(
    "models/xgboost_188_feature_tfidf.pkl"
)

print("\nXGBoost model saved successfully.")