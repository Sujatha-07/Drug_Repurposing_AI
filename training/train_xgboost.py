import os
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Import the common preprocessing results
from preprocessing.preprocess import (
    X_train_processed,
    X_test_processed,
    y_train,
    y_test,
    preprocessor,
    target_encoder
)


# ============================================================
# 1. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)


# ============================================================
# 2. DISPLAY DATA INFORMATION
# ============================================================

print("=" * 70)
print("XGBOOST DRUG REPURPOSING MODEL")
print("=" * 70)

print("\nTraining samples:", X_train_processed.shape[0])
print("Testing samples :", X_test_processed.shape[0])
print("Number of features:", X_train_processed.shape[1])
print("Number of classes :", len(target_encoder.classes_))


# ============================================================
# 3. CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(target_encoder.classes_),

    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_lambda=1.0,

    eval_metric="mlogloss",

    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train_processed,
    y_train
)

print("XGBoost training completed.")


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_processed)


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

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

f1 = f1_score(
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
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST RESULTS")
print("=" * 70)

print(f"\nAccuracy        : {accuracy * 100:.2f}%")
print(f"Weighted F1     : {f1 * 100:.2f}%")
print(f"Macro F1        : {macro_f1 * 100:.2f}%")
print(f"Weighted Recall : {recall * 100:.2f}%")
print(f"Weighted Precision: {precision * 100:.2f}%")


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    target_names=target_encoder.classes_,
    zero_division=0
)

print(report)


# ============================================================
# 9. SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_test,
    y_pred,
    target_names=target_encoder.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

report_df.to_csv(
    "evaluation/xgboost_classification_report.csv"
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=target_encoder.classes_,
    columns=target_encoder.classes_
)

cm_df.to_csv(
    "evaluation/xgboost_confusion_matrix.csv"
)

print("\nConfusion matrix saved.")


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    "evaluation/xgboost_feature_importance.csv",
    index=False
)

print("Feature importance saved.")


# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/xgboost_model.pkl"
)

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

joblib.dump(
    target_encoder,
    "models/target_encoder.pkl"
)

print("\nModel saved:")
print("models/xgboost_model.pkl")

print("\nPreprocessor saved:")
print("models/preprocessor.pkl")

print("\nTarget encoder saved:")
print("models/target_encoder.pkl")


# ============================================================
# 13. SAVE SUMMARY RESULTS
# ============================================================

results = pd.DataFrame([{
    "Algorithm": "XGBoost",
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1_Score": f1 * 100,
    "Macro_F1": macro_f1 * 100
}])

results.to_csv(
    "evaluation/xgboost_results.csv",
    index=False
)

print("\nResults saved:")
print("evaluation/xgboost_results.csv")

print("\n" + "=" * 70)
print("XGBOOST PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)