import pandas as pd
import os

print("=" * 70)
print("DRUG REPURPOSING AI - MODEL COMPARISON")
print("=" * 70)


# ============================================================
# 1. LOAD RESULTS
# ============================================================

files = {
    "XGBoost": "evaluation/xgboost_results.csv",
    "Decision Tree": "evaluation/decision_tree_results.csv",
    "Random Forest": "evaluation/random_forest_results.csv",
    "Logistic Regression": "evaluation/logistic_regression_results.csv"
}


results = []


# ============================================================
# 2. READ EACH MODEL RESULT
# ============================================================

for algorithm, file_path in files.items():

    df = pd.read_csv(file_path)

    row = df.iloc[0].to_dict()

    # --------------------------------------------
    # Handle different F1 column names
    # --------------------------------------------

    if "Weighted_F1" in row and pd.notna(row["Weighted_F1"]):
        weighted_f1 = row["Weighted_F1"]

    elif "F1_Score" in row and pd.notna(row["F1_Score"]):
        weighted_f1 = row["F1_Score"]

    elif "Weighted F1" in row and pd.notna(row["Weighted F1"]):
        weighted_f1 = row["Weighted F1"]

    else:
        weighted_f1 = None


    results.append({
        "Algorithm": algorithm,
        "Accuracy": row["Accuracy"],
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "Weighted_F1": weighted_f1,
        "Macro_F1": row["Macro_F1"]
    })


# ============================================================
# 3. CREATE COMPARISON DATAFRAME
# ============================================================

comparison = pd.DataFrame(results)


# ============================================================
# 4. ROUND VALUES
# ============================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "Weighted_F1",
    "Macro_F1"
]

comparison[metric_columns] = comparison[
    metric_columns
].round(2)


# ============================================================
# 5. DISPLAY COMPARISON
# ============================================================

print("\nMODEL PERFORMANCE COMPARISON")
print("-" * 70)

print(
    comparison.to_string(index=False)
)


# ============================================================
# 6. BEST MODELS
# ============================================================

print("\n" + "=" * 70)
print("BEST MODELS")
print("=" * 70)


best_accuracy = comparison.loc[
    comparison["Accuracy"].idxmax()
]

best_precision = comparison.loc[
    comparison["Precision"].idxmax()
]

best_recall = comparison.loc[
    comparison["Recall"].idxmax()
]

best_weighted_f1 = comparison.loc[
    comparison["Weighted_F1"].idxmax()
]

best_macro_f1 = comparison.loc[
    comparison["Macro_F1"].idxmax()
]


print(
    f"\nBest Accuracy : "
    f"{best_accuracy['Algorithm']} "
    f"({best_accuracy['Accuracy']:.2f}%)"
)

print(
    f"Best Precision: "
    f"{best_precision['Algorithm']} "
    f"({best_precision['Precision']:.2f}%)"
)

print(
    f"Best Recall   : "
    f"{best_recall['Algorithm']} "
    f"({best_recall['Recall']:.2f}%)"
)

print(
    f"Best Weighted F1: "
    f"{best_weighted_f1['Algorithm']} "
    f"({best_weighted_f1['Weighted_F1']:.2f}%)"
)

print(
    f"Best Macro F1: "
    f"{best_macro_f1['Algorithm']} "
    f"({best_macro_f1['Macro_F1']:.2f}%)"
)


# ============================================================
# 7. RANKING
# ============================================================

comparison["Accuracy_Rank"] = (
    comparison["Accuracy"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


comparison = comparison.sort_values(
    by="Accuracy_Rank"
)


# ============================================================
# 8. SAVE
# ============================================================

os.makedirs(
    "evaluation",
    exist_ok=True
)


comparison.to_csv(
    "evaluation/model_comparison.csv",
    index=False
)


print("\nComparison saved:")
print(
    "evaluation/model_comparison.csv"
)


print("\nModel comparison completed successfully.")