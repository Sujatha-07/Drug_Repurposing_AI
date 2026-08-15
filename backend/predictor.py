import os
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "drug_repurposing.xlsx"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_excel(DATA_PATH)


# ============================================================
# CREATE NUM_TARGETS FEATURE
# Same logic used during XGBoost training
# ============================================================

def count_targets(value):
    if pd.isna(value) or str(value).strip() == "":
        return 0

    value = str(value)

    targets = [
        x.strip()
        for x in value.replace("|", ";")
        .replace(",", ";")
        .split(";")
        if x.strip()
    ]

    return len(targets)


df["Num_Targets"] = df["Target_Protein_UniProt"].apply(
    count_targets
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost_188_feature_model.pkl"
    )
)

target_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost_188_feature_target_encoder.pkl"
    )
)

numeric_imputer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost_188_feature_numeric_imputer.pkl"
    )
)

drug_class_imputer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost_188_feature_drug_class_imputer.pkl"
    )
)

onehot = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost_188_feature_onehot.pkl"
    )
)

tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost_188_feature_tfidf.pkl"
    )
)


# ============================================================
# FEATURE DEFINITIONS
# Must match training exactly
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
# PREDICT DRUG
# ============================================================

def predict_drug(drug_name: str):

    if not drug_name or not drug_name.strip():
        raise ValueError("Drug name cannot be empty.")

    search_name = drug_name.strip().lower()

    # --------------------------------------------------------
    # Find drug
    # --------------------------------------------------------

    matches = df[
        df["Drug_Name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        == search_name
    ]

    if matches.empty:
        raise ValueError(
            f"Drug '{drug_name}' was not found in the dataset."
        )

    row = matches.iloc[0]

    # --------------------------------------------------------
    # Create one-row DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame([{
        "Molecular_Weight": row["Molecular_Weight"],
        "LogP": row["LogP"],
        "Hydrogen_Bond_Donors": row["Hydrogen_Bond_Donors"],
        "Hydrogen_Bond_Acceptors": row["Hydrogen_Bond_Acceptors"],
        "TPSA": row["TPSA"],
        "Rotatable_Bonds": row["Rotatable_Bonds"],
        "Num_Targets": row["Num_Targets"],
        "Drug_Class": row["Drug_Class"],
        "Current_Use": row["Current_Use"]
    }])

    # ========================================================
    # NUMERIC FEATURES
    # ========================================================

    X_numeric = numeric_imputer.transform(
        X[numeric_features]
    )

    X_numeric = csr_matrix(X_numeric)

    # ========================================================
    # DRUG CLASS
    # ========================================================

    X_drug_class = drug_class_imputer.transform(
        X[[categorical_feature]]
    )

    X_drug_class = onehot.transform(
        X_drug_class
    )

    # ========================================================
    # CURRENT USE TF-IDF
    # ========================================================

    X_text = (
        X[text_feature]
        .fillna("")
        .astype(str)
    )

    X_tfidf = tfidf.transform(X_text)

    # ========================================================
    # COMBINE
    # ========================================================

    X_processed = hstack([
        X_numeric,
        X_drug_class,
        X_tfidf
    ]).tocsr()

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        X_processed
    )[0]

    probabilities = model.predict_proba(
        X_processed
    )[0]

    predicted_category = target_encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = float(
        probabilities[prediction] * 100
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "drug_name": str(row["Drug_Name"]),
        "current_use": (
            "No Current Use"
            if pd.isna(row["Current_Use"])
            else str(row["Current_Use"])
        ),
        "repurposed_use": (
            "No Repurposed Use"
            if pd.isna(row["Repurposed_Use"])
            else str(row["Repurposed_Use"])
        ),
        "category": str(predicted_category),
        "confidence": round(confidence, 2)
    }