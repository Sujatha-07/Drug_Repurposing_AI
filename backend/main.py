from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.predictor import predict_drug

MODEL_COMPARISON = {
    "Logistic Regression": 55.44,
    "Decision Tree": 49.74,
    "Random Forest": 53.37,
    "XGBoost": 57.51
}


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Drug Repurposing AI",
    description="AI-based drug repurposing prediction API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class DrugRequest(BaseModel):
    drug_name: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Drug Repurposing AI Backend is running"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(request: DrugRequest):

    try:

        result = predict_drug(
            request.drug_name
        )

        return {
            "success": True,
            "result": result,
            "model_comparison": MODEL_COMPARISON

        }

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:

        print("Prediction error:", e)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )