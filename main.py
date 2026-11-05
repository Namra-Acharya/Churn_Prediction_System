from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).resolve().parent / "churn_predictor.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Run train.py first.")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
THRESHOLD = artifact["threshold"]
FEATURE_COLUMNS = artifact["feature_columns"]
CHURN_CLASS_INDEX = artifact["churn_class_index"]

app = FastAPI(
    title="Churn Predictor System",
    description="Customer churn prediction API using a trained ML pipeline.",
    version="2.0.0",
)


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    CustomerAge: int = Field(ge=0, le=120)
    TenureMonths: int = Field(ge=0, le=120)
    MonthlyCharges: float = Field(ge=0, le=1000)
    SupportTickets: int = Field(ge=0, le=100)
    UsageHours: float = Field(ge=0, le=1000)
    ContractType: Literal["Monthly", "Yearly"]


def score_customers(customers: list[Customer]) -> list[dict]:
    frame = pd.DataFrame([customer.model_dump() for customer in customers])[FEATURE_COLUMNS]
    probabilities = model.predict_proba(frame)[:, CHURN_CLASS_INDEX]
    results = []
    for probability in probabilities:
        churn_probability = float(probability)
        prediction = int(churn_probability >= THRESHOLD)
        results.append({
            "prediction": prediction,
            "label": "Churned" if prediction == 1 else "Not Churned",
            "churn_probability": round(churn_probability, 4),
            "threshold": THRESHOLD,
        })
    return results


@app.get("/")
def home():
    return {
        "message": "Churn Predictor System API is running.",
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {
        "model_name": artifact["model_name"],
        "threshold": THRESHOLD,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": artifact["numeric_columns"],
        "categorical_columns": artifact["categorical_columns"],
        "best_params": artifact["best_params"],
        "cv_f1": artifact["cv_f1"],
        "test_metrics": artifact["test_metrics"],
        "training_rows": artifact["training_rows"],
        "test_rows": artifact["test_rows"],
    }


@app.post("/predict")
def predict(customer: Customer):
    try:
        return score_customers([customer])[0]
    except Exception as exc:
        print(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        )


@app.post("/predict/batch")
def predict_batch(
    customers: list[Customer] = Body(..., min_length=1, max_length=1000),
):
    try:
        return {"predictions": score_customers(customers)}
    except Exception as exc:
        print(f"Batch prediction error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed. Check the server logs.",
        )
