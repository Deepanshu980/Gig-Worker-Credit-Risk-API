from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import json
from datetime import datetime

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(
    title="Gig Worker Credit Risk API",
    description="Batch & Single Credit Risk Prediction API",
    version="1.0"
)

class WorkerData(BaseModel):
    avg_daily_income: float
    monthly_income: float
    weekly_deliveries: int
    active_days: int
    rating: float
    income_std: float
    account_age_months: int
    cancel_rate: float

class WorkerBatch(BaseModel):
    workers: List[WorkerData]


def risk_segment(score: int) -> str:
    if score >= 750:
        return "Low Risk"
    elif score >= 600:
        return "Medium Risk"
    else:
        return "High Risk"

@app.get("/")
def health():
    return {"status": "API is running successfully"}


@app.post("/predict")
def predict_single(data: WorkerData):

    X = np.array([[
        data.avg_daily_income,
        data.monthly_income,
        data.weekly_deliveries,
        data.active_days,
        data.rating,
        data.income_std,
        data.account_age_months,
        data.cancel_rate,
        
    ]])

    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    credit_score = int(900 - prob * 600)

    return {
        "probability_of_default": round(float(prob), 4),
        "credit_score": credit_score,
        "risk_segment": risk_segment(credit_score)
    }


@app.post("/predict/batch")
def predict_batch(batch: WorkerBatch):

    X = np.array([
        [
            w.avg_daily_income,
            w.monthly_income,
            w.weekly_deliveries,
            w.active_days,
            w.rating,
            w.income_std,
            w.account_age_months,
            w.cancel_rate,
            w.platform_Zomato
        ]
        for w in batch.workers
    ])

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    scores = (900 - probs * 600).astype(int)

    predictions = []
    for i in range(len(probs)):
        predictions.append({
            "id": i + 1,
            "probability_of_default": round(float(probs[i]), 4),
            "credit_score": int(scores[i]),
            "risk_segment": risk_segment(int(scores[i]))
        })

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_records": len(predictions),
        "predictions": predictions
    }

   
    with open("batch_result.json", "w") as f:
        json.dump(output, f, indent=4)

    return output
