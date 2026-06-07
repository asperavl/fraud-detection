import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

model = joblib.load('model.joblib')
features = joblib.load('features.joblib')

class Transaction(BaseModel):
    data: dict

@app.get("/")
def root():
    return {"status": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    df = pd.DataFrame([transaction.data])

    df = df.reindex(columns=features,fill_value=-999)

    prob = model.predict_proba(df)[0][1]
    verdict = "Fraudulent" if prob>0.5 else "Legitimate"

    return {
        "fraud_probability": round(float(prob), 4),
        "verdict": verdict
    }