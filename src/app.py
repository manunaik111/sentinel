# src/app.py
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Sentinel — Churn Prediction API",
    description="Predicts customer churn probability using a trained ML model.",
    version="1.0.0"
)

# Load model and scaler
model = joblib.load("src/best_model.pkl")
scaler = joblib.load("src/scaler.pkl")

# ── Input Schema ─────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ── Feature Engineering (mirrors model.py) ───────────────────────────────────
def prepare_features(data: CustomerInput) -> np.ndarray:
    d = data.dict()

    # Engineered features
    d['tenure_group'] = pd.cut([d['tenure']],
                                bins=[0, 12, 24, 48, 72],
                                labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])[0]
    d['charges_per_tenure'] = d['TotalCharges'] / (d['tenure'] + 1)
    d['has_support_services'] = (
        int(d['TechSupport'] == 'Yes') +
        int(d['OnlineSecurity'] == 'Yes') +
        int(d['OnlineBackup'] == 'Yes')
    )
    d['is_echeque'] = int(d['PaymentMethod'] == 'Electronic check')

    # Encode all object/category fields to int
    encode_map = {
        'gender': {'Male': 1, 'Female': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'MultipleLines': {'Yes': 2, 'No': 1, 'No phone service': 0},
        'InternetService': {'Fiber optic': 2, 'DSL': 1, 'No': 0},
        'OnlineSecurity': {'Yes': 2, 'No': 1, 'No internet service': 0},
        'OnlineBackup': {'Yes': 2, 'No': 1, 'No internet service': 0},
        'DeviceProtection': {'Yes': 2, 'No': 1, 'No internet service': 0},
        'TechSupport': {'Yes': 2, 'No': 1, 'No internet service': 0},
        'StreamingTV': {'Yes': 2, 'No': 1, 'No internet service': 0},
        'StreamingMovies': {'Yes': 2, 'No': 1, 'No internet service': 0},
        'Contract': {'Two year': 2, 'One year': 1, 'Month-to-month': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'PaymentMethod': {
            'Electronic check': 3, 'Mailed check': 2,
            'Bank transfer (automatic)': 1, 'Credit card (automatic)': 0
        },
        'tenure_group': {'4-6yr': 3, '2-4yr': 2, '1-2yr': 1, '0-1yr': 0}
    }

    for col, mapping in encode_map.items():
        d[col] = mapping.get(str(d[col]), 0)

    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'tenure_group',
        'charges_per_tenure', 'has_support_services', 'is_echeque'
    ]

    features = np.array([d[f] for f in feature_order]).reshape(1, -1)
    return scaler.transform(features)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Sentinel Churn Prediction API is running ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: CustomerInput):
    features = prepare_features(customer)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": (
            "High" if probability >= 0.7 else
            "Medium" if probability >= 0.4 else
            "Low"
        ),
        "message": (
            "⚠️ Customer is at risk of churning." if prediction == 1
            else "✅ Customer is likely to stay."
        )
    }