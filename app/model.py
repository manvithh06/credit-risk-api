"""
model.py — Load the trained model once at startup.

WHY load at startup not per-request:
Loading a model file takes ~100-500ms. If you loaded it on every
request, your API would be 10x slower. Load once, reuse forever.
"""

import joblib
import numpy as np
import os

MODEL_PATH = "model/credit_model.joblib"
MODEL_VERSION = "1.0.0"

FEATURE_NAMES = [
    'age', 'income', 'loan_amount', 'credit_score',
    'employment_years', 'debt_to_income', 'num_credit_lines',
    'payment_history', 'loan_to_value', 'num_late_payments'
]

# Load model at module import time — happens once when API starts
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

def predict(features: dict) -> dict:
    """
    Run prediction on a single input.
    Returns prediction (0/1) and probability.
    """
    # Convert dict to array in correct feature order
    X = np.array([[features[f] for f in FEATURE_NAMES]])
    
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])  # prob of default
    
    return {
        'prediction': prediction,
        'probability_default': round(probability, 4),
        'risk_label': 'HIGH RISK' if prediction == 1 else 'LOW RISK',
        'model_version': MODEL_VERSION
    }