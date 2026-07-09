"""
schemas.py — Pydantic models for request/response validation.

WHY Pydantic: FastAPI uses these to automatically validate incoming JSON.
If a request is missing 'age' or sends a string instead of a number,
FastAPI returns a 422 error automatically — you don't write that logic.
"""

from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    """Input schema — exactly what the /predict endpoint expects."""
    age: float = Field(..., ge=18, le=100, description="Age in years")
    income: float = Field(..., ge=0, description="Annual income")
    loan_amount: float = Field(..., ge=0, description="Requested loan amount")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score")
    employment_years: float = Field(..., ge=0, description="Years at current employer")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio")
    num_credit_lines: float = Field(..., ge=0, description="Number of open credit lines")
    payment_history: float = Field(..., description="Payment history score")
    loan_to_value: float = Field(..., ge=0, description="Loan-to-value ratio")
    num_late_payments: float = Field(..., ge=0, description="Number of late payments")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "income": 65000,
                "loan_amount": 15000,
                "credit_score": 680,
                "employment_years": 5,
                "debt_to_income": 0.35,
                "num_credit_lines": 4,
                "payment_history": 0.9,
                "loan_to_value": 0.7,
                "num_late_payments": 1
            }
        }
    }

class PredictionResponse(BaseModel):
    """Output schema — what /predict returns."""
    prediction: int              # 0 = low risk, 1 = high risk
    probability_default: float   # probability of default
    risk_label: str              # "LOW RISK" or "HIGH RISK"
    model_version: str           # for tracking which model version answered