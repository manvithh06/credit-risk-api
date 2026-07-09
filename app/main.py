"""
main.py — FastAPI application with structured logging.

Endpoints:
  GET  /         → health check
  GET  /health   → detailed health check
  POST /predict  → credit risk prediction
  GET  /docs     → auto-generated API docs (free from FastAPI)
"""

import time
import json
import logging
import hashlib
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas import PredictionRequest, PredictionResponse
from app.model import predict, MODEL_VERSION

# ── Structured JSON logging ────────────────────────────────
# WHY JSON logging: machine-parseable, works with log aggregators
# Regular logging: "Prediction made for user 12345" → hard to parse
# JSON logging: {"event": "prediction", "user_id_hash": "a3f...", "latency_ms": 45}

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, 'extra'):
            log_obj.update(record.extra)
        return json.dumps(log_obj)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("credit_api")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ── FastAPI app ────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk API",
    description="Predict credit default risk using a Gradient Boosting model.",
    version=MODEL_VERSION
)

# ── Middleware: log every request ──────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing — no PII in logs."""
    start = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 2)
    
    logger.info("request", extra={
        "extra": {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms
        }
    })
    return response

# ── Endpoints ──────────────────────────────────────────────
@app.get("/")
def root():
    """Root health check."""
    return {"status": "ok", "service": "credit-risk-api", "version": MODEL_VERSION}

@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    """
    Credit risk prediction endpoint.
    
    Input: 10 financial features
    Output: prediction (0/1), probability, risk label
    
    PII handling: we never log raw input values.
    WHY: loan amounts and credit scores are sensitive financial data.
    We only log the prediction outcome and latency.
    """
    start = time.time()
    
    # Convert Pydantic model to dict for prediction
    features = request.model_dump()
    
    # Run prediction
    result = predict(features)
    
    latency_ms = round((time.time() - start) * 1000, 2)
    
    # Structured log — NO raw PII
    # WHY hash credit_score: if logs are leaked, raw scores aren't exposed
    # We hash for correlation purposes only
    logger.info("prediction", extra={
        "extra": {
            "prediction": result['prediction'],
            "probability_default": result['probability_default'],
            "risk_label": result['risk_label'],
            "latency_ms": latency_ms,
            # Hash a stable identifier — not storing raw values
            "request_hash": hashlib.sha256(
                str(sorted(features.items())).encode()
            ).hexdigest()[:8]
        }
    })
    
    return PredictionResponse(**result)