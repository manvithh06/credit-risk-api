# 🏦 Credit Risk API

A production-ready machine learning API that predicts credit default risk in real-time. Built with FastAPI, containerised with Docker, deployed on Render — with structured logging, data drift monitoring, and sub-200ms latency guarantees.

---

## 🔗 Live Demo

| Resource | URL |
|---|---|
| **Live API** | [https://credit-risk-api-xxxx.onrender.com](https://credit-risk-api-gjc9.onrender.com/) |


> ⚠️ Hosted on Render free tier — first request after inactivity may take ~30s (cold start). Subsequent requests respond in under 20ms.

---

## 📌 What This Project Does

Most ML projects live and die inside a Jupyter notebook. This project takes a trained credit risk classifier and wraps it into a **real production service** — the kind you'd actually deploy at a bank or fintech company.

Given 10 financial features about a loan applicant, the `/predict` endpoint returns:
- Whether the applicant is **HIGH RISK** or **LOW RISK**
- The **probability of default** (0.0 → 1.0)
- The model version that made the prediction

Everything is containerised, logged, monitored for drift, and load-tested.

---

## 🏗️ Architecture

```
Client Request
      │
      ▼
┌─────────────────┐
│   FastAPI App   │  ← validates input, logs request, measures latency
│   (main.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prediction     │  ← loads model once at startup, runs inference
│  Engine         │
│  (model.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Gradient        │  ← scikit-learn pipeline (StandardScaler + GBM)
│ Boosting Model  │    saved with joblib
│ (.joblib)       │
└─────────────────┘
```

---

## 📁 Project Structure

```
credit-risk-api/
├── app/
│   ├── main.py              ← FastAPI app, middleware, endpoints
│   ├── model.py             ← model loading + prediction logic
│   └── schemas.py           ← Pydantic request/response schemas
├── tools/
│   ├── load_test.py         ← p95 latency load tester (100 requests)
│   └── load_test_results.json ← results: p95 = 10ms ✅
├── model/
│   └── credit_model.joblib  ← trained Gradient Boosting pipeline
├── data/
│   └── reference.csv        ← training distribution for drift detection
├── drift_report.html        ← Evidently data drift report
├── train_model.py           ← script to train and save the model
├── generate_drift_report.py ← generates Evidently HTML report
├── Dockerfile               ← container definition
├── requirements.txt         ← pinned dependencies
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Run Locally with Python

```bash
# 1. Clone the repo
git clone https://github.com/manvithh06/credit-risk-api.git
cd credit-risk-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and save the model
python train_model.py

# 4. Generate drift report
python generate_drift_report.py

# 5. Start the API
uvicorn app.main:app --port 8000 --workers 2
```

API available at: http://localhost:8000  
Docs available at: http://localhost:8000/docs

### Option 2 — Run with Docker

```bash
# Build the image
docker build -t credit-risk-api .

# Run the container
docker run -p 8000:8000 credit-risk-api
```

Same endpoints, fully containerised.

---

## 📡 API Reference

### `GET /health`

Returns service health status.

```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "timestamp": "2026-06-01T10:00:00.000000"
}
```

### `POST /predict`

Predicts credit default risk for a loan applicant.

**Request:**
```bash
curl -X POST "https://credit-risk-api-xxxx.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "probability_default": 0.0823,
  "risk_label": "LOW RISK",
  "model_version": "1.0.0"
}
```

**Input Features:**

| Feature | Type | Description |
|---|---|---|
| `age` | float | Applicant age (18–100) |
| `income` | float | Annual income in USD |
| `loan_amount` | float | Requested loan amount |
| `credit_score` | float | Credit score (300–850) |
| `employment_years` | float | Years at current employer |
| `debt_to_income` | float | Debt-to-income ratio (0–1) |
| `num_credit_lines` | float | Number of open credit lines |
| `payment_history` | float | Payment history score |
| `loan_to_value` | float | Loan-to-value ratio |
| `num_late_payments` | float | Number of late payments |

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Library | scikit-learn 1.4.2 |
| Features | 10 financial features |
| Target | Binary (0 = no default, 1 = default) |
| Class balance | 85% low risk / 15% high risk |
| Saved with | joblib |
| Pipeline | StandardScaler → GradientBoostingClassifier |

The model is loaded **once at API startup** into memory and reused for every request — no per-request disk I/O, which is why latency is so low.

---

## ⚡ Performance

Load tested with `tools/load_test.py` — 100 requests against localhost:

| Metric | Result | Requirement |
|---|---|---|
| Min | 5.0ms | — |
| Mean | 6.8ms | — |
| Median | 6.4ms | — |
| **p95** | **10.0ms** | **< 200ms ✅** |
| p99 | 17.6ms | — |
| Max | 18.2ms | — |

p95 latency is **20× better** than the 200ms requirement.

---

## 📊 Data Drift Monitoring

`drift_report.html` (committed to repo root) is an Evidently report comparing:

- **Reference data** — the distribution the model was trained on (`data/reference.csv`, 1000 samples)
- **Production data** — simulated with realistic drift (income growth of 5–30%, slight credit score decline)

Open the HTML file in any browser to see:
- Which features have drifted
- Statistical test results per feature
- Dataset-level drift summary

**Key finding:** `income` and `credit_score` show the most significant drift — consistent with real-world economic changes over time.

---

## 🔒 Logging & Privacy

All requests are logged as structured JSON:

```json
{
  "timestamp": "2026-06-01T10:00:00",
  "level": "INFO",
  "message": "prediction",
  "prediction": 0,
  "probability_default": 0.0823,
  "risk_label": "LOW RISK",
  "latency_ms": 6.4,
  "request_hash": "a3f7b2c1"
}
```

**Raw PII is never logged.** Input values (income, credit score, loan amount) are never written to logs. Only prediction outcomes, latencies, and hashed request identifiers are recorded — compliant with data minimisation principles.

---

## 🐳 Docker Details

```dockerfile
FROM python:3.11-slim    # minimal base image
WORKDIR /app
COPY requirements.txt .
RUN pip install ...      # cached layer — only rebuilds when deps change
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

**Why Docker:** "Runs on my laptop" is not a deliverable. A Docker image runs identically on any machine, any cloud provider, any CI/CD system — no environment mismatch, no missing dependencies.

---

## 🧪 Running the Load Test

```bash
# Against local API
python tools/load_test.py --url http://localhost:8000

# Against deployed API
python tools/load_test.py --url https://credit-risk-api-xxxx.onrender.com

# Custom number of requests
python tools/load_test.py --url http://localhost:8000 --n 500
```

Results are saved to `tools/load_test_results.json`.

---

## 💡 Key Design Decisions

**Why Gradient Boosting over Logistic Regression?**  
Credit risk data is non-linear — the relationship between debt-to-income ratio and default probability isn't a straight line. GBM captures these interactions automatically.

**Why load the model at startup?**  
Loading a `.joblib` file takes ~100–500ms. Loading it per request would make every prediction 100× slower. Load once at startup, serve thousands of requests.

**Why Pydantic schemas?**  
FastAPI + Pydantic validates every incoming request automatically. If `credit_score` is missing or `age` is a string, the API returns a clear 422 error without any manual validation code.

**Why structured JSON logging?**  
Plain text logs (`"prediction made"`) can't be parsed by monitoring tools. JSON logs can be ingested by Datadog, Grafana, or CloudWatch to build dashboards, set alerts, and trace individual requests.

---

## 🔁 Reflection

The biggest mindset shift in this project was thinking about a model as a **service**, not a script.

A notebook runs once and produces a result. A service must handle concurrent requests gracefully, fail with clear error messages, log everything useful without logging anything sensitive, and run identically in development, staging, and production.

Docker solved the environment problem completely. Once the container ran locally, deployment to Render was just pushing to GitHub — no "it works on my machine" surprises.

The drift report was the most eye-opening part. Even simulating mild income growth (5–30%) triggered drift alerts on multiple features, showing exactly why monitoring is non-negotiable in production ML systems. A model that was accurate at deployment can silently degrade as the world changes around it.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **FastAPI** | REST API framework with auto-generated docs |
| **Pydantic** | Input validation and schema definition |
| **scikit-learn** | Model training (Gradient Boosting) |
| **joblib** | Model serialisation and loading |
| **Docker** | Containerisation for reproducible deployment |
| **Render** | Free cloud hosting for the Docker container |
| **Evidently** | Data drift detection and HTML reporting |
| **uvicorn** | ASGI server running FastAPI |

---

## 👨‍💻 Author

**Manvith Devadiga**  
AI/ML Engineering Student — NMAM Institute of Technology, Mangaluru  
[LinkedIn](https://linkedin.com/in/manvith-devadiga) · [GitHub](https://github.com/manvithh06)
