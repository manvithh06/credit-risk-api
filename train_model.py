"""
train_model.py — Train a credit risk classifier and save it.
Uses the built-in sklearn dataset as a proxy for credit risk data.
Run once: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# Generate synthetic credit risk dataset
# WHY synthetic: no credentials needed, perfectly sized, 
# and we control the feature names to make them meaningful
np.random.seed(42)
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.85, 0.15],  # 85% low risk, 15% high risk (realistic imbalance)
    random_state=42
)

# Give features meaningful credit-risk names
feature_names = [
    'age', 'income', 'loan_amount', 'credit_score',
    'employment_years', 'debt_to_income', 'num_credit_lines',
    'payment_history', 'loan_to_value', 'num_late_payments'
]

df = pd.DataFrame(X, columns=feature_names)
df['default'] = y  # 1 = default (high risk), 0 = no default

# Save reference data for drift detection (first 1000 rows = "training distribution")
os.makedirs('data', exist_ok=True)
df.head(1000).to_csv('data/reference.csv', index=False)
print(f"Reference data saved: data/reference.csv ({len(df.head(1000))} rows)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df['default'],
    test_size=0.2, random_state=42, stratify=df['default']
)

# Build pipeline — scaler + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# Evaluate
score = pipeline.score(X_test, y_test)
print(f"Model accuracy: {score:.4f}")

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/credit_model.joblib')
print("Model saved: model/credit_model.joblib")
print(f"File size: {os.path.getsize('model/credit_model.joblib') / 1024:.1f} KB")