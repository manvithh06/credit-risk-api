"""
generate_drift_report.py — Generate an Evidently drift report.

Compares reference data (training distribution) with simulated
production data (slightly shifted) to detect feature drift.

WHY drift matters: if the income distribution of loan applicants
shifts significantly from training data, model accuracy degrades
silently — the model still runs but gives wrong answers.
"""

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Load reference data (training distribution)
reference = pd.read_csv('data/reference.csv')

# Simulate production data with slight drift
# WHY: in real systems, production data comes from your API logs
# Here we simulate drift to show the report works
np.random.seed(123)
production = reference.copy()

# Introduce realistic drift in key features
# Income increases (economic growth), credit scores drop slightly
production['income'] = production['income'] * np.random.uniform(1.05, 1.3, len(production))
production['credit_score'] = production['credit_score'] * np.random.uniform(0.9, 0.95, len(production))
production['num_late_payments'] = production['num_late_payments'] + np.random.poisson(0.5, len(production))

# Save production data
production.to_csv('data/production.csv', index=False)
print("Production data saved: data/production.csv")

# Generate Evidently report
# DataDriftPreset checks every feature for statistical distribution shift
# DataQualityPreset checks for missing values, duplicates, outliers
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
])

report.run(
    reference_data=reference.drop(columns=['default']),
    current_data=production.drop(columns=['default'])
)

report.save_html('drift_report.html')
print("✅ Drift report saved: drift_report.html")
print("Open it in a browser to see which features have drifted.")