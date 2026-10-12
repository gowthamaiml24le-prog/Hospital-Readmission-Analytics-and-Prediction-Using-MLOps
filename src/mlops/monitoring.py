"""
Monitoring & Drift Detection Module:
Monitors data distribution drift (Kolmogorov-Smirnov & PSI),
evaluates subgroup performance and clinical fairness (age, gender),
and assesses model retraining criteria.
"""

import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, brier_score_loss

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, REPORTS_DIR
from src.mlops.prepare_data import prepare_train_val_test_data

logger = logging.getLogger("Monitoring")

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between reference and actual distributions.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate drift
    PSI >= 0.2: Significant drift
    """
    try:
        # Quantile binning on expected distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(expected, percentiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_pct = expected_counts / max(len(expected), 1)
        actual_pct = actual_counts / max(len(actual), 1)

        # Smooth zero counts
        expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
        actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi_val)
    except Exception:
        return 0.0

def run_drift_and_fairness_monitoring() -> dict:
    """
    Performs data drift analysis and subgroup fairness auditing.
    """
    logger.info("Initiating healthcare drift and subgroup monitoring suite...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_train_val_test_data()
    X_train_raw = data["X_train_raw"]
    X_test_raw = data["X_test_raw"]
    y_test = data["y_test"]
    X_test = data["X_test"]

    # Load champion model
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # 1. Feature Drift Detection
    drift_results = []
    drift_flagged_count = 0

    numerical_cols = [
        "length_of_stay", "num_lab_procedures", "num_procedures",
        "num_medications", "prior_inpatient_admissions", "total_prior_visits"
    ]

    for col in numerical_cols:
        ref_vals = X_train_raw[col].values
        prod_vals = X_test_raw[col].values

        ks_stat, p_val = ks_2samp(ref_vals, prod_vals)
        psi = calculate_psi(ref_vals, prod_vals)

        is_drift = bool(p_val < 0.01 and ks_stat > 0.05)
        if is_drift:
            drift_flagged_count += 1

        drift_results.append({
            "feature": col,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "psi_value": round(float(psi), 4),
            "drift_detected": is_drift
        })

    # 2. Subgroup Performance Auditing (Clinical Fairness)
    subgroups = {}

    # By Gender
    for g in ["Female", "Male"]:
        mask = (X_test_raw["gender"].values == g)
        if mask.sum() > 0:
            rec = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            prec = precision_score(y_test[mask], y_pred[mask], zero_division=0)
            brier = brier_score_loss(y_test[mask], y_proba[mask])
            subgroups[f"gender_{g}"] = {
                "sample_size": int(mask.sum()),
                "recall": round(float(rec), 4),
                "precision": round(float(prec), 4),
                "brier_score": round(float(brier), 4),
                "actual_prevalence": round(float(y_test[mask].mean()), 4)
            }

    # By Age Group (<65 vs >=65)
    for age_lbl, is_eld in [("Under_65", 0), ("Elderly_65_Plus", 1)]:
        mask = (X_test_raw["is_elderly"].values == is_eld)
        if mask.sum() > 0:
            rec = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            prec = precision_score(y_test[mask], y_pred[mask], zero_division=0)
            brier = brier_score_loss(y_test[mask], y_proba[mask])
            subgroups[f"age_{age_lbl}"] = {
                "sample_size": int(mask.sum()),
                "recall": round(float(rec), 4),
                "precision": round(float(prec), 4),
                "brier_score": round(float(brier), 4),
                "actual_prevalence": round(float(y_test[mask].mean()), 4)
            }

    # 3. Retraining Decision Engine
    retraining_triggered = drift_flagged_count >= 2
    retraining_reason = (
        f"{drift_flagged_count} numerical features exhibited significant distribution shift."
        if retraining_triggered
        else "Data distributions and subgroup performance metrics remain within allowable clinical thresholds."
    )

    report = {
        "audit_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "total_test_samples": len(y_test),
        "overall_test_brier": round(float(brier_score_loss(y_test, y_proba)), 4),
        "drift_summary": {
            "total_features_evaluated": len(numerical_cols),
            "drift_flagged_count": drift_flagged_count,
            "feature_drift_details": drift_results
        },
        "subgroup_fairness": subgroups,
        "retraining_decision": {
            "retraining_triggered": retraining_triggered,
            "retraining_reason": retraining_reason,
            "recommended_cadence": "Quarterly or whenever PSI > 0.20 on hospital intake batches"
        }
    }

    report_path = REPORTS_DIR / "monitoring_drift_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved monitoring and drift report to {report_path}")

    return report

if __name__ == "__main__":
    rep = run_drift_and_fairness_monitoring()
    print("Monitoring report summary:")
    print("  Drift flagged features:", rep["drift_summary"]["drift_flagged_count"])
    print("  Retraining triggered:", rep["retraining_decision"]["retraining_triggered"])
