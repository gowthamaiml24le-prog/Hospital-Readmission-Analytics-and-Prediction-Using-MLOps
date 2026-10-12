"""
Explainability & Interpretability Module:
Generates global feature importance and SHAP (SHapley Additive exPlanations) summary plots,
as well as patient-level local risk explanations for clinical decision support.
"""

import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, REPORTS_DIR
from src.mlops.prepare_data import prepare_train_val_test_data

logger = logging.getLogger("Explainability")

def generate_global_explainability(max_samples: int = 1000) -> dict:
    """
    Computes SHAP values on a representative validation subsample,
    generates summary plots, and computes feature importance rankings.
    """
    logger.info("Generating global model explainability with SHAP...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load champion model and metadata
    model_path = MODELS_DIR / "best_model.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Trained model or metadata not found. Run train.py first.")

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]

    # Load validation sample
    data = prepare_train_val_test_data()
    X_val = data["X_val"]
    
    # Subsample for SHAP computational efficiency
    sample_size = min(max_samples, len(X_val))
    X_sample = X_val[:sample_size]

    logger.info(f"Computing TreeExplainer SHAP values on {sample_size} validation encounters...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification with XGBoost, shap_values may be 2D array of log-odds
    if isinstance(shap_values, list):
        shap_vals_matrix = shap_values[1]  # positive class
    else:
        shap_vals_matrix = shap_values

    # 1. Feature Importance Bar Chart
    mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:15]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = mean_abs_shap[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importance[::-1], align="center", color="#1f77b4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Mean |SHAP Value| (Impact on 30-Day Readmission Risk)")
    ax.set_title("Top 15 Clinical Drivers of Hospital Readmission (SHAP)")
    plt.tight_layout()

    bar_path = REPORTS_DIR / "shap_feature_importance.png"
    fig.savefig(bar_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved SHAP feature importance plot to {bar_path}")

    # 2. SHAP Summary Beeswarm Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_vals_matrix,
        X_sample,
        feature_names=feature_names,
        max_display=15,
        show=False
    )
    plt.tight_layout()
    summary_path = REPORTS_DIR / "shap_summary_beeswarm.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary beeswarm plot to {summary_path}")

    # Top features ranking dict
    rankings = [{"feature": feat, "mean_abs_shap": round(float(imp), 4)} for feat, imp in zip(top_features, top_importance)]

    # Save rankings json
    rankings_path = REPORTS_DIR / "shap_feature_rankings.json"
    with open(rankings_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2)

    return {
        "status": "SUCCESS",
        "top_features": rankings[:10],
        "shap_bar_plot": str(bar_path),
        "shap_beeswarm_plot": str(summary_path)
    }

def explain_single_patient(patient_vector: np.ndarray, top_n: int = 5) -> list:
    """
    Computes local SHAP explanation for a single patient encounter,
    returning top positive and negative contributing factors.
    """
    model_path = MODELS_DIR / "best_model.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(patient_vector)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_row = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
    feature_names = meta["feature_names"]

    # Rank by impact
    sorted_idx = np.argsort(np.abs(shap_row))[::-1][:top_n]
    factors = []
    for idx in sorted_idx:
        val = shap_row[idx]
        direction = "Increases Risk" if val > 0 else "Decreases Risk"
        factors.append({
            "feature": feature_names[idx],
            "shap_value": round(float(val), 4),
            "effect": direction
        })
    return factors

if __name__ == "__main__":
    result = generate_global_explainability()
    print("SHAP explainability completed. Top drivers:", [f["feature"] for f in result["top_features"][:5]])
