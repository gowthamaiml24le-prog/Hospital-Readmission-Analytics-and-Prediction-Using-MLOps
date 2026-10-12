"""
Model Evaluation & Metrics Module:
Computes clinical evaluation metrics (Recall, Precision, F1, ROC-AUC, PR-AUC, Calibration/Brier score)
and generates diagnostic visualization curves.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

def compute_all_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes comprehensive classification and calibration metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_proba))

    # Also compute clinical sensitivity at high-recall operating point (threshold=0.20)
    y_pred_clin = (y_proba >= 0.20).astype(int)
    recall_at_20 = float(recall_score(y_true, y_pred_clin, zero_division=0))
    precision_at_20 = float(precision_score(y_true, y_pred_clin, zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "recall_at_thresh_0_20": round(recall_at_20, 4),
        "precision_at_thresh_0_20": round(precision_at_20, 4),
        "confusion_matrix": cm,
        "operating_threshold": threshold
    }

def generate_diagnostic_plots(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_dir: Path
) -> dict:
    """
    Generates and saves ROC curve, PR curve, Calibration curve, and Confusion Matrix plots.
    
    Returns:
        dict: File paths of generated diagnostic figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    # 1. ROC & PR Curve Combined Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[0].set_title(f"{model_name} - ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    axes[1].plot(rec, prec, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    baseline = np.mean(y_true)
    axes[1].axhline(y=baseline, color="grey", linestyle="--", label=f"Baseline ({baseline:.3f})")
    axes[1].set_title(f"{model_name} - Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    roc_pr_path = output_dir / f"{model_name.lower().replace(' ', '_')}_roc_pr.png"
    fig.savefig(roc_pr_path, dpi=200)
    plt.close(fig)
    figures["roc_pr_curve"] = str(roc_pr_path)

    # 2. Calibration Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_true, y_proba)
    ax.plot(prob_pred, prob_true, marker="o", color="#2ca02c", label=f"{model_name} (Brier: {brier:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect Calibration")
    ax.set_title(f"{model_name} - Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    calib_path = output_dir / f"{model_name.lower().replace(' ', '_')}_calibration.png"
    fig.savefig(calib_path, dpi=200)
    plt.close(fig)
    figures["calibration_curve"] = str(calib_path)

    return figures
