"""
Model Training & MLflow Experiment Tracking Module:
Trains and compares Logistic Regression, Random Forest, and XGBoost models.
Tracks hyperparameters, metrics, and diagnostic artifact plots via MLflow.
Saves and registers the champion model for deployment.
"""

import sys
import json
import logging
import joblib
import mlflow
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE
)
from src.mlops.prepare_data import prepare_train_val_test_data
from src.mlops.evaluate import compute_all_metrics, generate_diagnostic_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLOps_Trainer")

def train_and_evaluate_models() -> dict:
    """
    Executes the training and benchmarking pipeline across Logistic Regression,
    Random Forest, and XGBoost with MLflow logging.
    """
    logger.info("Initializing MLflow tracking...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 1. Prepare data splits
    logger.info("Preparing stratified training, validation, and test datasets...")
    data = prepare_train_val_test_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = data["feature_names"]

    # Calculate class imbalance ratio for XGBoost scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = round(neg_count / max(pos_count, 1), 2)
    logger.info(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}")

    # Model candidate definitions
    candidates = [
        {
            "name": "Logistic_Regression",
            "model": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="lbfgs"
            ),
            "params": {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs"}
        },
        {
            "name": "Random_Forest",
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 10,
                "class_weight": "balanced"
            }
        },
        {
            "name": "XGBoost",
            "model": XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 120,
                "max_depth": 5,
                "learning_rate": 0.08,
                "scale_pos_weight": scale_pos_weight
            }
        }
    ]

    results = []
    trained_instances = {}

    for cand in candidates:
        model_name = cand["name"]
        model = cand["model"]
        params = cand["params"]

        logger.info(f"\n==================== Training {model_name} ====================")
        with mlflow.start_run(run_name=model_name):
            # Log params
            mlflow.log_params(params)
            mlflow.log_param("num_features", len(feature_names))
            mlflow.log_param("train_samples", len(y_train))

            # Train model
            model.fit(X_train, y_train)
            trained_instances[model_name] = model

            # Predict probabilities
            val_proba = model.predict_proba(X_val)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]

            # Compute metrics
            val_metrics = compute_all_metrics(y_val, val_proba)
            test_metrics = compute_all_metrics(y_test, test_proba)

            # Log metrics to MLflow
            for metric_k, metric_v in val_metrics.items():
                if isinstance(metric_v, (int, float)):
                    mlflow.log_metric(f"val_{metric_k}", metric_v)
            for metric_k, metric_v in test_metrics.items():
                if isinstance(metric_v, (int, float)):
                    mlflow.log_metric(f"test_{metric_k}", metric_v)

            # Generate and log diagnostic figures
            plots = generate_diagnostic_plots(model_name, y_test, test_proba, REPORTS_DIR)
            for plot_k, plot_path in plots.items():
                mlflow.log_artifact(plot_path)

            logger.info(
                f"[{model_name}] Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                f"Test ROC-AUC: {test_metrics['roc_auc']:.4f} | "
                f"Test Recall: {test_metrics['recall']:.4f} | "
                f"Test Brier: {test_metrics['brier_score']:.4f}"
            )

            results.append({
                "model_name": model_name,
                "val_roc_auc": val_metrics["roc_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_recall": test_metrics["recall"],
                "test_precision": test_metrics["precision"],
                "test_f1": test_metrics["f1"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_brier": test_metrics["brier_score"],
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "params": params
            })

    # Champion Model Selection (Based on best validation ROC-AUC & PR-AUC)
    results_sorted = sorted(results, key=lambda x: (x["val_roc_auc"], x["test_pr_auc"]), reverse=True)
    best_candidate = results_sorted[0]
    best_name = best_candidate["model_name"]
    best_model = trained_instances[best_name]

    logger.info(f"\nChampion Model Selected: {best_name} with Val ROC-AUC = {best_candidate['val_roc_auc']:.4f}")

    # Persist Champion Model & Registration Metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    logger.info(f"Saved Champion model artifact to {best_model_path}")

    # Model Registry Metadata
    metadata = {
        "model_version": "v1.0.0",
        "champion_model_name": best_name,
        "artifact_path": str(best_model_path),
        "preprocessor_path": data["preprocessor_path"],
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "metrics_summary": {r["model_name"]: {
            "test_roc_auc": r["test_roc_auc"],
            "test_recall": r["test_recall"],
            "test_f1": r["test_f1"],
            "test_pr_auc": r["test_pr_auc"],
            "test_brier": r["test_brier"]
        } for r in results},
        "operating_recommendation": {
            "default_threshold": 0.50,
            "clinical_high_sensitivity_threshold": 0.20,
            "intended_use": "Decision-support triage for inpatient discharge planning"
        }
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model registry metadata to {metadata_path}")

    return {
        "champion_model": best_name,
        "leaderboard": results_sorted,
        "metadata_path": str(metadata_path)
    }

if __name__ == "__main__":
    summary = train_and_evaluate_models()
    print("\nModel Training Leaderboard:")
    for m in summary["leaderboard"]:
        print(f"  {m['model_name']}: Val ROC-AUC={m['val_roc_auc']}, Test ROC-AUC={m['test_roc_auc']}, Recall={m['test_recall']}")
