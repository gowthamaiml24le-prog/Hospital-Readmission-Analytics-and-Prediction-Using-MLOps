"""
ML Data Preparation Module:
Loads curated healthcare data, constructs pre-processing pipelines,
and executes stratified train/validation/test splitting.
"""

import sys
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import CURATED_DATA_DIR, MODELS_DIR, RANDOM_STATE, TARGET_COL

logger = logging.getLogger(__name__)

# Feature definitions for MLOps modeling
NUMERICAL_FEATURES = [
    "length_of_stay",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "prior_inpatient_admissions",
    "prior_outpatient_admissions",
    "prior_emergency_admissions",
    "total_prior_visits",
    "diagnosis_count",
    "active_diabetic_meds_count",
    "age_midpoint",
    "clinical_risk_index"
]

BINARY_FEATURES = [
    "has_prior_inpatient",
    "frequent_utilizer_flag",
    "extended_stay_flag",
    "high_comorbidity_flag",
    "comorbidity_cardio_diabetes",
    "medication_change_flag",
    "diabetes_med_flag",
    "on_insulin",
    "insulin_titration_up",
    "insulin_titration_down",
    "on_metformin",
    "a1c_tested",
    "a1c_high",
    "glucose_tested",
    "glucose_high",
    "is_elderly"
]

CATEGORICAL_FEATURES = [
    "gender",
    "admission_category",
    "discharge_category",
    "diag_1_category"
]

ALL_FEATURE_COLS = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

def prepare_train_val_test_data(
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> dict:
    """
    Loads curated data, applies stratified train/val/test split, fits
    preprocessor transformer, and returns transformed datasets with metadata.
    """
    logger.info("Loading curated dataset for MLOps pipeline...")
    mart_file = CURATED_DATA_DIR / "healthcare_readmission_mart.csv"
    if not mart_file.exists():
        raise FileNotFoundError(f"Curated file {mart_file} does not exist. Run ETL pipeline first.")

    df = pd.read_csv(mart_file, low_memory=False)
    logger.info(f"Loaded {len(df):,} encounters. Target column: {TARGET_COL}")

    if "is_elderly" not in df.columns:
        df["is_elderly"] = (df["age_midpoint"] >= 65).astype(int)

    X = df[ALL_FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int).values

    # Stratified Split 1: Split into Train (70%) and Temp (30%)
    temp_size = val_size + test_size
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Stratified Split 2: Split Temp into Validation (15%) and Test (15%)
    val_ratio_of_temp = val_size / temp_size
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp,
        test_size=(1.0 - val_ratio_of_temp),
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    logger.info(
        f"Data split sizes: Train={len(X_train_raw):,} (Pos: {y_train.mean()*100:.1f}%), "
        f"Val={len(X_val_raw):,} (Pos: {y_val.mean()*100:.1f}%), "
        f"Test={len(X_test_raw):,} (Pos: {y_test.mean()*100:.1f}%)"
    )

    # Build Column Transformer Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    # Fit preprocessor strictly on Training data to prevent data leakage
    preprocessor.fit(X_train_raw)

    # Transform feature matrices
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERICAL_FEATURES + BINARY_FEATURES + cat_feature_names

    # Persist Preprocessor and Metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved fitted preprocessor to {preprocessor_path}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "preprocessor_path": str(preprocessor_path)
    }

if __name__ == "__main__":
    prepared = prepare_train_val_test_data()
    print("Prepared feature count:", len(prepared["feature_names"]))
