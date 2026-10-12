"""
FastAPI Model Serving Application:
Provides low-latency RESTful inference, health checking, explainability,
and clinical decision support for hospital readmission prediction.
"""

import os
import sys
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, REPORTS_DIR
from src.mlops.prepare_data import ALL_FEATURE_COLS
from src.mlops.explainability import explain_single_patient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Readmission_API")

app = FastAPI(
    title="Hospital Readmission Prediction & Decision Support API",
    description="Operationalized MLOps inference service for predicting 30-day hospital readmission risk.",
    version="1.0.0"
)

# Global model and preprocessor cache
model = None
preprocessor = None
metadata = {}
START_TIME = time.time()

def load_artifacts():
    global model, preprocessor, metadata
    model_path = MODELS_DIR / "best_model.joblib"
    prep_path = MODELS_DIR / "preprocessor.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"

    if model_path.exists() and prep_path.exists():
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        logger.info(f"Loaded champion model and preprocessor from {MODELS_DIR}")
    else:
        logger.warning("Model artifacts not yet found in artifacts/models/")

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

@app.on_event("startup")
def startup_event():
    load_artifacts()

# Pydantic Schemas
class PatientEncounterInput(BaseModel):
    # Demographics
    gender: str = Field(default="Female", description="Patient gender (Female or Male)")
    age_midpoint: int = Field(default=65, ge=5, le=95, description="Approximate age midpoint in years")
    is_elderly: int = Field(default=1, ge=0, le=1, description="1 if age >= 65, else 0")
    
    # Stay & Utilization
    length_of_stay: int = Field(default=4, ge=1, le=14, description="Hospital stay duration in days")
    num_lab_procedures: int = Field(default=45, ge=1, le=150, description="Total lab procedures performed")
    num_procedures: int = Field(default=1, ge=0, le=10, description="Total non-lab surgical/diagnostic procedures")
    num_medications: int = Field(default=15, ge=1, le=100, description="Number of distinct medications administered")
    diagnosis_count: int = Field(default=7, ge=1, le=16, description="Number of diagnoses recorded")
    clinical_risk_index: int = Field(default=4, ge=0, le=12, description="Heuristic clinical risk index")

    prior_inpatient_admissions: int = Field(default=1, ge=0, description="Number of prior inpatient admissions in last year")
    prior_outpatient_admissions: int = Field(default=0, ge=0, description="Number of prior outpatient visits")
    prior_emergency_admissions: int = Field(default=0, ge=0, description="Number of prior emergency room visits")
    total_prior_visits: int = Field(default=1, ge=0, description="Total prior healthcare interactions")

    has_prior_inpatient: int = Field(default=1, ge=0, le=1)
    frequent_utilizer_flag: int = Field(default=0, ge=0, le=1)
    extended_stay_flag: int = Field(default=0, ge=0, le=1)
    high_comorbidity_flag: int = Field(default=0, ge=0, le=1)
    comorbidity_cardio_diabetes: int = Field(default=1, ge=0, le=1)

    # Regimen & Glycemic Controls
    active_diabetic_meds_count: int = Field(default=2, ge=0, le=10)
    medication_change_flag: int = Field(default=1, ge=0, le=1, description="1 if diabetic medication dosage was adjusted")
    diabetes_med_flag: int = Field(default=1, ge=0, le=1)
    on_insulin: int = Field(default=1, ge=0, le=1)
    insulin_titration_up: int = Field(default=0, ge=0, le=1)
    insulin_titration_down: int = Field(default=0, ge=0, le=1)
    on_metformin: int = Field(default=1, ge=0, le=1)
    
    a1c_tested: int = Field(default=1, ge=0, le=1)
    a1c_high: int = Field(default=1, ge=0, le=1, description="1 if HbA1c > 8%")
    glucose_tested: int = Field(default=0, ge=0, le=1)
    glucose_high: int = Field(default=0, ge=0, le=1)

    # Categories
    admission_category: str = Field(default="Emergency", description="Emergency, Urgent, Elective, Other")
    discharge_category: str = Field(default="Discharged to Home", description="Discharged to Home, Transferred / Skilled Facility, Left AMA, Other")
    diag_1_category: str = Field(default="Circulatory", description="Circulatory, Diabetes, Respiratory, Digestive, etc.")

class PredictionResponse(BaseModel):
    readmission_risk_probability: float
    risk_percentage: str
    risk_tier: str
    predicted_class_default: int
    predicted_class_clinical_threshold: int
    clinical_recommendation: str
    model_version: str

@app.get("/", tags=["General"])
def root():
    return {
        "service": "Hospital Readmission Prediction & Decision Support API",
        "status": "active",
        "documentation": "/docs",
        "endpoints": ["/health", "/predict", "/explain", "/metrics"]
    }

@app.get("/health", tags=["Health"])
def health_check():
    global model
    if model is None:
        load_artifacts()
    uptime_sec = round(time.time() - START_TIME, 2)
    return {
        "status": "HEALTHY" if model is not None else "DEGRADED",
        "uptime_seconds": uptime_sec,
        "model_loaded": model is not None,
        "champion_model": metadata.get("champion_model_name", "Unknown"),
        "version": metadata.get("model_version", "1.0.0")
    }

@app.get("/metrics", tags=["Model Performance"])
def get_metrics():
    if not metadata:
        raise HTTPException(status_code=404, detail="Model metadata not loaded.")
    return {
        "model_version": metadata.get("model_version"),
        "champion_model_name": metadata.get("champion_model_name"),
        "metrics_summary": metadata.get("metrics_summary"),
        "operating_recommendation": metadata.get("operating_recommendation")
    }

def _build_input_dataframe(payload: PatientEncounterInput) -> pd.DataFrame:
    data_dict = payload.model_dump()
    df = pd.DataFrame([data_dict])
    # Ensure all required features are present
    for col in ALL_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[ALL_FEATURE_COLS]

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_readmission(payload: PatientEncounterInput):
    if model is None or preprocessor is None:
        load_artifacts()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not initialized. Please train the model first.")

    input_df = _build_input_dataframe(payload)
    X_trans = preprocessor.transform(input_df)
    proba = float(model.predict_proba(X_trans)[0, 1])

    # Categorize Risk Tier & Clinical Guidance
    if proba < 0.20:
        tier = "Low Risk"
        recommendation = "Standard discharge protocols. Routine follow-up scheduled with primary care within 14-30 days."
    elif 0.20 <= proba < 0.45:
        tier = "Moderate Risk"
        recommendation = "Enhanced discharge planning recommended: Pharmacist medication reconciliation and telehealth check-in within 7 days."
    else:
        tier = "High Risk"
        recommendation = "URGENT INTERVENTION: Post-discharge care management coordinator assigned, home-health check, and follow-up appointment within 48-72 hours."

    return PredictionResponse(
        readmission_risk_probability=round(proba, 4),
        risk_percentage=f"{proba * 100:.1f}%",
        risk_tier=tier,
        predicted_class_default=int(proba >= 0.50),
        predicted_class_clinical_threshold=int(proba >= 0.20),
        clinical_recommendation=recommendation,
        model_version=metadata.get("model_version", "1.0.0")
    )

@app.post("/explain", tags=["Explainability"])
def explain_readmission(payload: PatientEncounterInput):
    if model is None or preprocessor is None:
        load_artifacts()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not initialized.")

    input_df = _build_input_dataframe(payload)
    X_trans = preprocessor.transform(input_df)
    factors = explain_single_patient(X_trans, top_n=6)
    
    proba = float(model.predict_proba(X_trans)[0, 1])
    return {
        "readmission_probability": round(proba, 4),
        "top_contributing_risk_factors": factors
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
