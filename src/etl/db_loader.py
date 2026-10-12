"""
Database Loader Module:
Loads dimensional data mart tables (patient_summary, admission, diagnosis, readmission)
into the relational database (PostgreSQL or local SQLite/DuckDB) and exports curated CSV artifacts.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import CURATED_DATA_DIR, get_db_engine

logger = logging.getLogger(__name__)

def build_and_load_data_mart(df_engineered: pd.DataFrame) -> dict:
    """
    Constructs dimensional tables and loads them into the database and curated folder.
    
    Tables created:
    1. patient_summary: Patient-level aggregations and demographics
    2. admission: Encounter-level admission and discharge characteristics
    3. diagnosis: Diagnostic categories and comorbidity flags
    4. readmission: Comprehensive analytical fact mart
    
    Returns:
        dict: Table row counts and database load status.
    """
    logger.info("Constructing relational data mart tables...")
    CURATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_db_engine()

    # 1. Patient Summary Table (Aggregated per patient)
    logger.info("Building patient_summary table...")
    patient_summary = df_engineered.groupby("surrogate_patient_id").agg(
        race=("race", "first"),
        gender=("gender", "first"),
        age_group=("age", "first"),
        total_encounters=("surrogate_encounter_id", "count"),
        avg_length_of_stay=("length_of_stay", "mean"),
        total_prior_inpatient=("prior_inpatient_admissions", "max"),
        total_prior_emergency=("prior_emergency_admissions", "max"),
        has_cardio_diabetes=("comorbidity_cardio_diabetes", "max"),
        has_readmission_30d=("readmitted_30d", "max")
    ).reset_index()
    patient_summary["avg_length_of_stay"] = patient_summary["avg_length_of_stay"].round(2)
    patient_summary["high_utilizer_flag"] = (patient_summary["total_encounters"] >= 2).astype(int)

    # 2. Admission Dimension Table
    logger.info("Building admission table...")
    admission_cols = [
        "surrogate_encounter_id", "surrogate_patient_id",
        "admission_type_id", "admission_type_desc", "admission_category",
        "admission_source_id", "admission_source_desc",
        "discharge_disposition_id", "discharge_disposition_desc", "discharge_category",
        "length_of_stay", "extended_stay_flag"
    ]
    admission = df_engineered[admission_cols].copy()

    # 3. Diagnosis Dimension Table
    logger.info("Building diagnosis table...")
    diagnosis_cols = [
        "surrogate_encounter_id", "surrogate_patient_id",
        "diag_1", "diag_1_category",
        "diag_2", "diag_2_category",
        "diag_3", "diag_3_category",
        "diagnosis_count", "high_comorbidity_flag",
        "comorbidity_cardio_diabetes"
    ]
    diagnosis = df_engineered[diagnosis_cols].copy()

    # 4. Fact Readmission Analytics Mart
    logger.info("Building readmission analytics fact table...")
    readmission_cols = [
        "surrogate_encounter_id", "surrogate_patient_id",
        "gender", "age", "age_midpoint", "race",
        "admission_category", "discharge_category",
        "length_of_stay", "extended_stay_flag",
        "num_lab_procedures", "num_procedures", "num_medications",
        "prior_inpatient_admissions", "prior_outpatient_admissions", "prior_emergency_admissions",
        "total_prior_visits", "has_prior_inpatient", "frequent_utilizer_flag",
        "diag_1_category", "diagnosis_count", "high_comorbidity_flag", "comorbidity_cardio_diabetes",
        "active_diabetic_meds_count", "medication_change_flag", "diabetes_med_flag",
        "on_insulin", "insulin_titration_up", "insulin_titration_down", "on_metformin",
        "a1c_tested", "a1c_high", "glucose_tested", "glucose_high",
        "clinical_risk_index", "is_elderly",
        "readmission_category", "readmitted_30d"
    ]
    readmission_mart = df_engineered[readmission_cols].copy()

    # Save to Curated CSV files for portable access
    patient_summary.to_csv(CURATED_DATA_DIR / "patient_summary.csv", index=False)
    admission.to_csv(CURATED_DATA_DIR / "admission.csv", index=False)
    diagnosis.to_csv(CURATED_DATA_DIR / "diagnosis.csv", index=False)
    readmission_mart.to_csv(CURATED_DATA_DIR / "healthcare_readmission_mart.csv", index=False)
    logger.info("Saved all curated tables to CSV in data/curated/")

    # Load into Relational Database via SQLAlchemy
    loaded_counts = {}
    try:
        with engine.begin() as conn:
            logger.info("Loading tables into relational database...")
            patient_summary.to_sql("patient_summary", con=conn, if_exists="replace", index=False)
            admission.to_sql("admission", con=conn, if_exists="replace", index=False)
            diagnosis.to_sql("diagnosis", con=conn, if_exists="replace", index=False)
            readmission_mart.to_sql("fact_readmission", con=conn, if_exists="replace", index=False)
            # Create primary analytical view or mart table
            readmission_mart.to_sql("healthcare_mart", con=conn, if_exists="replace", index=False)
            
        loaded_counts = {
            "patient_summary": len(patient_summary),
            "admission": len(admission),
            "diagnosis": len(diagnosis),
            "fact_readmission": len(readmission_mart),
            "database_engine": str(engine.url)
        }
        logger.info(f"Successfully loaded tables into database: {loaded_counts}")
    except Exception as e:
        logger.error(f"Failed to load into database: {e}")
        loaded_counts = {"error": str(e)}

    return {
        "status": "SUCCESS",
        "tables": loaded_counts,
        "csv_artifacts": [
            str(CURATED_DATA_DIR / "patient_summary.csv"),
            str(CURATED_DATA_DIR / "admission.csv"),
            str(CURATED_DATA_DIR / "diagnosis.csv"),
            str(CURATED_DATA_DIR / "healthcare_readmission_mart.csv")
        ]
    }
