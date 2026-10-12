"""
Feature Engineering Module:
Engineers clinical, demographic, utilization, and medication variables
for hospital readmission analytics and machine learning modeling.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import DIABETIC_MEDICATIONS
from src.etl.icd9_mapper import apply_diagnosis_mapping

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all clinical, utilization, and diagnostic engineered features.
    
    Parameters:
        df (pd.DataFrame): Cleaned encounters dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with rich engineered features and target variables.
    """
    logger.info("Computing engineered healthcare features...")
    df_feat = df.copy()

    # 1. Length of Stay (LOS) variable
    df_feat["length_of_stay"] = df_feat["time_in_hospital"].astype(int)
    df_feat["extended_stay_flag"] = (df_feat["length_of_stay"] >= 7).astype(int)

    # 2. Healthcare Utilization (Prior Admissions)
    df_feat["prior_inpatient_admissions"] = df_feat["number_inpatient"].astype(int)
    df_feat["prior_outpatient_admissions"] = df_feat["number_outpatient"].astype(int)
    df_feat["prior_emergency_admissions"] = df_feat["number_emergency"].astype(int)
    
    df_feat["total_prior_visits"] = (
        df_feat["prior_inpatient_admissions"] +
        df_feat["prior_outpatient_admissions"] +
        df_feat["prior_emergency_admissions"]
    )
    df_feat["has_prior_inpatient"] = (df_feat["prior_inpatient_admissions"] > 0).astype(int)
    df_feat["frequent_utilizer_flag"] = (df_feat["total_prior_visits"] >= 3).astype(int)

    # 3. Clinical Intensity & Diagnostic Count
    df_feat["diagnosis_count"] = df_feat["number_diagnoses"].astype(int)
    df_feat["high_comorbidity_flag"] = (df_feat["diagnosis_count"] >= 9).astype(int)
    df_feat["num_procedures"] = df_feat["num_procedures"].astype(int)
    df_feat["num_lab_procedures"] = df_feat["num_lab_procedures"].astype(int)

    # 4. Medication Count and Regimen Complexity
    df_feat["medication_count"] = df_feat["num_medications"].astype(int)
    df_feat["polypharmacy_flag"] = (df_feat["medication_count"] >= 16).astype(int)

    # Active Diabetic Medications Count (count of meds with value 'Steady', 'Up', 'Down')
    existing_med_cols = [m for m in DIABETIC_MEDICATIONS if m in df_feat.columns]
    
    def count_active_meds(row):
        return sum(1 for m in existing_med_cols if row[m] in ["Steady", "Up", "Down"])

    df_feat["active_diabetic_meds_count"] = df_feat.apply(count_active_meds, axis=1)
    
    # Medication Changes (Medication Titration)
    df_feat["medication_change_flag"] = (df_feat["change"] == "Ch").astype(int)
    df_feat["diabetes_med_flag"] = (df_feat["diabetesMed"] == "Yes").astype(int)

    # Specific key medications
    df_feat["on_insulin"] = (df_feat["insulin"] != "No").astype(int)
    df_feat["insulin_titration_up"] = (df_feat["insulin"] == "Up").astype(int)
    df_feat["insulin_titration_down"] = (df_feat["insulin"] == "Down").astype(int)
    df_feat["on_metformin"] = (df_feat["metformin"] != "No").astype(int)

    # 5. Glycemic & Laboratory Testing Flags
    df_feat["a1c_tested"] = (df_feat["A1Cresult"].notna() & (df_feat["A1Cresult"] != "None")).astype(int)
    df_feat["a1c_high"] = df_feat["A1Cresult"].isin([">7", ">8"]).astype(int)
    df_feat["glucose_tested"] = (df_feat["max_glu_serum"].notna() & (df_feat["max_glu_serum"] != "None")).astype(int)
    df_feat["glucose_high"] = df_feat["max_glu_serum"].isin([">200", ">300"]).astype(int)

    # 6. Apply ICD-9 Categorization
    df_feat = apply_diagnosis_mapping(df_feat)

    # Specific Chronic Organ Disease Indicators
    df_feat["primary_diag_circulatory"] = (df_feat["diag_1_category"] == "Circulatory").astype(int)
    df_feat["primary_diag_diabetes"] = (df_feat["diag_1_category"] == "Diabetes").astype(int)
    df_feat["primary_diag_respiratory"] = (df_feat["diag_1_category"] == "Respiratory").astype(int)
    df_feat["primary_diag_digestive"] = (df_feat["diag_1_category"] == "Digestive").astype(int)

    # 7. Composite Healthcare Risk Score (Heuristic clinical triage index: 0-10)
    df_feat["clinical_risk_index"] = (
        (df_feat["prior_inpatient_admissions"] >= 1).astype(int) * 3 +
        df_feat["extended_stay_flag"] * 2 +
        df_feat["high_comorbidity_flag"] * 2 +
        df_feat["polypharmacy_flag"] * 1 +
        df_feat["comorbidity_cardio_diabetes"] * 2
    )

    # 8. Define Readmission Target Variables
    # Clean readmitted field
    df_feat["readmission_category"] = df_feat["readmitted"].astype(str)
    # Binary 30-day readmission flag (Primary MLOps target)
    df_feat["readmitted_30d"] = (df_feat["readmission_category"] == "<30").astype(int)

    # Clean Age intervals into ordinal numeric & clean label
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }
    df_feat["age_midpoint"] = df_feat["age"].map(age_map).fillna(55).astype(int)
    df_feat["is_elderly"] = (df_feat["age_midpoint"] >= 65).astype(int)

    logger.info(
        f"Feature engineering complete. Total features: {df_feat.shape[1]}. "
        f"30-Day Readmission prevalence: {df_feat['readmitted_30d'].mean()*100:.2f}%"
    )
    return df_feat
