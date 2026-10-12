"""
Tests for ETL, Data Cleaning & Feature Engineering Modules.
"""

import pytest
import numpy as np
import pandas as pd
from src.etl.icd9_mapper import map_icd9_code
from src.etl.cleaner import clean_and_standardize_encounters, generate_surrogate_key
from src.etl.feature_engineering import engineer_features
from src.etl.validators import run_data_quality_checks

def test_icd9_mapping():
    """Verify correct clinical classification of ICD-9 codes."""
    assert map_icd9_code("250.02") == "Diabetes"
    assert map_icd9_code("428") == "Circulatory"
    assert map_icd9_code("486") == "Respiratory"
    assert map_icd9_code("530.81") == "Digestive"
    assert map_icd9_code("584.9") == "Genitourinary"
    assert map_icd9_code("715.90") == "Musculoskeletal"
    assert map_icd9_code("199.1") == "Neoplasms"
    assert map_icd9_code("820") == "Injury"
    assert map_icd9_code("?") == "Missing/Unknown"
    assert map_icd9_code(None) == "Missing/Unknown"

def test_surrogate_key_generation():
    """Verify surrogate keys are deterministic, non-empty, and irreversible."""
    k1 = generate_surrogate_key(12345, "TEST_SALT_")
    k2 = generate_surrogate_key(12345, "TEST_SALT_")
    k3 = generate_surrogate_key(67890, "TEST_SALT_")
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 16

def test_cleaning_and_filtering():
    """Verify filtering of expired cases and invalid gender."""
    sample_df = pd.DataFrame({
        "encounter_id": [1, 2, 3, 4],
        "patient_nbr": [101, 102, 103, 104],
        "race": ["Caucasian", "?", "AfricanAmerican", "Caucasian"],
        "gender": ["Female", "Male", "Unknown/Invalid", "Male"],
        "age": ["[50-60)", "[60-70)", "[70-80)", "[80-90)"],
        "weight": ["?", "?", "?", "?"],
        "admission_type_id": [1, 2, 1, 3],
        "discharge_disposition_id": [1, 11, 1, 1],  # 11 = Expired
        "admission_source_id": [7, 7, 7, 1],
        "time_in_hospital": [3, 5, 2, 4],
        "payer_code": ["?", "MC", "?", "BC"],
        "medical_specialty": ["?", "Cardiology", "?", "?"],
        "num_lab_procedures": [40, 50, 30, 20],
        "num_procedures": [1, 2, 0, 1],
        "num_medications": [10, 15, 8, 12],
        "number_outpatient": [0, 0, 0, 0],
        "number_emergency": [0, 1, 0, 0],
        "number_inpatient": [0, 2, 0, 1],
        "diag_1": ["428", "250", "486", "715"],
        "diag_2": ["250", "428", "?", "401"],
        "diag_3": ["?", "?", "?", "?"],
        "number_diagnoses": [5, 7, 4, 6],
        "max_glu_serum": ["None", "None", "None", "None"],
        "A1Cresult": [">8", "None", "Norm", "None"],
        "metformin": ["Steady", "No", "No", "Steady"],
        "insulin": ["No", "Steady", "Up", "No"],
        "change": ["Ch", "No", "Ch", "No"],
        "diabetesMed": ["Yes", "Yes", "Yes", "Yes"],
        "readmitted": ["<30", "NO", ">30", "NO"]
    })
    
    # Add dummy remaining medication columns
    for m in ["repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
              "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
              "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
              "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
              "metformin-rosiglitazone", "metformin-pioglitazone"]:
        sample_df[m] = "No"

    cleaned_df, rejected_df = clean_and_standardize_encounters(sample_df)
    
    # Encounter 2 (Expired id=11) and Encounter 3 (Unknown/Invalid gender) must be rejected
    assert len(cleaned_df) == 2
    assert len(rejected_df) == 2
    assert "surrogate_patient_id" in cleaned_df.columns
    assert "surrogate_encounter_id" in cleaned_df.columns
    assert cleaned_df["race"].iloc[0] == "Caucasian"

def test_feature_engineering_and_dq():
    """Verify engineered metrics and quality check execution."""
    sample_clean = pd.DataFrame({
        "surrogate_encounter_id": ["enc_1", "enc_2"],
        "surrogate_patient_id": ["pat_1", "pat_2"],
        "gender": ["Female", "Male"],
        "age": ["[60-70)", "[70-80)"],
        "time_in_hospital": [4, 8],
        "number_inpatient": [1, 0],
        "number_outpatient": [1, 2],
        "number_emergency": [0, 1],
        "number_diagnoses": [6, 9],
        "num_procedures": [1, 2],
        "num_lab_procedures": [45, 55],
        "num_medications": [12, 18],
        "change": ["Ch", "No"],
        "diabetesMed": ["Yes", "Yes"],
        "insulin": ["Up", "No"],
        "metformin": ["Steady", "No"],
        "A1Cresult": [">8", "None"],
        "max_glu_serum": ["None", "None"],
        "diag_1": ["428", "250.01"],
        "diag_2": ["250", "401"],
        "diag_3": ["486", "715"],
        "readmitted": ["<30", "NO"],
        "admission_type_desc": ["Emergency", "Elective"],
        "discharge_disposition_desc": ["Discharged to home", "Transferred"],
        "admission_source_desc": ["Emergency Room", "Physician Referral"],
        "admission_category": ["Emergency", "Elective"],
        "discharge_category": ["Discharged to Home", "Transferred / Skilled Facility"]
    })
    for m in ["repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
              "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
              "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
              "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
              "metformin-rosiglitazone", "metformin-pioglitazone"]:
        sample_clean[m] = "No"

    engineered = engineer_features(sample_clean)
    assert "length_of_stay" in engineered.columns
    assert "total_prior_visits" in engineered.columns
    assert engineered["total_prior_visits"].iloc[0] == 2
    assert engineered["total_prior_visits"].iloc[1] == 3
    assert engineered["readmitted_30d"].iloc[0] == 1
    assert engineered["readmitted_30d"].iloc[1] == 0

    dq = run_data_quality_checks(engineered)
    assert dq["all_checks_passed"] is True
