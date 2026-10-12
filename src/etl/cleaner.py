"""
Data Cleaning & Anonymization Module.
Identifies and cleans missing value codes, removes invalid/expired encounters with audit logging,
standardizes clinical values, and generates patient-safe surrogate keys.
"""

import sys
import hashlib
import logging
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import REJECTED_DATA_DIR, STAGING_DATA_DIR
from src.etl.id_mapper import load_id_mappings, EXPIRED_OR_HOSPICE_DISCHARGE_IDS

logger = logging.getLogger(__name__)

SALT_PATIENT = "HEALTHCARE_PATIENT_SALT_V1_"
SALT_ENCOUNTER = "HEALTHCARE_ENCOUNTER_SALT_V1_"

def generate_surrogate_key(value: any, salt: str) -> str:
    """Generates a secure, deterministic, irreversible surrogate key for de-identification."""
    text = f"{salt}{str(value)}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:16]

def clean_and_standardize_encounters(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans raw hospital encounters, standardizes missing values, filters invalid/expired records,
    and replaces direct identifiers with surrogate keys.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (cleaned_df, rejected_records_df)
    """
    initial_count = len(df)
    logger.info(f"Starting data cleaning pipeline on {initial_count:,} raw encounter records.")
    
    # 1. Replace special missing code '?' with NaN
    df_cleaned = df.copy()
    df_cleaned.replace("?", np.nan, inplace=True)
    
    # Track rejected records
    rejected_list = []
    
    # 2. Filter Invalid Gender (e.g. Unknown/Invalid)
    invalid_gender_mask = df_cleaned["gender"].isin(["Unknown/Invalid", np.nan])
    invalid_genders = df_cleaned[invalid_gender_mask]
    for _, row in invalid_genders.iterrows():
        rejected_list.append({
            "raw_encounter_id": row["encounter_id"],
            "raw_patient_nbr": row["patient_nbr"],
            "rejection_reason": f"Invalid gender value: {row['gender']}",
            "discharge_disposition_id": row["discharge_disposition_id"],
            "rejected_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
    df_cleaned = df_cleaned[~invalid_gender_mask].copy()

    # 3. Filter Ineligible Encounters: Expired or Hospice Discharges
    # Patients who died or entered hospice care cannot be readmitted
    expired_mask = df_cleaned["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE_DISCHARGE_IDS)
    expired_encounters = df_cleaned[expired_mask]
    for _, row in expired_encounters.iterrows():
        rejected_list.append({
            "raw_encounter_id": row["encounter_id"],
            "raw_patient_nbr": row["patient_nbr"],
            "rejection_reason": f"Ineligible for readmission: Expired/Hospice (Discharge ID {row['discharge_disposition_id']})",
            "discharge_disposition_id": row["discharge_disposition_id"],
            "rejected_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
    df_cleaned = df_cleaned[~expired_mask].copy()

    # 4. Check & Remove duplicate encounter IDs if any
    duplicate_mask = df_cleaned.duplicated(subset=["encounter_id"], keep="first")
    if duplicate_mask.any():
        for _, row in df_cleaned[duplicate_mask].iterrows():
            rejected_list.append({
                "raw_encounter_id": row["encounter_id"],
                "raw_patient_nbr": row["patient_nbr"],
                "rejection_reason": "Duplicate encounter record",
                "discharge_disposition_id": row["discharge_disposition_id"],
                "rejected_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })
        df_cleaned = df_cleaned[~duplicate_mask].copy()

    # 5. Load and Apply ID Mappings (Admission, Discharge, Source)
    adm_types, disch_disp, adm_src = load_id_mappings()
    df_cleaned["admission_type_desc"] = df_cleaned["admission_type_id"].map(adm_types).fillna("Unknown")
    df_cleaned["discharge_disposition_desc"] = df_cleaned["discharge_disposition_id"].map(disch_disp).fillna("Unknown")
    df_cleaned["admission_source_desc"] = df_cleaned["admission_source_id"].map(adm_src).fillna("Unknown")

    # High-level Admission Category (Emergency, Urgent, Elective, Other)
    df_cleaned["admission_category"] = df_cleaned["admission_type_desc"].apply(
        lambda x: "Emergency" if "emergency" in str(x).lower()
        else ("Urgent" if "urgent" in str(x).lower()
        else ("Elective" if "elective" in str(x).lower()
        else "Other"))
    )

    # High-level Discharge Category
    df_cleaned["discharge_category"] = df_cleaned["discharge_disposition_desc"].apply(
        lambda x: "Discharged to Home" if "home" in str(x).lower() and "hospice" not in str(x).lower()
        else ("Transferred / Skilled Facility" if any(w in str(x).lower() for w in ["hospital", "snf", "icf", "facility", "institution", "rehab"])
        else ("Left AMA" if "ama" in str(x).lower()
        else "Other"))
    )

    # 6. Standardize missing categorical columns
    df_cleaned["race"] = df_cleaned["race"].fillna("Unknown")
    df_cleaned["payer_code"] = df_cleaned["payer_code"].fillna("Missing/Unknown")
    df_cleaned["medical_specialty"] = df_cleaned["medical_specialty"].fillna("Missing/Unknown")
    
    # Weight column indicator (>97% missing in UCI dataset)
    df_cleaned["weight_recorded_flag"] = (~df_cleaned["weight"].isna()).astype(int)

    # 7. Generate patient-safe surrogate keys & drop raw IDs
    df_cleaned["surrogate_patient_id"] = df_cleaned["patient_nbr"].apply(
        lambda x: generate_surrogate_key(x, SALT_PATIENT)
    )
    df_cleaned["surrogate_encounter_id"] = df_cleaned["encounter_id"].apply(
        lambda x: generate_surrogate_key(x, SALT_ENCOUNTER)
    )

    # Prepare rejected records dataframe
    rejected_df = pd.DataFrame(rejected_list)
    if not rejected_df.empty:
        REJECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        rejected_path = REJECTED_DATA_DIR / "rejected_records_log.csv"
        rejected_df.to_csv(rejected_path, index=False)
        logger.info(f"Logged {len(rejected_df):,} rejected records to {rejected_path}")

    logger.info(
        f"Cleaning complete: {len(df_cleaned):,} valid records retained "
        f"({len(rejected_df):,} rejected, {len(df_cleaned)/initial_count*100:.2f}% retention)."
    )
    return df_cleaned, rejected_df
