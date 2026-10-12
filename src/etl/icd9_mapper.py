"""
ICD-9 Diagnosis Mapper Module:
Standardizes alphanumeric ICD-9 codes into Clinical Classifications Software (CCS)
diagnostic categories and comorbidity indices.
"""

import pandas as pd

def map_icd9_code(code: str) -> str:
    """
    Maps an individual ICD-9 code string to a standardized clinical diagnostic category.
    
    Categories:
    - Diabetes (250.xx)
    - Circulatory (390-459, 785)
    - Respiratory (460-519, 786)
    - Digestive (520-579, 787)
    - Genitourinary (580-629, 788)
    - Musculoskeletal (710-739)
    - Neoplasms (140-239)
    - Injury (800-999)
    - Endocrine/Metabolic Other (240-249, 251-279)
    - Other/Supplemental (V/E codes, other signs/symptoms)
    - Missing/Unknown (? or empty)
    """
    if pd.isna(code):
        return "Missing/Unknown"
    
    code_str = str(code).strip()
    if code_str in ["?", "", "None", "nan"]:
        return "Missing/Unknown"
    
    # Check for Diabetes first (250.xx)
    if code_str.startswith("250"):
        return "Diabetes"
    
    # Check supplementary classifications (E-codes external cause, V-codes supplementary factors)
    if code_str.startswith(("V", "v", "E", "e")):
        return "Other/Supplemental"
    
    # Attempt numeric range parsing
    try:
        val = float(code_str)
        if (390 <= val <= 459) or val == 785:
            return "Circulatory"
        elif (460 <= val <= 519) or val == 786:
            return "Respiratory"
        elif (520 <= val <= 579) or val == 787:
            return "Digestive"
        elif (580 <= val <= 629) or val == 788:
            return "Genitourinary"
        elif 710 <= val <= 739:
            return "Musculoskeletal"
        elif 140 <= val <= 239:
            return "Neoplasms"
        elif 800 <= val <= 999:
            return "Injury"
        elif (240 <= val <= 279):
            return "Endocrine/Metabolic Other"
        else:
            return "Other"
    except ValueError:
        return "Other"

def apply_diagnosis_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies ICD-9 categorization across primary, secondary, and tertiary diagnosis codes
    and flags key comorbidity interactions.
    """
    df["diag_1_category"] = df["diag_1"].apply(map_icd9_code)
    df["diag_2_category"] = df["diag_2"].apply(map_icd9_code)
    df["diag_3_category"] = df["diag_3"].apply(map_icd9_code)
    
    # Comorbidity Interaction: Diabetes + Circulatory condition present
    circulatory_in_any = (
        (df["diag_1_category"] == "Circulatory") |
        (df["diag_2_category"] == "Circulatory") |
        (df["diag_3_category"] == "Circulatory")
    )
    diabetes_in_any = (
        (df["diag_1_category"] == "Diabetes") |
        (df["diag_2_category"] == "Diabetes") |
        (df["diag_3_category"] == "Diabetes")
    )
    
    df["comorbidity_cardio_diabetes"] = (circulatory_in_any & diabetes_in_any).astype(int)
    return df
