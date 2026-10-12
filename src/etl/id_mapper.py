"""
ID Mapper Module:
Parses IDS_mapping.csv to translate numeric codes for admission types,
discharge dispositions, and admission sources into human-readable clinical descriptions.
"""

import sys
import logging
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_MAPPING_CSV

logger = logging.getLogger(__name__)

# Fallback dictionaries in case mapping file has anomalies
DEFAULT_ADMISSION_TYPES = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped"
}

DEFAULT_DISCHARGE_DISPOSITIONS = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to Medicare approved swing bed",
    16: "Discharged/transferred/referred another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "NULL",
    19: "Expired at home. Medicaid only, hospice.",
    20: "Expired in a medical facility. Medicaid only, hospice.",
    21: "Expired, place unknown. Medicaid only, hospice.",
    22: "Discharged/transferred to another rehab fac including rehab units of a hospital",
    23: "Discharged/transferred to a long term care hospital.",
    24: "Discharged/transferred to a nursing facility certified under Medicaid",
    25: "Not Mapped",
    26: "Unknown/Invalid"
}

DEFAULT_ADMISSION_SOURCES = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from a hospital",
    5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another health care facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from critical access hospital",
    11: "Normal Delivery",
    12: "Premature Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    15: "Not Available",
    17: "NULL",
    20: "Not Mapped"
}

# Discharge IDs representing expired or hospice care (ineligible for readmission)
EXPIRED_OR_HOSPICE_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

def load_id_mappings(mapping_csv_path: Path = RAW_MAPPING_CSV) -> tuple[dict, dict, dict]:
    """
    Parses IDS_mapping.csv and returns dictionaries for:
    (admission_type_map, discharge_disposition_map, admission_source_map)
    """
    adm_types = dict(DEFAULT_ADMISSION_TYPES)
    disch_disp = dict(DEFAULT_DISCHARGE_DISPOSITIONS)
    adm_src = dict(DEFAULT_ADMISSION_SOURCES)

    if not mapping_csv_path.exists():
        logger.warning(f"Mapping CSV not found at {mapping_csv_path}. Using built-in standard dictionaries.")
        return adm_types, disch_disp, adm_src

    try:
        with open(mapping_csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_section = None
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            lower_line = line.lower()
            if "admission_type_id" in lower_line:
                current_section = "admission_type"
                continue
            elif "discharge_disposition_id" in lower_line:
                current_section = "discharge_disposition"
                continue
            elif "admission_source_id" in lower_line:
                current_section = "admission_source"
                continue

            parts = [p.strip().strip('"') for p in line.split(",", 1)]
            if len(parts) == 2 and parts[0].isdigit():
                code_id = int(parts[0])
                desc = parts[1]
                if current_section == "admission_type":
                    adm_types[code_id] = desc
                elif current_section == "discharge_disposition":
                    disch_disp[code_id] = desc
                elif current_section == "admission_source":
                    adm_src[code_id] = desc

        logger.info(f"Loaded mappings: {len(adm_types)} admission types, {len(disch_disp)} discharge types, {len(adm_src)} admission sources.")
    except Exception as e:
        logger.warning(f"Error parsing {mapping_csv_path}: {e}. Retaining fallback mappings.")

    return adm_types, disch_disp, adm_src
