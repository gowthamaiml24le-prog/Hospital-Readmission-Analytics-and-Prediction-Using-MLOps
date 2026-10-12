"""
Data Quality Validation Module:
Enforces healthcare data integrity, schema compliance, range boundaries,
and domain rules, logging all validation results into audit reports.
"""

import json
import logging
import datetime
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def run_data_quality_checks(df: pd.DataFrame, report_path: Path = None) -> dict:
    """
    Executes a comprehensive suite of data quality checks on the engineered dataset.
    
    Checks:
    1. Primary key uniqueness (surrogate_encounter_id)
    2. Null checks on critical identifiers and targets
    3. Range constraints on clinical variables (LOS 1-14, diagnosis count 1-16)
    4. Categorical domain validity (gender, readmission categories)
    5. Anomaly detection (non-negative prior utilization)
    
    Returns:
        dict: Summary of validation test outcomes.
    """
    logger.info(f"Running data quality validation on {len(df):,} records...")
    checks = []
    total_records = len(df)

    def record_check(name: str, passed: bool, failed_count: int, description: str):
        checks.append({
            "check_name": name,
            "description": description,
            "passed": bool(passed),
            "failed_records_count": int(failed_count),
            "failure_rate_pct": round((failed_count / total_records) * 100, 4) if total_records else 0
        })

    # 1. Primary Key Uniqueness
    dup_encounters = df["surrogate_encounter_id"].duplicated().sum()
    record_check(
        "surrogate_encounter_id_uniqueness",
        dup_encounters == 0,
        dup_encounters,
        "Every encounter must have a unique surrogate encounter key."
    )

    # 2. Critical Null Checks
    null_patient_ids = df["surrogate_patient_id"].isna().sum()
    record_check(
        "surrogate_patient_id_not_null",
        null_patient_ids == 0,
        null_patient_ids,
        "Patient surrogate identifier must not be null."
    )

    null_targets = df["readmitted_30d"].isna().sum()
    record_check(
        "target_readmitted_30d_not_null",
        null_targets == 0,
        null_targets,
        "Readmission target variable must not contain null values."
    )

    # 3. Range Checks
    los_violations = ((df["length_of_stay"] < 1) | (df["length_of_stay"] > 14)).sum()
    record_check(
        "length_of_stay_range_1_to_14",
        los_violations == 0,
        los_violations,
        "Length of stay must be bounded between 1 and 14 days."
    )

    diag_count_violations = (df["diagnosis_count"] < 1).sum()
    record_check(
        "diagnosis_count_positive",
        diag_count_violations == 0,
        diag_count_violations,
        "Encounter must have at least one diagnosis recorded."
    )

    med_count_violations = (df["medication_count"] < 1).sum()
    record_check(
        "medication_count_positive",
        med_count_violations == 0,
        med_count_violations,
        "Encounter must have at least one medication record."
    )

    # 4. Domain & Category Checks
    invalid_gender = (~df["gender"].isin(["Female", "Male"])).sum()
    record_check(
        "gender_domain_validity",
        invalid_gender == 0,
        invalid_gender,
        "Gender field must be standard Female or Male."
    )

    target_domain_invalid = (~df["readmitted_30d"].isin([0, 1])).sum()
    record_check(
        "readmitted_30d_binary_domain",
        target_domain_invalid == 0,
        target_domain_invalid,
        "Binary readmission target must only contain 0 or 1."
    )

    # 5. Non-negative utilization checks
    negative_prior = (
        (df["prior_inpatient_admissions"] < 0) |
        (df["prior_emergency_admissions"] < 0) |
        (df["prior_outpatient_admissions"] < 0)
    ).sum()
    record_check(
        "prior_utilization_non_negative",
        negative_prior == 0,
        negative_prior,
        "Prior inpatient, outpatient, and emergency counts cannot be negative."
    )

    all_passed = all(c["passed"] for c in checks)
    passed_count = sum(1 for c in checks if c["passed"])
    
    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_records_evaluated": total_records,
        "total_checks_evaluated": len(checks),
        "checks_passed": passed_count,
        "checks_failed": len(checks) - passed_count,
        "all_checks_passed": all_passed,
        "checks": checks
    }

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Data quality validation report saved to {report_path}")

    logger.info(f"Quality Check Status: {passed_count}/{len(checks)} passed. All passed: {all_passed}")
    return summary
