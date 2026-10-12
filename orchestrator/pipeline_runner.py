"""
Pipeline Runner Orchestrator:
Executes the full end-to-end healthcare data engineering pipeline.
Emulates the Airflow DAG stages with comprehensive execution logging and audit tracking.
"""

import sys
import time
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import STAGING_DATA_DIR, REPORTS_DIR, REJECTED_DATA_DIR
from src.ingestion.ingest_raw import ingest_raw_data
from src.etl.cleaner import clean_and_standardize_encounters
from src.etl.feature_engineering import engineer_features
from src.etl.validators import run_data_quality_checks
from src.etl.db_loader import build_and_load_data_mart

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ETL_Orchestrator")

def run_full_pipeline() -> dict:
    """
    Executes the complete Part 1 ETL workflow:
    Ingestion -> Cleaning & Anonymization -> Feature Engineering -> Data Quality Validation -> Data Mart Load.
    """
    total_start = time.time()
    logger.info("==================================================================")
    logger.info("STARTING HEALTHCARE READMISSION ETL PIPELINE EXECUTION")
    logger.info("==================================================================")

    # Stage 1: Raw Data Ingestion
    logger.info(">>> [Stage 1/5] Ingesting Raw UCI Hospital Data...")
    ingest_result = ingest_raw_data()
    logger.info(f"Stage 1 Complete: {ingest_result['row_count']:,} raw records ingested.")

    # Stage 2: Data Cleaning & Surrogate Key Generation
    logger.info(">>> [Stage 2/5] Cleaning, Filtering Ineligible Encounters & Anonymizing...")
    stg_raw_file = STAGING_DATA_DIR / "stg_raw_encounters.csv"
    raw_df = pd.read_csv(stg_raw_file, low_memory=False)
    cleaned_df, rejected_df = clean_and_standardize_encounters(raw_df)
    
    stg_cleaned_file = STAGING_DATA_DIR / "stg_cleaned_encounters.csv"
    cleaned_df.to_csv(stg_cleaned_file, index=False)
    logger.info(f"Stage 2 Complete: {len(cleaned_df):,} valid records, {len(rejected_df):,} rejected records.")

    # Stage 3: Clinical Feature Engineering
    logger.info(">>> [Stage 3/5] Engineering Clinical, Diagnostic, and Utilization Features...")
    engineered_df = engineer_features(cleaned_df)
    stg_eng_file = STAGING_DATA_DIR / "stg_engineered_encounters.csv"
    engineered_df.to_csv(stg_eng_file, index=False)
    logger.info(f"Stage 3 Complete: Engineered dataset shape: {engineered_df.shape}")

    # Stage 4: Data Quality Checks & Error Auditing
    logger.info(">>> [Stage 4/5] Executing Data Quality Integrity Checks...")
    report_file = REPORTS_DIR / "data_quality_report.json"
    dq_summary = run_data_quality_checks(engineered_df, report_path=report_file)
    logger.info(f"Stage 4 Complete: Passed {dq_summary['checks_passed']}/{dq_summary['total_checks_evaluated']} checks.")

    # Stage 5: Relational Data Mart & Warehouse Load
    logger.info(">>> [Stage 5/5] Building & Loading Relational Data Mart Tables...")
    mart_result = build_and_load_data_mart(engineered_df)
    logger.info("Stage 5 Complete: Data Mart successfully populated.")

    total_duration = round(time.time() - total_start, 2)
    logger.info("==================================================================")
    logger.info(f"PIPELINE RUN COMPLETED SUCCESSFULLY IN {total_duration}s")
    logger.info("==================================================================")

    return {
        "status": "SUCCESS",
        "total_duration_seconds": total_duration,
        "raw_records": ingest_result["row_count"],
        "cleaned_records": len(cleaned_df),
        "rejected_records": len(rejected_df),
        "data_quality_passed": dq_summary["all_checks_passed"],
        "data_mart_tables": mart_result["tables"]
    }

if __name__ == "__main__":
    result = run_full_pipeline()
    print("\nPipeline Summary:")
    for k, v in result.items():
        print(f"  {k}: {v}")
