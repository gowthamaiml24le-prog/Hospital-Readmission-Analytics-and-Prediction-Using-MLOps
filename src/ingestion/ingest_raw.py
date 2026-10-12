"""
Raw Data Ingestion Module.
Maintains a raw copy of data, captures extraction metadata, and loads the initial staging layer.
"""

import os
import sys
import time
import logging
import datetime
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RAW_DATA_CSV,
    RAW_MAPPING_CSV,
    STAGING_DATA_DIR,
    UCI_DIABETES_URL,
    get_db_engine
)
from src.ingestion.download_dataset import download_uci_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def ingest_raw_data() -> dict:
    """
    Ingests the raw UCI diabetes dataset into the staging directory
    and logs metadata to an audit trail.
    
    Returns:
        dict: Ingestion summary metrics.
    """
    start_time = time.time()
    STAGING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ensure source files exist
    if not RAW_DATA_CSV.exists() or not RAW_MAPPING_CSV.exists():
        logger.info("Raw files not detected. Initiating download...")
        download_uci_dataset()

    logger.info(f"Reading raw data from {RAW_DATA_CSV} ...")
    raw_df = pd.read_csv(RAW_DATA_CSV, low_memory=False)
    row_count, col_count = raw_df.shape
    logger.info(f"Loaded raw dataset with {row_count:,} rows and {col_count} columns.")

    # Save to staging layer
    staging_file = STAGING_DATA_DIR / "stg_raw_encounters.csv"
    raw_df.to_csv(staging_file, index=False)
    logger.info(f"Saved raw staging snapshot to {staging_file}")

    duration = round(time.time() - start_time, 2)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    audit_entry = {
        "execution_id": f"INGEST_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "extraction_timestamp": timestamp,
        "source": UCI_DIABETES_URL,
        "raw_file": str(RAW_DATA_CSV),
        "row_count": row_count,
        "column_count": col_count,
        "duration_seconds": duration,
        "status": "SUCCESS"
    }

    # Persist audit entry to staging audit log CSV
    audit_df = pd.DataFrame([audit_entry])
    audit_log_path = STAGING_DATA_DIR / "ingestion_audit_log.csv"
    if audit_log_path.exists():
        audit_df.to_csv(audit_log_path, mode="a", header=False, index=False)
    else:
        audit_df.to_csv(audit_log_path, mode="w", header=True, index=False)

    # Attempt logging to database
    try:
        engine = get_db_engine()
        audit_df.to_sql("ingestion_audit_log", con=engine, if_exists="append", index=False)
        logger.info("Successfully recorded ingestion audit entry to database.")
    except Exception as e:
        logger.warning(f"Could not persist audit log to DB (persisted to CSV): {e}")

    logger.info(f"Ingestion completed in {duration}s. Audit ID: {audit_entry['execution_id']}")
    return audit_entry

if __name__ == "__main__":
    result = ingest_raw_data()
    print("Ingestion result:", result)
