"""
Tests for Data Ingestion Module.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_CSV, RAW_MAPPING_CSV, STAGING_DATA_DIR
from src.ingestion.download_dataset import download_uci_dataset
from src.ingestion.ingest_raw import ingest_raw_data

def test_raw_files_exist():
    """Verify that raw UCI files exist after download."""
    assert RAW_DATA_CSV.exists(), "diabetic_data.csv must exist in data/raw"
    assert RAW_MAPPING_CSV.exists(), "IDS_mapping.csv must exist in data/raw"

def test_ingest_raw_execution():
    """Verify that raw ingestion generates the staging snapshot and audit log."""
    result = ingest_raw_data()
    assert result["status"] == "SUCCESS"
    assert result["row_count"] > 90000

    stg_file = STAGING_DATA_DIR / "stg_raw_encounters.csv"
    assert stg_file.exists()
    
    audit_file = STAGING_DATA_DIR / "ingestion_audit_log.csv"
    assert audit_file.exists()
