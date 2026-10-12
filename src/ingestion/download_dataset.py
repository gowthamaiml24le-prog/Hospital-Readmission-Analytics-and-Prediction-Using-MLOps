"""
Data Acquisition Module: Downloads and unpacks the UCI Diabetes 130-US Hospitals dataset.
Ensures reproducible, automated extraction of raw source files.
"""

import os
import sys
import logging
import zipfile
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import UCI_DIABETES_URL, RAW_DATA_DIR, RAW_ZIP_FILE, RAW_DATA_CSV, RAW_MAPPING_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def download_uci_dataset(force_download: bool = False) -> dict:
    """
    Downloads the official UCI Diabetes 130-US Hospitals dataset zip file
    and extracts diabetic_data.csv and IDS_mapping.csv into data/raw/.
    
    Returns:
        dict: Metadata with file paths, sizes, and row count.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Download zip if not already present
    if not RAW_ZIP_FILE.exists() or force_download:
        logger.info(f"Downloading dataset from {UCI_DIABETES_URL} ...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(UCI_DIABETES_URL, RAW_ZIP_FILE)
        logger.info(f"Downloaded zip file to {RAW_ZIP_FILE} (Size: {RAW_ZIP_FILE.stat().st_size:,} bytes)")
    else:
        logger.info(f"Raw zip archive already exists at {RAW_ZIP_FILE}")

    # 2. Extract contents
    logger.info("Extracting raw CSV files from archive...")
    with zipfile.ZipFile(RAW_ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
        extracted_files = zip_ref.namelist()
        logger.info(f"Extracted files: {extracted_files}")

    if not RAW_DATA_CSV.exists():
        raise FileNotFoundError(f"Expected file {RAW_DATA_CSV} was not found after extraction.")

    # 3. Gather initial file stats
    file_size_mb = RAW_DATA_CSV.stat().st_size / (1024 * 1024)
    logger.info(f"Raw data file ready: {RAW_DATA_CSV} ({file_size_mb:.2f} MB)")
    
    return {
        "status": "SUCCESS",
        "zip_path": str(RAW_ZIP_FILE),
        "data_csv": str(RAW_DATA_CSV),
        "mapping_csv": str(RAW_MAPPING_CSV),
        "size_mb": round(file_size_mb, 2)
    }

if __name__ == "__main__":
    result = download_uci_dataset()
    print("Download result:", result)
