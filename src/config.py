"""
Configuration module for the Hospital Readmission Analytics & MLOps project.
Handles environment variables, paths, database connection strings, and clinical constants.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
STAGING_DATA_DIR = DATA_DIR / "staging"
CURATED_DATA_DIR = DATA_DIR / "curated"
ANALYTICS_DATA_DIR = DATA_DIR / "analytics"
REJECTED_DATA_DIR = DATA_DIR / "rejected"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, STAGING_DATA_DIR, CURATED_DATA_DIR, ANALYTICS_DATA_DIR, REJECTED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset URLs & Filenames
UCI_DIABETES_URL = os.getenv(
    "DATA_SOURCE_URL",
    "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
RAW_ZIP_FILE = RAW_DATA_DIR / "diabetes_dataset.zip"
RAW_DATA_CSV = RAW_DATA_DIR / "diabetic_data.csv"
RAW_MAPPING_CSV = RAW_DATA_DIR / "IDS_mapping.csv"

# Database Configuration
DEFAULT_SQLITE_PATH = ANALYTICS_DATA_DIR / "healthcare_mart.db"
DB_TYPE = os.getenv("DB_TYPE", "sqlite")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "healthcare_db")

if DB_TYPE.lower() == "postgresql":
    DB_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DB_CONNECTION_STRING = os.getenv(
        "DB_CONNECTION_STRING",
        f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
    )

def get_db_engine():
    """
    Creates and returns a SQLAlchemy database engine.
    Tries configured connection; gracefully falls back to SQLite if PostgreSQL is unreachable.
    """
    try:
        engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)
        # Test connection
        with engine.connect() as conn:
            pass
        return engine
    except Exception as e:
        fallback_str = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
        print(f"[Config Warning] Primary DB connection failed ({e}). Falling back to local SQLite at {fallback_str}")
        return create_engine(fallback_str)

# MLflow Configuration
MLFLOW_DB_PATH = ANALYTICS_DATA_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "hospital-readmission-prediction")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Clinical Constants: 23 Antidiabetic medications present in UCI dataset
DIABETIC_MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
]

# Random State for Reproducibility
RANDOM_STATE = 42
TARGET_COL = "readmitted_30d"
