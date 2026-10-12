"""
Apache Airflow DAG: Hospital Readmission ETL Pipeline
Orchestrates raw data ingestion, clinical standardization, feature engineering,
data quality validation, and loading into the PostgreSQL/relational healthcare data mart.
"""

from datetime import datetime, timedelta
import sys
from pathlib import Path

# Insert project root into sys.path to enable importing pipeline modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Attempt Airflow import; fallback gracefully for local syntax checking
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

default_args = {
    "owner": "healthcare_mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

def airflow_step_ingest():
    from src.ingestion.ingest_raw import ingest_raw_data
    return ingest_raw_data()

def airflow_step_clean():
    import pandas as pd
    from src.config import STAGING_DATA_DIR
    from src.etl.cleaner import clean_and_standardize_encounters
    
    stg_file = STAGING_DATA_DIR / "stg_raw_encounters.csv"
    raw_df = pd.read_csv(stg_file, low_memory=False)
    cleaned_df, rejected_df = clean_and_standardize_encounters(raw_df)
    
    cleaned_path = STAGING_DATA_DIR / "stg_cleaned_encounters.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    return {"cleaned_count": len(cleaned_df), "rejected_count": len(rejected_df)}

def airflow_step_engineer_features():
    import pandas as pd
    from src.config import STAGING_DATA_DIR
    from src.etl.feature_engineering import engineer_features
    
    cleaned_path = STAGING_DATA_DIR / "stg_cleaned_encounters.csv"
    cleaned_df = pd.read_csv(cleaned_path, low_memory=False)
    engineered_df = engineer_features(cleaned_df)
    
    engineered_path = STAGING_DATA_DIR / "stg_engineered_encounters.csv"
    engineered_df.to_csv(engineered_path, index=False)
    return {"engineered_shape": list(engineered_df.shape)}

def airflow_step_validate_quality():
    import pandas as pd
    from src.config import STAGING_DATA_DIR, REPORTS_DIR
    from src.etl.validators import run_data_quality_checks
    
    engineered_path = STAGING_DATA_DIR / "stg_engineered_encounters.csv"
    df = pd.read_csv(engineered_path, low_memory=False)
    report_file = REPORTS_DIR / "data_quality_report.json"
    summary = run_data_quality_checks(df, report_path=report_file)
    if not summary["all_checks_passed"]:
        print(f"Warning: Some quality checks failed: {summary['checks_failed']}")
    return summary

def airflow_step_load_mart():
    import pandas as pd
    from src.config import STAGING_DATA_DIR
    from src.etl.db_loader import build_and_load_data_mart
    
    engineered_path = STAGING_DATA_DIR / "stg_engineered_encounters.csv"
    df = pd.read_csv(engineered_path, low_memory=False)
    result = build_and_load_data_mart(df)
    return result

if AIRFLOW_AVAILABLE:
    dag = DAG(
        dag_id="hospital_readmission_etl_pipeline",
        default_args=default_args,
        description="End-to-end Healthcare Readmission ETL and Data Quality Pipeline",
        schedule_interval="0 2 * * *",  # Daily at 02:00 AM UTC
        catchup=False,
        tags=["healthcare", "etl", "readmission", "mlops"],
    )

    t1 = PythonOperator(
        task_id="ingest_raw_hospital_data",
        python_callable=airflow_step_ingest,
        dag=dag,
    )

    t2 = PythonOperator(
        task_id="clean_and_standardize_encounters",
        python_callable=airflow_step_clean,
        dag=dag,
    )

    t3 = PythonOperator(
        task_id="engineer_clinical_features",
        python_callable=airflow_step_engineer_features,
        dag=dag,
    )

    t4 = PythonOperator(
        task_id="validate_data_quality",
        python_callable=airflow_step_validate_quality,
        dag=dag,
    )

    t5 = PythonOperator(
        task_id="load_healthcare_data_mart",
        python_callable=airflow_step_load_mart,
        dag=dag,
    )

    t1 >> t2 >> t3 >> t4 >> t5
