# Hospital Readmission Analytics and Prediction Using Public Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**Course**: Data Engineering and MLOps | **Project 5**  
**Dataset**: UCI Diabetes 130-US Hospitals for Years 1999–2008 (101,766 Encounters)  
**Implementation**: Complete Part 1 (Data Engineering Pipeline) + Part 2 (MLOps Pipeline Extension)

---

## 🌟 Project Highlights

* **Automated & Audited Ingestion**: Automated acquisition of the 130-US Hospitals dataset with metadata auditing and raw snapshot persistence.
* **HIPAA Safe Harbor Privacy Compliance**: Direct patient and encounter identifiers are cryptographically salted and hashed into irreversible SHA-256 surrogate keys.
* **Clinical Data Cleansing**: Identifies special missing codes (`?`), standardizes alphanumeric ICD-9 codes into Clinical Classifications Software (CCS) diagnostic categories, and quarantines non-readmittable expired/hospice encounters.
* **Automated Data Quality Validation**: 9-point integrity test suite enforcing primary key uniqueness, nullity, clinical value bounds, and categorical domains (100% pass rate).
* **Star Schema Healthcare Data Mart**: Relational star schema (`patient_summary`, `admission`, `diagnosis`, `fact_readmission`) loaded into PostgreSQL / open-source relational database.
* **Interactive Streamlit Dashboard**: 7 rich clinical views including executive KPIs, age disparities, diagnosis complexity, length of stay hazards, discharge dispositions, high-risk triage, and pipeline audit logs.
* **Comparative MLOps Benchmarking**: Stratified 70/15/15 train/val/test splitting, cost-sensitive class imbalance correction, and MLflow experiment tracking comparing **Logistic Regression**, **Random Forest**, and **XGBoost** (Champion Model: **Test ROC-AUC = 0.6767**, **Recall = 60.40%** at baseline, **88.24%** at clinical sensitivity threshold).
* **Explainable AI (SHAP)**: Global beeswarm summary plots and local patient-level bedside feature attributions.
* **FastAPI Serving & Docker Containerization**: Low-latency RESTful API (`/predict`, `/explain`, `/health`, `/metrics`) containerized with Docker Compose.
* **Continuous Monitoring & Governance**: Automated Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI) drift detection, subgroup clinical fairness audits, and clear retraining triggers.

---

## 📂 Project Repository Structure

```text
MLops_Project/
├── .env.example                     # Environment variables template
├── .env                             # Active configuration
├── .gitignore                       # Git ignore rules
├── pytest.ini                       # Test configuration
├── requirements.txt                 # Python dependencies
├── docker-compose.yml               # Multi-container orchestration (Postgres, API, UI, MLflow)
├── Dockerfile.api                   # FastAPI Dockerfile
├── Dockerfile.streamlit             # Streamlit Dockerfile
│
├── data/
│   ├── raw/                         # Downloaded raw zip archive and extracted CSVs
│   ├── staging/                     # Staged raw snapshot and ingestion audit trail
│   ├── curated/                     # Star-schema CSV artifacts (patient_summary, admission, etc.)
│   ├── analytics/                   # Relational database (healthcare_mart.db, mlflow.db)
│   └── rejected/                    # Quarantined expired/hospice/invalid records
│
├── sql/
│   ├── 01_schema_ddl.sql            # PostgreSQL relational star-schema DDL
│   ├── 02_create_views.sql          # Pre-compiled analytical views for dashboards
│   └── 03_sample_queries.sql        # Verification and healthcare KPI queries
│
├── src/
│   ├── config.py                    # Global paths, constants, DB connection factory
│   ├── ingestion/
│   │   ├── download_dataset.py      # Automated dataset downloader
│   │   └── ingest_raw.py            # Ingestion script with audit metadata recording
│   ├── etl/
│   │   ├── cleaner.py               # Missing '?' handling, expired filtering, SHA-256 keys
│   │   ├── icd9_mapper.py           # ICD-9 to clinical category (CCS) mapper
│   │   ├── id_mapper.py             # Parses IDS_mapping.csv for admission/discharge descriptions
│   │   ├── feature_engineering.py   # Length of stay, prior visits, comorbidity indices
│   │   ├── validators.py            # Automated 9-point data quality validation engine
│   │   └── db_loader.py             # Relational data mart loader and CSV exporter
│   ├── mlops/
│   │   ├── prepare_data.py          # Stratified train/val/test splitting & ColumnTransformer
│   │   ├── train.py                 # Multi-model training, MLflow tracking, champion registry
│   │   ├── evaluate.py              # Metrics (ROC-AUC, PR-AUC, Recall, Brier) & diagnostic plots
│   │   ├── explainability.py        # Global and local SHAP explainability engine
│   │   └── monitoring.py            # Data drift (KS test / PSI) and subgroup fairness monitoring
│   └── api/
│       └── main.py                  # FastAPI inference and decision-support microservice
│
├── dags/
│   └── hospital_readmission_dag.py  # Production Apache Airflow DAG
│
├── orchestrator/
│   └── pipeline_runner.py           # Standalone local orchestrator running full ETL pipeline
│
├── app/
│   └── dashboard.py                 # Streamlit clinical analytics & prediction web app
│
├── tests/
│   ├── test_ingestion.py            # Unit tests for data ingestion
│   ├── test_etl.py                  # Unit tests for cleaning, hashing, and feature engineering
│   └── test_api.py                  # Integration tests for FastAPI endpoints
│
├── docs/
│   ├── ARCHITECTURE.md              # Multi-tier architecture & Mermaid pipeline flows
│   ├── DATASET_INFO.md              # Dataset source details and access guidelines
│   ├── DATA_DICTIONARY.md           # Star-schema data dictionary and validation rules
│   ├── EXECUTION_EVIDENCE.md        # Complete terminal execution logs and benchmarks
│   ├── RETRAINING_POLICY.md         # Continuous monitoring, drift thresholds & retraining policy
│   └── PROJECT_REPORT.md            # Comprehensive 8-10 page academic project report
│
└── artifacts/
    ├── models/                      # Serialized champion model, preprocessor, and metadata
    └── reports/                     # Quality reports, ROC/PR curves, SHAP beeswarm plots, drift audits
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
Clone the repository and install the dependencies:
```bash
# Clone the project repository
cd MLops_Project

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute the Full Data Pipeline (Part 1)
Run the automated pipeline orchestrator to download raw data, clean, de-identify, engineer features, validate data quality, and load the relational data mart:
```bash
python orchestrator/pipeline_runner.py
```
*Expected Execution Duration: ~25-30 seconds.*

### 3. Train Models & Track with MLflow (Part 2)
Train and compare Logistic Regression, Random Forest, and XGBoost models, log experiments to MLflow, generate diagnostic plots, and register the champion model:
```bash
python src/mlops/train.py
```

### 4. Generate SHAP Explainability & Run Drift Monitoring
```bash
# Generate global beeswarm and feature importance plots
python src/mlops/explainability.py

# Run KS-test drift detection and subgroup clinical fairness audits
python src/mlops/monitoring.py
```

---

## 🚀 Running the Applications

### Launch the Streamlit Clinical Dashboard
```bash
streamlit run app/dashboard.py
```
Open your browser at `http://localhost:8501` to explore:
1. **Executive Overview & KPIs**: Inpatient population metrics and HRRP benchmark indicators.
2. **Readmission by Age & Demographics**: Age bracket hazard curves and gender comparisons.
3. **Readmission by Diagnosis & Department**: ICD-9 CCS categories and diabetes comorbidity interactions.
4. **Length of Stay (LOS) Analysis**: Hospitalization duration vs. readmission rate.
5. **Admission & Discharge Patterns**: Emergency vs elective intake and skilled nursing transfer hazards.
6. **High-Risk Cohort Triage**: Interactive filter prioritizing multimorbid high-utilizers.
7. **MLOps Predictive Decision Support**: Interactive What-If patient risk calculator with live SHAP waterfall attributions!
8. **Data Quality & Audit Trail**: Real-time inspection of validation reports and rejected encounter logs.

---

### Launch the FastAPI Inference Microservice
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Access the interactive Swagger UI at:
* **Interactive API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Service Heartbeat**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* **Model Benchmark Metrics**: [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics)

#### Sample Prediction Request via cURL:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Male",
       "age_midpoint": 75,
       "is_elderly": 1,
       "length_of_stay": 6,
       "prior_inpatient_admissions": 2,
       "total_prior_visits": 3,
       "num_medications": 18,
       "diagnosis_count": 8,
       "admission_category": "Emergency",
       "discharge_category": "Transferred / Skilled Facility",
       "diag_1_category": "Circulatory"
     }'
```

#### Sample JSON Response:
```json
{
  "readmission_risk_probability": 0.6078,
  "risk_percentage": "60.8%",
  "risk_tier": "High Risk",
  "predicted_class_default": 1,
  "predicted_class_clinical_threshold": 1,
  "clinical_recommendation": "URGENT INTERVENTION: Post-discharge care management coordinator assigned, home-health check, and follow-up appointment within 48-72 hours.",
  "model_version": "v1.0.0"
}
```

---

## 🧪 Automated Test Suite

Execute the complete test suite verifying ingestion, data cleaning, feature engineering, and API endpoints:
```bash
python -m pytest tests -v
```
*Result: 11 / 11 tests passed (100% pass rate).*

---

## 🐳 Docker Multi-Container Deployment

To spin up the entire production stack (PostgreSQL, FastAPI API, Streamlit UI, MLflow Server) via Docker Compose:
```bash
docker-compose up --build
```
* **PostgreSQL Database**: `localhost:5432`
* **FastAPI Microservice**: `http://localhost:8000/docs`
* **Streamlit Dashboard**: `http://localhost:8501`
* **MLflow Tracking UI**: `http://localhost:5000`

---

## 📊 Summary of Model Performance

| Model Candidate | Validation ROC-AUC | Test ROC-AUC | Test Recall (Sensitivity) | Test Precision | Test Brier Score | Registration Status |
|---|---|---|---|---|---|---|
| **XGBoost (Tuned)** | **0.6678** | **0.6767** | **60.40%** | **18.72%** | **0.2189** | **Champion Model (`v1.0.0`)** |
| **Random Forest** | 0.6621 | 0.6738 | 58.87% | 18.55% | 0.2181 | Challenger Model |
| **Logistic Regression** | 0.6526 | 0.6690 | 58.40% | 17.89% | 0.2291 | Linear Baseline |

*At the clinical high-sensitivity operating threshold of 0.20, XGBoost achieves **88.24% sensitivity**, successfully intercepting nearly 9 out of 10 readmission candidates for preventative care transitions.*

---

## 📑 Assignment Submission Package Checklist

| Item | Description | File / Directory |
|:---:|---|---|
| **1** | Source code and ETL scripts | [src/](src/) (`ingestion/`, `etl/`, `mlops/`, `api/`) |
| **2** | Airflow DAG & Orchestration | [dags/hospital_readmission_dag.py](dags/hospital_readmission_dag.py), [orchestrator/](orchestrator/) |
| **3** | Database schema and sample tables | [sql/01_schema_ddl.sql](sql/01_schema_ddl.sql), [data/curated/](data/curated/) |
| **4** | Dataset source information & access | [docs/DATASET_INFO.md](docs/DATASET_INFO.md) |
| **5** | Architecture diagram & pipeline flow | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| **6** | Data dictionary and validation rules | [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) |
| **7** | Streamlit application | [app/dashboard.py](app/dashboard.py) |
| **8** | Project report (8-10 pages) | [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) |
| **9** | Execution evidence & benchmarks | [docs/EXECUTION_EVIDENCE.md](docs/EXECUTION_EVIDENCE.md) |
| **10** | README with setup & run instructions | [README.md](README.md) |
