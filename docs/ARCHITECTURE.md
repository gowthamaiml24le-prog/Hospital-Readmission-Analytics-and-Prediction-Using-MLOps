# Hospital Readmission Analytics & Prediction - Architecture Design

## 1. High-Level Multi-Tier Architecture Diagram

```mermaid
flowchart TD
    subgraph S1["1. Source Layer"]
        UCI["UCI Machine Learning Repository<br/>(130-US Hospitals Dataset 1999-2008)"]
        MAP["Clinical Mapping Lookup Tables<br/>(IDS_mapping.csv: Admission, Discharge, Source)"]
    end

    subgraph S2["2. Ingestion & Landing Layer"]
        DOWN["Automated Downloader<br/>(src/ingestion/download_dataset.py)"]
        LAND["Secure Landing Area<br/>(data/raw/diabetic_data.csv)"]
        INGEST["Ingestion Script & Audit Logging<br/>(src/ingestion/ingest_raw.py)"]
        AUDIT_LOG[("Ingestion Audit Log<br/>ingestion_audit_log")]
    end

    subgraph S3["3. Staging Layer"]
        STG_RAW["Raw Staging Snapshot<br/>(data/staging/stg_raw_encounters.csv)"]
    end

    subgraph S4["4. Transformation & Validation Layer"]
        CLEAN["Cleaner & De-Identification<br/>(src/etl/cleaner.py)<br/>• Replace '?' missing values<br/>• Filter expired/hospice cases<br/>• Generate SHA-256 Surrogate Keys"]
        REJECT[("Rejected Encounters Log<br/>data/rejected/rejected_records_log.csv")]
        ICD9["ICD-9 Clinical Categorizer<br/>(src/etl/icd9_mapper.py)<br/>CCS Mapping (Diabetes, Circulatory, etc.)"]
        FEAT["Clinical Feature Engineering<br/>(src/etl/feature_engineering.py)<br/>• Length of Stay, Prior Utilization<br/>• Comorbidity indices & Med changes"]
        VAL["Data Quality Validation Engine<br/>(src/etl/validators.py)<br/>• Schema, PK Uniqueness, Domain bounds"]
        DQ_REP["Quality Report<br/>(data_quality_report.json)"]
    end

    subgraph S5["5. Storage Layer (PostgreSQL / Relational Mart)"]
        DB[("Relational Healthcare Data Mart<br/>(PostgreSQL / healthcare_mart.db)")]
        T_PAT["dim_patient<br/>(patient_summary)"]
        T_ADM["dim_admission<br/>(admission)"]
        T_DIAG["dim_diagnosis<br/>(diagnosis)"]
        T_FACT["fact_readmission<br/>(readmission_mart)"]
    end

    subgraph S6["6. Analytics & Visualization Layer"]
        ST_APP["Streamlit Clinical Dashboard<br/>(app/dashboard.py)<br/>• View 1: Age & Demographics<br/>• View 2: Diagnosis & Department<br/>• View 3: Length of Stay (LOS)<br/>• View 4: Admission & Discharge<br/>• View 5: High-Risk Cohort Triage"]
    end

    subgraph S7["7. MLOps Layer (Prediction & Lifecycle)"]
        PREP["Stratified Train/Val/Test Split<br/>(src/mlops/prepare_data.py)"]
        TRAIN["Model Training & Benchmark<br/>(src/mlops/train.py)<br/>• Logistic Regression<br/>• Random Forest<br/>• XGBoost (Champion)"]
        MLFLOW[("MLflow Tracking & Registry<br/>(mlflow.db / mlruns)<br/>Metrics: ROC-AUC, Recall, Brier")]
        SHAP_MOD["Explainability Engine<br/>(src/mlops/explainability.py)<br/>• Global Beeswarm<br/>• Local Patient SHAP Attribution"]
        API["FastAPI Inference Service<br/>(src/api/main.py)<br/>Endpoints: /predict, /explain, /health"]
        MONITOR["Drift & Fairness Monitoring<br/>(src/mlops/monitoring.py)<br/>• KS Test & PSI Drift Detection<br/>• Subgroup Auditing (Age, Gender)"]
        DOCKER["Containerized Deployment<br/>(docker-compose.yml)"]
    end

    UCI --> DOWN
    MAP --> DOWN
    DOWN --> LAND
    LAND --> INGEST
    INGEST --> AUDIT_LOG
    INGEST --> STG_RAW
    STG_RAW --> CLEAN
    CLEAN --> REJECT
    CLEAN --> ICD9
    ICD9 --> FEAT
    FEAT --> VAL
    VAL --> DQ_REP
    FEAT --> DB
    DB --> T_PAT
    DB --> T_ADM
    DB --> T_DIAG
    DB --> T_FACT
    T_FACT --> ST_APP
    T_FACT --> PREP
    PREP --> TRAIN
    TRAIN --> MLFLOW
    TRAIN --> SHAP_MOD
    TRAIN --> API
    API --> ST_APP
    TRAIN --> MONITOR
    API --> DOCKER
    ST_APP --> DOCKER
    DB --> DOCKER
```

## 2. Orchestrated Pipeline Execution Flow

```mermaid
sequenceDiagram
    autonumber
    actor Admin as Healthcare Data Engineer / Orchestrator
    participant DAG as Apache Airflow DAG / Pipeline Runner
    participant Ingestion as Ingestion Module
    participant ETL as ETL & Cleaning Engine
    participant Validator as Quality Validator
    participant DB as PostgreSQL / Relational Mart
    participant ML as MLOps Training & MLflow
    participant API as FastAPI Inference
    participant Dash as Streamlit Dashboard

    Admin->>DAG: Trigger Pipeline Execution
    DAG->>Ingestion: Ingest Raw UCI Dataset & Record Audit
    Ingestion-->>DAG: Staged Raw Data & Logged Audit ID
    DAG->>ETL: Clean '?', Filter Expired/Hospice, Hash Keys & Engineer Features
    ETL-->>DAG: Engineered Dataset & Rejected Records Log
    DAG->>Validator: Run Data Quality Checks (9 Constraints)
    Validator-->>DAG: 9/9 Checks Passed (data_quality_report.json)
    DAG->>DB: Build and Populate Dimensional Tables
    DB-->>DAG: Loaded patient_summary, admission, diagnosis, fact_readmission
    DAG->>ML: Run Stratified Split, Train Models & Log to MLflow
    ML-->>DAG: XGBoost Selected Champion (ROC-AUC 0.6767, Recall 60.4%)
    DAG->>API: Serve Champion Model on /predict & /explain
    DAG->>Dash: Render 5 Clinical Views + Interactive Risk Scoring
    Dash-->>Admin: Full End-to-End Analytics & Decision-Support Ready
```

## 3. Layer Breakdown & Privacy Principles
* **Source Layer**: Public, anonymized UCI Diabetes 130-US Hospitals dataset.
* **Landing / Staging Layer**: Segregated landing zones (`data/raw/` and `data/staging/`) preserving exact raw extractions for reproducibility.
* **Transformation Layer**:
  - Replaces `?` characters with standard `NaN`/nulls.
  - Filters out deceased and hospice encounters (2,423 cases) who cannot be readmitted, preserving clinical validity.
  - Generates irreversible SHA-256 surrogate keys for HIPAA Safe Harbor compliance.
  - Standardizes diagnoses into CCS diagnostic categories.
* **Storage Layer**: Open-source relational dimensional model (PostgreSQL or SQLite data mart) structured into star schema (`patient_summary`, `admission`, `diagnosis`, `fact_readmission`).
* **Analytics Layer**: High-interactivity Streamlit application with 5 business views.
* **MLOps Layer**: Robust reproducible pipeline comparing Logistic Regression, Random Forest, and XGBoost; tracking experiments with MLflow; explaining predictions with SHAP; serving via FastAPI; containerized with Docker; monitored for data drift and demographic fairness.
