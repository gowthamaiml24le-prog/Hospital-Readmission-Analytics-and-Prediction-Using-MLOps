# Execution Evidence & Pipeline Demonstration

This document provides rigorous execution evidence for the end-to-end Healthcare Readmission Data Engineering & MLOps Pipeline, satisfying Item 9 of the Assignment submission package.

---

## 1. End-to-End Pipeline Execution Evidence

### Execution Command:
```bash
python orchestrator/pipeline_runner.py
```

### Complete Terminal Output Log:
```text
2026-10-12 21:34:18,010 [INFO] STARTING HEALTHCARE READMISSION ETL PIPELINE EXECUTION
2026-10-12 21:34:18,010 [INFO] >>> [Stage 1/5] Ingesting Raw UCI Hospital Data...
2026-10-12 21:34:19,005 [INFO] Loaded raw dataset with 101,766 rows and 50 columns.
2026-10-12 21:34:20,519 [INFO] Saved raw staging snapshot to data/staging/stg_raw_encounters.csv
2026-10-12 21:34:20,586 [INFO] Successfully recorded ingestion audit entry to database.
2026-10-12 21:34:20,586 [INFO] Ingestion completed in 2.51s. Audit ID: INGEST_20261012_160420
2026-10-12 21:34:20,592 [INFO] Stage 1 Complete: 101,766 raw records ingested.
2026-10-12 21:34:20,592 [INFO] >>> [Stage 2/5] Cleaning, Filtering Ineligible Encounters & Anonymizing...
2026-10-12 21:34:21,781 [INFO] Starting data cleaning pipeline on 101,766 raw encounter records.
2026-10-12 21:34:22,217 [INFO] Loaded mappings: 8 admission types, 30 discharge types, 25 admission sources.
2026-10-12 21:34:22,872 [INFO] Logged 2,426 rejected records to data/rejected/rejected_records_log.csv
2026-10-12 21:34:22,872 [INFO] Cleaning complete: 99,340 valid records retained (2,426 rejected, 97.62% retention).
2026-10-12 21:34:24,169 [INFO] Stage 2 Complete: 99,340 valid records, 2,426 rejected records.
2026-10-12 21:34:24,169 [INFO] >>> [Stage 3/5] Engineering Clinical, Diagnostic, and Utilization Features...
2026-10-12 21:34:32,803 [INFO] Feature engineering complete. Total features: 94. 30-Day Readmission prevalence: 11.39%
2026-10-12 21:34:35,121 [INFO] Stage 3 Complete: Engineered dataset shape: (99340, 94)
2026-10-12 21:34:35,122 [INFO] >>> [Stage 4/5] Executing Data Quality Integrity Checks...
2026-10-12 21:34:35,122 [INFO] Running data quality validation on 99,340 records...
2026-10-12 21:34:35,156 [INFO] Data quality validation report saved to artifacts/reports/data_quality_report.json
2026-10-12 21:34:35,156 [INFO] Quality Check Status: 9/9 passed. All passed: True
2026-10-12 21:34:35,156 [INFO] Stage 4 Complete: Passed 9/9 checks.
2026-10-12 21:34:35,156 [INFO] >>> [Stage 5/5] Building & Loading Relational Data Mart Tables...
2026-10-12 21:34:35,158 [INFO] Building patient_summary table...
2026-10-12 21:34:35,360 [INFO] Building admission table...
2026-10-12 21:34:35,368 [INFO] Building diagnosis table...
2026-10-12 21:34:35,372 [INFO] Building readmission analytics fact table...
2026-10-12 21:34:37,141 [INFO] Saved all curated tables to CSV in data/curated/
2026-10-12 21:34:47,183 [INFO] Successfully loaded tables into database: {'patient_summary': 69987, 'admission': 99340, 'diagnosis': 99340, 'fact_readmission': 99340}
2026-10-12 21:34:47,187 [INFO] PIPELINE RUN COMPLETED SUCCESSFULLY IN 29.18s
```

---

## 2. Automated Test Suite Execution Evidence

### Execution Command:
```bash
python -m pytest tests -v
```

### Complete Test Results Log:
```text
tests/test_api.py::test_root_endpoint PASSED                             [  9%]
tests/test_api.py::test_health_endpoint PASSED                           [ 18%]
tests/test_api.py::test_metrics_endpoint PASSED                          [ 27%]
tests/test_api.py::test_predict_endpoint PASSED                          [ 36%]
tests/test_api.py::test_explain_endpoint PASSED                          [ 45%]
tests/test_etl.py::test_icd9_mapping PASSED                              [ 54%]
tests/test_etl.py::test_surrogate_key_generation PASSED                  [ 63%]
tests/test_etl.py::test_cleaning_and_filtering PASSED                    [ 72%]
tests/test_etl.py::test_feature_engineering_and_dq PASSED                [ 81%]
tests/test_ingestion.py::test_raw_files_exist PASSED                     [ 90%]
tests/test_ingestion.py::test_ingest_raw_execution PASSED                [100%]

======================= 11 passed in 9.09s =======================
```

---

## 3. MLOps Model Training & Comparison Leaderboard

### Execution Command:
```bash
python src/mlops/train.py
```

### Benchmark Metrics Table (Test Set):
| Model Candidate | Validation ROC-AUC | Test ROC-AUC | Test Recall (Sensitivity) | Test Precision | Test F1-Score | Test PR-AUC | Test Brier Score | Status |
|---|---|---|---|---|---|---|---|---|
| **XGBoost (Tuned)** | **0.6678** | **0.6767** | **60.40%** | **18.72%** | **0.2858** | **0.2014** | **0.2189** | **Champion Model (Registered v1.0.0)** |
| **Random Forest** | 0.6621 | 0.6738 | 58.87% | 18.55% | 0.2821 | 0.1982 | 0.2181 | Challenger |
| **Logistic Regression** | 0.6526 | 0.6690 | 58.40% | 17.89% | 0.2740 | 0.1895 | 0.2291 | Linear Baseline |

---

## 4. Model Serving API Output Evidence

### Sample POST Request (`/predict`):
```json
{
  "gender": "Male",
  "age_midpoint": 75,
  "is_elderly": 1,
  "length_of_stay": 6,
  "prior_inpatient_admissions": 2,
  "total_prior_visits": 3,
  "num_medications": 18,
  "diag_1_category": "Circulatory",
  "discharge_category": "Transferred / Skilled Facility"
}
```

### Response Received (Status: 200 OK):
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

### Local SHAP Explainability Response (`/explain`):
```json
{
  "readmission_probability": 0.6078,
  "top_contributing_risk_factors": [
    {
      "feature": "discharge_category_Transferred / Skilled Facility",
      "shap_value": 0.3103,
      "effect": "Increases Risk"
    },
    {
      "feature": "prior_inpatient_admissions",
      "shap_value": 0.1549,
      "effect": "Increases Risk"
    },
    {
      "feature": "a1c_tested",
      "shap_value": -0.1345,
      "effect": "Decreases Risk"
    },
    {
      "feature": "on_metformin",
      "shap_value": -0.0823,
      "effect": "Decreases Risk"
    },
    {
      "feature": "diagnosis_count",
      "shap_value": 0.0776,
      "effect": "Increases Risk"
    }
  ]
}
```

---

## 5. Data Quality Validation Verification

All 9 automated data quality checks passed with 100% compliance:
```json
{
  "total_records_evaluated": 99340,
  "total_checks_evaluated": 9,
  "checks_passed": 9,
  "checks_failed": 0,
  "all_checks_passed": true
}
```
All artifacts, charts, and database tables are located in `artifacts/reports/`, `artifacts/models/`, and `data/curated/`.
