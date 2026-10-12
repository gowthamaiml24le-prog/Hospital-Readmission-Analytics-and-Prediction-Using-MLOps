-- =============================================================================
-- Healthcare Readmission Analytics - Sample Verification & KPI Queries
-- =============================================================================

-- Query 1: Overall Hospital 30-Day Readmission Baseline KPI
SELECT 
    COUNT(*) AS total_eligible_encounters,
    SUM(readmitted_30d) AS readmitted_30d_encounters,
    ROUND(AVG(readmitted_30d) * 100, 2) AS readmission_rate_pct,
    ROUND(AVG(length_of_stay), 2) AS mean_los_days,
    ROUND(AVG(num_medications), 2) AS mean_medications_per_patient
FROM fact_readmission;

-- Query 2: Impact of Medication Changes on Readmission Rates
SELECT 
    CASE WHEN medication_change_flag = 1 THEN 'Medication Changed' ELSE 'No Medication Change' END AS medication_status,
    COUNT(*) AS encounter_count,
    SUM(readmitted_30d) AS readmitted_count,
    ROUND(AVG(readmitted_30d) * 100, 2) AS readmission_rate_pct
FROM fact_readmission
GROUP BY medication_change_flag;

-- Query 3: Top Comorbidity Drivers of 30-Day Readmission
SELECT 
    diag_1_category AS primary_diagnosis,
    comorbidity_cardio_diabetes,
    COUNT(*) AS encounters,
    ROUND(AVG(readmitted_30d) * 100, 2) AS readmission_rate_pct
FROM fact_readmission
GROUP BY diag_1_category, comorbidity_cardio_diabetes
HAVING COUNT(*) > 500
ORDER BY readmission_rate_pct DESC;

-- Query 4: Audit Verification - Ingestion and Rejected Record Counts
SELECT 
    'Valid Fact Encounters' AS metric, COUNT(*) AS record_count FROM fact_readmission
UNION ALL
SELECT 
    'Rejected Encounters (Expired/Hospice/Invalid)', COUNT(*) FROM rejected_records_log
UNION ALL
SELECT 
    'Unique De-identified Patients', COUNT(*) FROM patient_summary;
