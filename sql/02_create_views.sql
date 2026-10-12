-- =============================================================================
-- Healthcare Readmission Analytics - Analytical Views
-- Dialect: PostgreSQL / ANSI SQL
-- Description: Aggregated business intelligence views supporting dashboards.
-- =============================================================================

-- View 1: Readmission Rate by Age Group
CREATE OR REPLACE VIEW vw_readmission_by_age AS
SELECT 
    age AS age_group,
    COUNT(*) AS total_encounters,
    SUM(readmitted_30d) AS readmitted_30d_count,
    ROUND(AVG(readmitted_30d) * 100, 2) AS readmission_rate_pct,
    ROUND(AVG(length_of_stay), 2) AS avg_length_of_stay,
    ROUND(AVG(medication_count), 2) AS avg_medications
FROM fact_readmission
GROUP BY age
ORDER BY MIN(age_midpoint);

-- View 2: Readmission by Primary Diagnosis Category
CREATE OR REPLACE VIEW vw_readmission_by_diagnosis AS
SELECT 
    diag_1_category AS primary_diagnosis_category,
    COUNT(*) AS total_encounters,
    SUM(readmitted_30d) AS readmitted_30d_count,
    ROUND(AVG(readmitted_30d) * 100, 2) AS readmission_rate_pct,
    ROUND(AVG(length_of_stay), 2) AS avg_los,
    ROUND(AVG(clinical_risk_index), 2) AS avg_risk_score
FROM fact_readmission
GROUP BY diag_1_category
ORDER BY total_encounters DESC;

-- View 3: Length of Stay (LOS) vs Readmission Analysis
CREATE OR REPLACE VIEW vw_length_of_stay_analysis AS
SELECT 
    length_of_stay,
    COUNT(*) AS total_encounters,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d) * 100, 2) AS readmission_rate_pct,
    ROUND(AVG(num_lab_procedures), 1) AS avg_lab_procedures,
    ROUND(AVG(num_procedures), 1) AS avg_procedures
FROM fact_readmission
GROUP BY length_of_stay
ORDER BY length_of_stay;

-- View 4: Admission and Discharge Pattern Analysis
CREATE OR REPLACE VIEW vw_admission_discharge_patterns AS
SELECT 
    admission_category,
    discharge_category,
    COUNT(*) AS total_encounters,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d) * 100, 2) AS readmission_rate_pct,
    ROUND(AVG(length_of_stay), 2) AS avg_los
FROM fact_readmission
GROUP BY admission_category, discharge_category
ORDER BY total_encounters DESC;

-- View 5: High-Risk Cohort Identification (Clinical Triage)
CREATE OR REPLACE VIEW vw_high_risk_cohort AS
SELECT 
    surrogate_encounter_id,
    surrogate_patient_id,
    age,
    gender,
    diag_1_category,
    prior_inpatient_admissions,
    total_prior_visits,
    medication_count,
    length_of_stay,
    clinical_risk_index,
    readmitted_30d
FROM fact_readmission
WHERE prior_inpatient_admissions >= 1
   OR clinical_risk_index >= 5
ORDER BY clinical_risk_index DESC, prior_inpatient_admissions DESC;
