-- =============================================================================
-- Healthcare Readmission Analytics - Database Schema DDL
-- Database Dialect: PostgreSQL / ANSI SQL
-- Description: Star schema and analytical data mart for hospital readmission analytics.
-- =============================================================================

-- 1. Ingestion Audit Log Table
CREATE TABLE IF NOT EXISTS ingestion_audit_log (
    execution_id VARCHAR(64) PRIMARY KEY,
    extraction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    source VARCHAR(255) NOT NULL,
    raw_file VARCHAR(255),
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    duration_seconds NUMERIC(8, 2),
    status VARCHAR(32) NOT NULL
);

-- 2. Rejected Records Log Table
CREATE TABLE IF NOT EXISTS rejected_records_log (
    id SERIAL PRIMARY KEY,
    raw_encounter_id BIGINT,
    raw_patient_nbr BIGINT,
    rejection_reason VARCHAR(255) NOT NULL,
    discharge_disposition_id INTEGER,
    rejected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Patient Dimension Table (patient_summary)
CREATE TABLE IF NOT EXISTS patient_summary (
    surrogate_patient_id VARCHAR(64) PRIMARY KEY,
    race VARCHAR(64),
    gender VARCHAR(16),
    age_group VARCHAR(16),
    total_encounters INTEGER DEFAULT 1,
    avg_length_of_stay NUMERIC(5, 2),
    total_prior_inpatient INTEGER DEFAULT 0,
    total_prior_emergency INTEGER DEFAULT 0,
    has_cardio_diabetes SMALLINT DEFAULT 0,
    has_readmission_30d SMALLINT DEFAULT 0,
    high_utilizer_flag SMALLINT DEFAULT 0
);

-- 4. Admission Dimension Table
CREATE TABLE IF NOT EXISTS admission (
    surrogate_encounter_id VARCHAR(64) PRIMARY KEY,
    surrogate_patient_id VARCHAR(64) NOT NULL,
    admission_type_id INTEGER,
    admission_type_desc VARCHAR(128),
    admission_category VARCHAR(32),
    admission_source_id INTEGER,
    admission_source_desc VARCHAR(128),
    discharge_disposition_id INTEGER,
    discharge_disposition_desc VARCHAR(128),
    discharge_category VARCHAR(64),
    length_of_stay INTEGER NOT NULL,
    extended_stay_flag SMALLINT DEFAULT 0,
    FOREIGN KEY (surrogate_patient_id) REFERENCES patient_summary(surrogate_patient_id) ON DELETE CASCADE
);

-- 5. Diagnosis Dimension Table
CREATE TABLE IF NOT EXISTS diagnosis (
    surrogate_encounter_id VARCHAR(64) PRIMARY KEY,
    surrogate_patient_id VARCHAR(64) NOT NULL,
    diag_1 VARCHAR(32),
    diag_1_category VARCHAR(64),
    diag_2 VARCHAR(32),
    diag_2_category VARCHAR(64),
    diag_3 VARCHAR(32),
    diag_3_category VARCHAR(64),
    diagnosis_count INTEGER NOT NULL,
    high_comorbidity_flag SMALLINT DEFAULT 0,
    comorbidity_cardio_diabetes SMALLINT DEFAULT 0,
    FOREIGN KEY (surrogate_encounter_id) REFERENCES admission(surrogate_encounter_id) ON DELETE CASCADE
);

-- 6. Readmission Fact Table (fact_readmission)
CREATE TABLE IF NOT EXISTS fact_readmission (
    surrogate_encounter_id VARCHAR(64) PRIMARY KEY,
    surrogate_patient_id VARCHAR(64) NOT NULL,
    gender VARCHAR(16),
    age VARCHAR(16),
    age_midpoint INTEGER,
    race VARCHAR(64),
    admission_category VARCHAR(32),
    discharge_category VARCHAR(64),
    length_of_stay INTEGER NOT NULL,
    extended_stay_flag SMALLINT DEFAULT 0,
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    prior_inpatient_admissions INTEGER DEFAULT 0,
    prior_outpatient_admissions INTEGER DEFAULT 0,
    prior_emergency_admissions INTEGER DEFAULT 0,
    total_prior_visits INTEGER DEFAULT 0,
    has_prior_inpatient SMALLINT DEFAULT 0,
    frequent_utilizer_flag SMALLINT DEFAULT 0,
    diag_1_category VARCHAR(64),
    diagnosis_count INTEGER,
    high_comorbidity_flag SMALLINT DEFAULT 0,
    comorbidity_cardio_diabetes SMALLINT DEFAULT 0,
    active_diabetic_meds_count INTEGER,
    medication_change_flag SMALLINT DEFAULT 0,
    diabetes_med_flag SMALLINT DEFAULT 0,
    on_insulin SMALLINT DEFAULT 0,
    insulin_titration_up SMALLINT DEFAULT 0,
    insulin_titration_down SMALLINT DEFAULT 0,
    on_metformin SMALLINT DEFAULT 0,
    a1c_tested SMALLINT DEFAULT 0,
    a1c_high SMALLINT DEFAULT 0,
    glucose_tested SMALLINT DEFAULT 0,
    glucose_high SMALLINT DEFAULT 0,
    clinical_risk_index INTEGER DEFAULT 0,
    readmission_category VARCHAR(16) NOT NULL,
    readmitted_30d SMALLINT NOT NULL,
    FOREIGN KEY (surrogate_patient_id) REFERENCES patient_summary(surrogate_patient_id) ON DELETE CASCADE
);

-- Optimization Indexes
CREATE INDEX IF NOT EXISTS idx_fact_patient ON fact_readmission(surrogate_patient_id);
CREATE INDEX IF NOT EXISTS idx_fact_readmitted ON fact_readmission(readmitted_30d);
CREATE INDEX IF NOT EXISTS idx_fact_age ON fact_readmission(age);
CREATE INDEX IF NOT EXISTS idx_fact_diag ON fact_readmission(diag_1_category);
CREATE INDEX IF NOT EXISTS idx_admission_type ON admission(admission_category);
