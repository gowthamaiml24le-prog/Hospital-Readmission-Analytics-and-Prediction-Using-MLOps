# Healthcare Readmission Data Dictionary & Validation Rules

## 1. Dimensional Model Schema

### Table 1: `patient_summary` (Patient Dimension)
Contains patient-level aggregations and demographics keyed by irreversible surrogate keys.

| Column Name | Data Type | Nullable | Description | Source / Derivation |
|---|---|---|---|---|
| `surrogate_patient_id` | VARCHAR(64) | NO (PK) | Irreversible salted SHA-256 hash of patient identifier | Derived from `patient_nbr` |
| `race` | VARCHAR(64) | YES | Patient racial background (Caucasian, AfricanAmerican, Hispanic, Asian, Other, Unknown) | `race` (imputed 'Unknown' for '?') |
| `gender` | VARCHAR(16) | NO | Biological sex ('Female', 'Male') | `gender` |
| `age_group` | VARCHAR(16) | NO | 10-year age bracket (e.g. `[60-70)`) | `age` |
| `total_encounters` | INTEGER | NO | Total inpatient hospital encounters recorded | COUNT(encounters) |
| `avg_length_of_stay` | NUMERIC(5,2) | NO | Average hospitalization duration in days | AVG(time_in_hospital) |
| `total_prior_inpatient` | INTEGER | NO | Maximum prior inpatient visits recorded | MAX(number_inpatient) |
| `total_prior_emergency` | INTEGER | NO | Maximum prior emergency department visits | MAX(number_emergency) |
| `has_cardio_diabetes` | SMALLINT | NO | Flag (1/0) indicating concurrent diabetic and cardiovascular diagnoses | Derived |
| `has_readmission_30d` | SMALLINT | NO | Flag (1/0) indicating if patient ever experienced a 30-day readmission | MAX(readmitted_30d) |
| `high_utilizer_flag` | SMALLINT | NO | Flag (1/0) indicating patient with >=2 hospital encounters | Derived |

---

### Table 2: `admission` (Encounter Admission & Discharge Dimension)
Contains encounter-level admission acuity, referral sources, and post-acute discharge disposition.

| Column Name | Data Type | Nullable | Description | Source / Derivation |
|---|---|---|---|---|
| `surrogate_encounter_id` | VARCHAR(64) | NO (PK) | Irreversible salted SHA-256 hash of encounter ID | Derived from `encounter_id` |
| `surrogate_patient_id` | VARCHAR(64) | NO (FK) | Surrogate key linking to `patient_summary` | FK |
| `admission_type_id` | INTEGER | NO | Numeric identifier for admission type | `admission_type_id` |
| `admission_type_desc` | VARCHAR(128) | NO | Mapped text description (Emergency, Urgent, Elective, Newborn, etc.) | Mapped via `IDS_mapping.csv` |
| `admission_category` | VARCHAR(32) | NO | Standardized high-level category (Emergency, Urgent, Elective, Other) | Standardized |
| `admission_source_id` | INTEGER | NO | Numeric identifier for admission source | `admission_source_id` |
| `admission_source_desc` | VARCHAR(128) | NO | Source description (Emergency Room, Physician Referral, Clinic, etc.) | Mapped via `IDS_mapping.csv` |
| `discharge_disposition_id`| INTEGER | NO | Numeric identifier for discharge destination | `discharge_disposition_id` |
| `discharge_disposition_desc`| VARCHAR(128)| NO | Discharge text description (Home, SNF, Home Health, etc.) | Mapped via `IDS_mapping.csv` |
| `discharge_category` | VARCHAR(64) | NO | High-level category (Discharged to Home, Transferred / Skilled Facility, Left AMA, Other) | Standardized |
| `length_of_stay` | INTEGER | NO | Hospital length of stay in days (1 to 14) | `time_in_hospital` |
| `extended_stay_flag` | SMALLINT | NO | Flag (1/0) indicating prolonged stay (>=7 days) | Derived |

---

### Table 3: `diagnosis` (Diagnostic Dimension)
Captures primary, secondary, and tertiary ICD-9 diagnoses mapped to clinical categories.

| Column Name | Data Type | Nullable | Description | Source / Derivation |
|---|---|---|---|---|
| `surrogate_encounter_id` | VARCHAR(64) | NO (PK) | Surrogate encounter identifier | FK |
| `surrogate_patient_id` | VARCHAR(64) | NO | Surrogate patient identifier | FK |
| `diag_1` | VARCHAR(32) | NO | Primary ICD-9 diagnostic code (e.g. 428, 250.02) | `diag_1` |
| `diag_1_category` | VARCHAR(64) | NO | Primary CCS category (Circulatory, Diabetes, Respiratory, etc.) | Standardized via CCS ICD-9 Mapper |
| `diag_2` | VARCHAR(32) | YES | Secondary ICD-9 diagnostic code | `diag_2` |
| `diag_2_category` | VARCHAR(64) | YES | Secondary CCS category | Standardized via CCS ICD-9 Mapper |
| `diag_3` | VARCHAR(32) | YES | Tertiary ICD-9 diagnostic code | `diag_3` |
| `diag_3_category` | VARCHAR(64) | YES | Tertiary CCS category | Standardized via CCS ICD-9 Mapper |
| `diagnosis_count` | INTEGER | NO | Total diagnoses recorded for encounter (1 to 16) | `number_diagnoses` |
| `high_comorbidity_flag` | SMALLINT | NO | Flag (1/0) indicating multimorbid patient (>=9 diagnoses) | Derived |
| `comorbidity_cardio_diabetes` | SMALLINT | NO | Flag (1/0) indicating presence of both diabetes and cardiovascular disease | Derived |

---

### Table 4: `fact_readmission` (Analytics & Fact Mart)
The unified healthcare analytics mart linking clinical utilization, medications, and readmission labels.

| Column Name | Data Type | Nullable | Description | Derivation / Formula |
|---|---|---|---|---|
| `surrogate_encounter_id` | VARCHAR(64) | NO (PK) | Surrogate encounter identifier | SHA-256 |
| `surrogate_patient_id` | VARCHAR(64) | NO (FK) | Surrogate patient identifier | SHA-256 |
| `gender` | VARCHAR(16) | NO | Patient biological sex ('Female', 'Male') | Cleaned |
| `age` | VARCHAR(16) | NO | Original 10-year bracket (`[70-80)`) | Cleaned |
| `age_midpoint` | INTEGER | NO | Numeric midpoint of age bracket (e.g. 75 for `[70-80)`) | Cleaned |
| `race` | VARCHAR(64) | NO | Patient race | Cleaned |
| `length_of_stay` | INTEGER | NO | Inpatient days (1 to 14) | `time_in_hospital` |
| `num_lab_procedures` | INTEGER | NO | Count of laboratory procedures during stay | `num_lab_procedures` |
| `num_procedures` | INTEGER | NO | Count of surgical/diagnostic procedures during stay | `num_procedures` |
| `num_medications` | INTEGER | NO | Count of distinct medications administered | `num_medications` |
| `prior_inpatient_admissions`| INTEGER | NO | Number of prior inpatient hospitalizations | `number_inpatient` |
| `prior_outpatient_admissions`| INTEGER | NO | Number of prior outpatient encounters | `number_outpatient` |
| `prior_emergency_admissions`| INTEGER | NO | Number of prior emergency department visits | `number_emergency` |
| `total_prior_visits` | INTEGER | NO | Sum of all prior inpatient, outpatient, and emergency visits | `inpatient + outpatient + emergency` |
| `active_diabetic_meds_count`| INTEGER | NO | Count of active diabetic medications with dosage prescribed | Count of meds != 'No' |
| `medication_change_flag`| SMALLINT | NO | 1 if diabetic medication dosage was adjusted or changed | `change == 'Ch'` |
| `diabetes_med_flag` | SMALLINT | NO | 1 if any diabetic medication was prescribed | `diabetesMed == 'Yes'` |
| `on_insulin` | SMALLINT | NO | 1 if patient received insulin therapy | `insulin != 'No'` |
| `a1c_tested` | SMALLINT | NO | 1 if HbA1c test was performed | `A1Cresult != 'None'` |
| `a1c_high` | SMALLINT | NO | 1 if HbA1c test result was > 8% (suboptimal control) | `A1Cresult in ['>7', '>8']` |
| `clinical_risk_index` | INTEGER | NO | Composite triage score (0-10) factoring prior admissions, LOS, comorbidities | Heuristic risk index |
| `readmission_category` | VARCHAR(16) | NO | Original dataset readmission label (`<30`, `>30`, `NO`) | `readmitted` |
| `readmitted_30d` | SMALLINT | NO | **Primary Target Variable**: 1 if readmitted <30 days, 0 otherwise | `readmitted == '<30'` |

---

## 2. Automated Data Quality Validation Rules

The automated data quality engine (`src/etl/validators.py`) enforces 9 strict integrity constraints on every pipeline execution:

| Check Name | Target Column / Entity | Rule Type | Acceptance Criteria | Enforcement Action |
|---|---|---|---|---|
| `surrogate_encounter_id_uniqueness` | `surrogate_encounter_id` | Uniqueness | Zero duplicate keys | Pipeline failure / deduplication |
| `surrogate_patient_id_not_null` | `surrogate_patient_id` | Non-Null | Zero null values allowed | Pipeline failure |
| `target_readmitted_30d_not_null` | `readmitted_30d` | Non-Null | Zero null values allowed | Pipeline failure |
| `length_of_stay_range_1_to_14` | `length_of_stay` | Value Boundary | 1 <= `length_of_stay` <= 14 | Log violation & cap |
| `diagnosis_count_positive` | `diagnosis_count` | Range Check | `diagnosis_count` >= 1 | Log violation |
| `medication_count_positive` | `medication_count` | Range Check | `medication_count` >= 1 | Log violation |
| `gender_domain_validity` | `gender` | Categorical Domain | Value in `['Female', 'Male']` | Route to `rejected_records_log` |
| `readmitted_30d_binary_domain`| `readmitted_30d` | Binary Domain | Value in `[0, 1]` | Pipeline failure |
| `prior_utilization_non_negative`| Prior visits | Boundary Check | All prior counts >= 0 | Log violation |

---

## 3. Rejected Records Policy
Encounters matching any of the following clinical exclusion criteria are automatically quarantined to `data/rejected/rejected_records_log.csv` and excluded from readmission models:
1. **Expired Inpatient Discharges**: Patients who expired in the hospital (`discharge_disposition_id` in `[11, 19, 20, 21]`). Clinically ineligible for readmission.
2. **Hospice Transfers**: Patients transferred to home hospice or medical hospice facility (`discharge_disposition_id` in `[13, 14]`).
3. **Invalid Demographics**: Encounters with `gender == 'Unknown/Invalid'` (3 encounters in raw data).
