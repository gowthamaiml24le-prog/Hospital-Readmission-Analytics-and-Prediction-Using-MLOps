# Dataset Source Information and Access Instructions

## 1. Primary Dataset Overview
* **Name**: Diabetes 130-US Hospitals for Years 1999–2008 Data Set
* **Originating Institution**: Center for Clinical and Translational Research, Virginia Commonwealth University
* **Repository**: UC Irvine Machine Learning Repository (UCI ML Repository)
* **Dataset DOI / Identifier**: `10.24432/C5230J`
* **Direct Archive Download URL**: [https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip](https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip)
* **Access License**: Open Data Commons Public Domain Dedication and License (CC BY 4.0 / Open Access for Research)

## 2. Dataset Description & Clinical Significance
The dataset represents 10 years (1999–2008) of clinical care at 130 US hospitals and integrated delivery networks. The records comprise hospitalized patients with a confirmed diagnosis of diabetes mellitus where the length of stay ranged from 1 to 14 days.

### Clinical Inclusion Criteria:
1. Encounter is an inpatient hospitalization.
2. Patient has an established diagnosis of diabetes (ICD-9 codes starting with `250.xx` or specified antidiabetic medication administration).
3. Hospital stay duration was between 1 and 14 days.
4. Laboratory tests and antidiabetic medications were administered during the encounter.

### Dataset Volume:
* **Total Encounters**: 101,766 inpatient encounters
* **Unique Patients**: 71,518 unique patients
* **Total Raw Attributes**: 50 features spanning demographics, clinical laboratory procedures, diagnoses, medications, and readmission outcomes.

## 3. Data Privacy and Safe Harbor Compliance
In accordance with HIPAA Safe Harbor and project requirements:
1. All direct identifiers (e.g., patient names, social security numbers, medical record numbers, dates of birth) were completely stripped by the originating institution prior to public dissemination.
2. In our pipeline, raw `patient_nbr` and `encounter_id` are cryptographically salted and hashed into irreversible surrogate keys (`surrogate_patient_id` and `surrogate_encounter_id`) using SHA-256 before ingestion into downstream analytical and machine learning layers.
3. No protected health information (PHI) or personally identifiable information (PII) is stored or exposed.

## 4. Reproducible Automated Download Instructions
To automatically acquire, verify, and unpack the dataset:
```bash
python src/ingestion/download_dataset.py
```
This script downloads the official UCI zip archive directly into `data/raw/diabetes_dataset.zip`, extracts `diabetic_data.csv` and `IDS_mapping.csv`, and records file size and SHA integrity.
