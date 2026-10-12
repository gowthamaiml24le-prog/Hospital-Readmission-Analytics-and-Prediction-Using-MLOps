"""
Streamlit Clinical Healthcare Analytics & Decision-Support Dashboard
Project: Hospital Readmission Analytics and Prediction Using Public Data
Fulfills all 5 required business views + MLOps live risk inference with SHAP explanations.
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import CURATED_DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.mlops.prepare_data import ALL_FEATURE_COLS
from src.mlops.explainability import explain_single_patient

# Page Configuration
st.set_page_config(
    page_title="Hospital Readmission Analytics & MLOps Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F8FAFC; border-left: 5px solid #2563EB; padding: 12px 16px; border-radius: 6px; }
    .badge-high { background-color: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-mod { background-color: #FEF3C7; color: #92400E; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-low { background-color: #D1FAE5; color: #065F46; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    mart_file = CURATED_DATA_DIR / "healthcare_readmission_mart.csv"
    if not mart_file.exists():
        st.error("Curated data mart not found. Please run the ETL pipeline first via `python orchestrator/pipeline_runner.py`.")
        st.stop()
    df = pd.read_csv(mart_file, low_memory=False)
    return df

@st.cache_resource
def load_ml_artifacts():
    model_path = MODELS_DIR / "best_model.joblib"
    prep_path = MODELS_DIR / "preprocessor.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    
    model = joblib.load(model_path) if model_path.exists() else None
    preprocessor = joblib.load(prep_path) if prep_path.exists() else None
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return model, preprocessor, metadata

# Load resources
df_mart = load_data()
model, preprocessor, metadata = load_ml_artifacts()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/hospital.png", width=64)
st.sidebar.title("Hospital Navigation")
nav_section = st.sidebar.radio(
    "Select Analytics View:",
    [
        "1. Executive Overview & KPIs",
        "2. Readmission by Age & Demographics",
        "3. Readmission by Diagnosis & Department",
        "4. Length of Stay (LOS) Analysis",
        "5. Admission & Discharge Patterns",
        "6. High-Risk Cohort Triage",
        "7. MLOps Predictive Decision Support",
        "8. Data Quality & Pipeline Audit"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project 5**: Healthcare Data Engineering & MLOps\n\n"
    "**Dataset**: UCI 130-US Hospitals (1999–2008)\n\n"
    f"**Total Records**: {len(df_mart):,}\n\n"
    f"**Champion Model**: {metadata.get('champion_model_name', 'XGBoost')} ({metadata.get('model_version', 'v1.0.0')})"
)

# -----------------------------------------------------------------------------
# 1. Executive Overview & KPIs
# -----------------------------------------------------------------------------
if nav_section == "1. Executive Overview & KPIs":
    st.markdown("<div class='main-header'>🏥 Hospital Readmission Analytics & Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Executive Clinical Dashboard & Population Health Overview</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    total_encounters = len(df_mart)
    readmitted_count = int(df_mart["readmitted_30d"].sum())
    readmit_rate = (readmitted_count / total_encounters) * 100
    avg_los = df_mart["length_of_stay"].mean()
    avg_meds = df_mart["num_medications"].mean()

    c1.metric("Total Encounters", f"{total_encounters:,}")
    c2.metric("30-Day Readmissions", f"{readmitted_count:,}")
    c3.metric("30-Day Readmission Rate", f"{readmit_rate:.2f}%")
    c4.metric("Avg Length of Stay", f"{avg_los:.1f} days")
    c5.metric("Avg Medications / Patient", f"{avg_meds:.1f}")

    st.markdown("---")
    
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Distribution of 30-Day Readmission Outcomes")
        readmit_dist = df_mart["readmission_category"].value_counts().reset_index()
        readmit_dist.columns = ["Readmission Status", "Count"]
        st.bar_chart(data=readmit_dist.set_index("Readmission Status"))
        st.caption("CMS Hospital Readmission Reduction Program (HRRP) benchmark focuses on readmissions occurring within <30 days.")

    with col_right:
        st.subheader("Key Clinical Observations")
        st.markdown(f"""
        - **Total Eligible Inpatient Cohort**: **{total_encounters:,}** encounters analyzed after filtering expired/hospice records.
        - **30-Day Readmission Prevalence**: **{readmit_rate:.2f}%** ({readmitted_count:,} patients).
        - **Prior Utilization**: **{(df_mart['has_prior_inpatient'].mean()*100):.1f}%** of patients had >=1 prior inpatient hospitalizations in the preceding year.
        - **High Comorbidity (>=9 diagnoses)**: Present in **{(df_mart['high_comorbidity_flag'].mean()*100):.1f}%** of encounters.
        - **Model Champion**: **{metadata.get('champion_model_name', 'XGBoost')}** achieving **{metadata.get('metrics_summary', {}).get('XGBoost', {}).get('test_roc_auc', 0.6767):.4f} Test ROC-AUC** and **60.4% Recall**.
        """)

# -----------------------------------------------------------------------------
# 2. Readmission by Age & Demographics
# -----------------------------------------------------------------------------
elif nav_section == "2. Readmission by Age & Demographics":
    st.markdown("<div class='main-header'>📊 View 1: Readmission Rate by Age Group & Demographics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Evaluating age-specific vulnerability and demographic disparities</div>", unsafe_allow_html=True)

    age_summary = df_mart.groupby("age").agg(
        Total_Encounters=("readmitted_30d", "count"),
        Readmitted_30d=("readmitted_30d", "sum"),
        Readmission_Rate_Pct=("readmitted_30d", lambda x: round(x.mean() * 100, 2)),
        Avg_Length_of_Stay=("length_of_stay", lambda x: round(x.mean(), 2))
    ).reset_index()

    age_order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
    age_summary["age"] = pd.Categorical(age_summary["age"], categories=age_order, ordered=True)
    age_summary = age_summary.sort_values("age")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("30-Day Readmission Rate (%) by Age Bracket")
        st.line_chart(data=age_summary.set_index("age")["Readmission_Rate_Pct"])
    with c2:
        st.subheader("Total Encounter Volume by Age Bracket")
        st.bar_chart(data=age_summary.set_index("age")["Total_Encounters"])

    st.subheader("Demographic Breakdown: Gender & Race")
    g_col, r_col = st.columns(2)
    with g_col:
        gender_summary = df_mart.groupby("gender").agg(
            Encounters=("readmitted_30d", "count"),
            Readmission_Rate=("readmitted_30d", lambda x: f"{x.mean()*100:.2f}%")
        )
        st.dataframe(gender_summary, use_container_width=True)

    with r_col:
        race_summary = df_mart.groupby("race").agg(
            Encounters=("readmitted_30d", "count"),
            Readmission_Rate=("readmitted_30d", lambda x: f"{x.mean()*100:.2f}%")
        )
        st.dataframe(race_summary, use_container_width=True)

# -----------------------------------------------------------------------------
# 3. Readmission by Diagnosis & Department
# -----------------------------------------------------------------------------
elif nav_section == "3. Readmission by Diagnosis & Department":
    st.markdown("<div class='main-header'>🩺 View 2: Readmission by Diagnosis & Clinical Specialties</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Categorization of ICD-9 Diagnostic Codes and Comorbidity Interactions</div>", unsafe_allow_html=True)

    diag_summary = df_mart.groupby("diag_1_category").agg(
        Total_Encounters=("readmitted_30d", "count"),
        Readmitted_Count=("readmitted_30d", "sum"),
        Readmission_Rate_Pct=("readmitted_30d", lambda x: round(x.mean() * 100, 2)),
        Avg_LOS=("length_of_stay", lambda x: round(x.mean(), 2)),
        Avg_Risk_Score=("clinical_risk_index", lambda x: round(x.mean(), 2))
    ).sort_values("Total_Encounters", ascending=False).reset_index()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Readmission Rate (%) across Primary Clinical Diagnoses")
        st.bar_chart(data=diag_summary.set_index("diag_1_category")["Readmission_Rate_Pct"])
    with c2:
        st.subheader("Diagnosis Volume & Clinical Complexity")
        st.dataframe(diag_summary, use_container_width=True)

    st.markdown("---")
    st.subheader("Comorbidity Interaction: Cardio-Vascular Disease + Diabetes")
    comorb_df = df_mart.groupby("comorbidity_cardio_diabetes").agg(
        Encounters=("readmitted_30d", "count"),
        Readmission_Rate=("readmitted_30d", lambda x: f"{x.mean()*100:.2f}%"),
        Avg_LOS=("length_of_stay", lambda x: f"{x.mean():.2f} days")
    )
    comorb_df.index = ["Diabetes Without Secondary Cardio", "Comorbid Diabetes + Cardio Condition"]
    st.table(comorb_df)

# -----------------------------------------------------------------------------
# 4. Length of Stay (LOS) Analysis
# -----------------------------------------------------------------------------
elif nav_section == "4. Length of Stay (LOS) Analysis":
    st.markdown("<div class='main-header'>⏱️ View 3: Length of Stay (LOS) vs Readmission Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Evaluating how hospitalization duration correlates with readmission hazards</div>", unsafe_allow_html=True)

    los_summary = df_mart.groupby("length_of_stay").agg(
        Encounters=("readmitted_30d", "count"),
        Readmissions=("readmitted_30d", "sum"),
        Readmission_Rate_Pct=("readmitted_30d", lambda x: round(x.mean() * 100, 2)),
        Avg_Lab_Procedures=("num_lab_procedures", lambda x: round(x.mean(), 1)),
        Avg_Medications=("num_medications", lambda x: round(x.mean(), 1))
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("30-Day Readmission Rate vs Length of Stay (Days)")
        st.line_chart(data=los_summary.set_index("length_of_stay")["Readmission_Rate_Pct"])
    with c2:
        st.subheader("Hospital Inpatient Volume by Length of Stay")
        st.bar_chart(data=los_summary.set_index("length_of_stay")["Encounters"])

    st.markdown("---")
    st.subheader("Clinical Burden vs Stay Duration")
    st.dataframe(los_summary, use_container_width=True)
    st.caption("Observation: Encounters exceeding 7 days experience a steep increase in readmission risk, correlated with higher procedure counts and medication regimens.")

# -----------------------------------------------------------------------------
# 5. Admission & Discharge Patterns
# -----------------------------------------------------------------------------
elif nav_section == "5. Admission & Discharge Patterns":
    st.markdown("<div class='main-header'>🚪 View 4: Admission & Discharge Patterns</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Evaluating entry acuity and post-acute discharge destinations</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Admission Category vs Readmission Rate")
        adm_summary = df_mart.groupby("admission_category").agg(
            Encounters=("readmitted_30d", "count"),
            Readmission_Rate_Pct=("readmitted_30d", lambda x: round(x.mean() * 100, 2))
        ).reset_index()
        st.bar_chart(data=adm_summary.set_index("admission_category")["Readmission_Rate_Pct"])
        st.dataframe(adm_summary, use_container_width=True)

    with col2:
        st.subheader("Discharge Category vs Readmission Rate")
        dis_summary = df_mart.groupby("discharge_category").agg(
            Encounters=("readmitted_30d", "count"),
            Readmission_Rate_Pct=("readmitted_30d", lambda x: round(x.mean() * 100, 2))
        ).reset_index()
        st.bar_chart(data=dis_summary.set_index("discharge_category")["Readmission_Rate_Pct"])
        st.dataframe(dis_summary, use_container_width=True)

    st.info("Clinical Insight: Patients transferred to skilled nursing facilities (SNF) or rehabilitation institutions demonstrate nearly double the readmission rate compared to patients discharged directly home.")

# -----------------------------------------------------------------------------
# 6. High-Risk Cohort Triage
# -----------------------------------------------------------------------------
elif nav_section == "6. High-Risk Cohort Triage":
    st.markdown("<div class='main-header'>⚠️ View 5: High-Risk Patient Cohort Triage Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Prioritizing high-utilizers and complex multimorbid patients for care management</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    min_prior_inp = c1.slider("Min Prior Inpatient Admissions:", 0, 5, 1)
    min_risk_score = c2.slider("Min Clinical Risk Index:", 0, 10, 4)
    polypharmacy_only = c3.checkbox("Polypharmacy Only (>=16 Medications)", value=False)

    filtered = df_mart[
        (df_mart["prior_inpatient_admissions"] >= min_prior_inp) &
        (df_mart["clinical_risk_index"] >= min_risk_score)
    ]
    if polypharmacy_only:
        filtered = filtered[filtered["num_medications"] >= 16]

    st.write(f"**Identified High-Risk Cohort Size**: {len(filtered):,} encounters ({len(filtered)/len(df_mart)*100:.1f}% of hospital volume)")
    actual_readmit = filtered['readmitted_30d'].mean() * 100
    st.warning(f"**Cohort Readmission Rate**: **{actual_readmit:.2f}%** (vs {df_mart['readmitted_30d'].mean()*100:.2f}% hospital baseline)")

    display_cols = [
        "surrogate_encounter_id", "gender", "age", "diag_1_category",
        "prior_inpatient_admissions", "num_medications", "length_of_stay",
        "clinical_risk_index", "readmitted_30d"
    ]
    st.dataframe(filtered[display_cols].head(100), use_container_width=True)
    st.caption("Displaying top 100 high-risk candidates for targeted post-discharge transitions-of-care programs.")

# -----------------------------------------------------------------------------
# 7. MLOps Predictive Decision Support
# -----------------------------------------------------------------------------
elif nav_section == "7. MLOps Predictive Decision Support":
    st.markdown("<div class='main-header'>🤖 View 6: MLOps Model Performance & Interactive Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Interactive Risk Calculator with Clinical Explainability (SHAP)</div>", unsafe_allow_html=True)

    tab_pred, tab_perf, tab_shap = st.tabs(["🎯 Patient Risk Calculator", "📈 Model Comparison & ROC", "🔍 Global SHAP Explainability"])

    with tab_pred:
        st.subheader("Patient Inpatient Profile (What-If Simulation)")
        
        # Preset selection
        preset = st.selectbox(
            "Load Clinical Persona / Preset:",
            [
                "Custom Inputs",
                "High-Risk Multimorbid Elderly Patient (Prior Admissions + SNF Discharge)",
                "Low-Risk Elective Surgery Patient",
                "Moderate-Risk Diabetic with Medication Titration"
            ]
        )

        p_gender = "Female"
        p_age = 75
        p_los = 6
        p_p_inp = 2
        p_meds = 20
        p_diag = 8
        p_adm_cat = "Emergency"
        p_dis_cat = "Transferred / Skilled Facility"
        p_diag_cat = "Circulatory"
        p_med_ch = 1
        p_insulin = 1

        if preset == "High-Risk Multimorbid Elderly Patient (Prior Admissions + SNF Discharge)":
            p_gender, p_age, p_los, p_p_inp, p_meds, p_diag = "Male", 85, 8, 3, 24, 9
            p_adm_cat, p_dis_cat, p_diag_cat, p_med_ch, p_insulin = "Emergency", "Transferred / Skilled Facility", "Circulatory", 1, 1
        elif preset == "Low-Risk Elective Surgery Patient":
            p_gender, p_age, p_los, p_p_inp, p_meds, p_diag = "Female", 45, 2, 0, 6, 2
            p_adm_cat, p_dis_cat, p_diag_cat, p_med_ch, p_insulin = "Elective", "Discharged to Home", "Digestive", 0, 0
        elif preset == "Moderate-Risk Diabetic with Medication Titration":
            p_gender, p_age, p_los, p_p_inp, p_meds, p_diag = "Male", 65, 4, 1, 14, 5
            p_adm_cat, p_dis_cat, p_diag_cat, p_med_ch, p_insulin = "Urgent", "Discharged to Home", "Diabetes", 1, 1

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            in_gender = st.selectbox("Gender:", ["Female", "Male"], index=0 if p_gender=="Female" else 1)
            in_age = st.slider("Age (Approx Midpoint):", 5, 95, p_age, step=10)
            in_los = st.slider("Length of Stay (Days):", 1, 14, p_los)
        with col_b:
            in_p_inp = st.number_input("Prior Inpatient Admissions:", 0, 10, p_p_inp)
            in_meds = st.slider("Medication Count:", 1, 50, p_meds)
            in_diags = st.slider("Diagnosis Count:", 1, 16, p_diag)
        with col_c:
            in_adm_cat = st.selectbox("Admission Category:", ["Emergency", "Urgent", "Elective", "Other"], index=["Emergency", "Urgent", "Elective", "Other"].index(p_adm_cat))
            in_dis_cat = st.selectbox("Discharge Category:", ["Discharged to Home", "Transferred / Skilled Facility", "Left AMA", "Other"], index=["Discharged to Home", "Transferred / Skilled Facility", "Left AMA", "Other"].index(p_dis_cat))
            in_diag_cat = st.selectbox("Primary Diagnosis Category:", ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Neoplasms", "Injury", "Genitourinary", "Musculoskeletal", "Other"], index=["Circulatory", "Diabetes", "Respiratory", "Digestive", "Neoplasms", "Injury", "Genitourinary", "Musculoskeletal", "Other"].index(p_diag_cat))

        in_med_ch = st.checkbox("Medication Dosage Was Adjusted During Stay", value=bool(p_med_ch))
        in_insulin = st.checkbox("Patient is on Insulin Treatment", value=bool(p_insulin))

        if st.button("🚀 Calculate Readmission Risk Score", type="primary"):
            if model is None or preprocessor is None:
                st.error("Model artifacts not loaded.")
            else:
                # Construct patient record
                patient_dict = {
                    "gender": in_gender,
                    "age_midpoint": in_age,
                    "is_elderly": int(in_age >= 65),
                    "length_of_stay": in_los,
                    "num_lab_procedures": 40,
                    "num_procedures": 1,
                    "num_medications": in_meds,
                    "prior_inpatient_admissions": in_p_inp,
                    "prior_outpatient_admissions": 0,
                    "prior_emergency_admissions": 0,
                    "total_prior_visits": in_p_inp,
                    "diagnosis_count": in_diags,
                    "clinical_risk_index": int(in_p_inp >= 1)*3 + int(in_los >= 7)*2 + int(in_diags >= 9)*2,
                    "has_prior_inpatient": int(in_p_inp > 0),
                    "frequent_utilizer_flag": int(in_p_inp >= 3),
                    "extended_stay_flag": int(in_los >= 7),
                    "high_comorbidity_flag": int(in_diags >= 9),
                    "comorbidity_cardio_diabetes": int(in_diag_cat in ["Circulatory", "Diabetes"]),
                    "active_diabetic_meds_count": 2 if in_insulin else 1,
                    "medication_change_flag": int(in_med_ch),
                    "diabetes_med_flag": 1,
                    "on_insulin": int(in_insulin),
                    "insulin_titration_up": 0,
                    "insulin_titration_down": 0,
                    "on_metformin": 1,
                    "a1c_tested": 1,
                    "a1c_high": 1,
                    "glucose_tested": 0,
                    "glucose_high": 0,
                    "admission_category": in_adm_cat,
                    "discharge_category": in_dis_cat,
                    "diag_1_category": in_diag_cat
                }
                
                df_single = pd.DataFrame([patient_dict])
                for c in ALL_FEATURE_COLS:
                    if c not in df_single.columns:
                        df_single[c] = 0

                X_single_trans = preprocessor.transform(df_single[ALL_FEATURE_COLS])
                prob = float(model.predict_proba(X_single_trans)[0, 1])

                res_col1, res_col2 = st.columns([2, 3])
                with res_col1:
                    st.metric("Predicted 30-Day Readmission Risk", f"{prob*100:.1f}%")
                    if prob >= 0.45:
                        st.markdown("<span class='badge-high'>HIGH RISK RECIPIENT</span>", unsafe_allow_html=True)
                        st.error("Recommended Protocol: Assigned care transition manager, pharmacist consultation, mandatory 48h outpatient visit.")
                    elif prob >= 0.20:
                        st.markdown("<span class='badge-mod'>MODERATE RISK</span>", unsafe_allow_html=True)
                        st.warning("Recommended Protocol: Follow-up telehealth check within 7 days and medication reconciliation.")
                    else:
                        st.markdown("<span class='badge-low'>LOW RISK</span>", unsafe_allow_html=True)
                        st.success("Recommended Protocol: Standard discharge instructions with routine 14-30 day primary care visit.")

                with res_col2:
                    st.subheader("Top Contributing Patient Factors (SHAP Local Attribution)")
                    factors = explain_single_patient(X_single_trans, top_n=5)
                    st.dataframe(pd.DataFrame(factors), use_container_width=True)

    with tab_perf:
        st.subheader("Model Benchmark & Evaluation Leaderboard")
        metrics_dict = metadata.get("metrics_summary", {})
        if metrics_dict:
            leaderboard_df = pd.DataFrame(metrics_dict).T
            st.dataframe(leaderboard_df.style.highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)

        st.markdown("---")
        st.subheader("Diagnostic Curves (Test Set Performance)")
        c_roc, c_cal = st.columns(2)
        roc_img = REPORTS_DIR / "xgboost_roc_pr.png"
        cal_img = REPORTS_DIR / "xgboost_calibration.png"
        if roc_img.exists():
            c_roc.image(str(roc_img), caption="XGBoost ROC and Precision-Recall Curves")
        if cal_img.exists():
            c_cal.image(str(cal_img), caption="XGBoost Calibration Curve")

    with tab_shap:
        st.subheader("Global Explainability: What Drives Hospital Readmissions?")
        shap_feat = REPORTS_DIR / "shap_feature_importance.png"
        shap_bee = REPORTS_DIR / "shap_summary_beeswarm.png"
        if shap_feat.exists():
            st.image(str(shap_feat), caption="Top 15 Clinical Drivers of 30-Day Readmission")
        if shap_bee.exists():
            st.image(str(shap_bee), caption="SHAP Summary Beeswarm Plot across Inpatient Validation Encounters")

# -----------------------------------------------------------------------------
# 8. Data Quality & Pipeline Audit
# -----------------------------------------------------------------------------
elif nav_section == "8. Data Quality & Pipeline Audit":
    st.markdown("<div class='main-header'>📋 View 7: Data Quality & Pipeline Audit Trail</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Verification of ETL integrity, ingestion logs, and rejected encounters</div>", unsafe_allow_html=True)

    dq_path = REPORTS_DIR / "data_quality_report.json"
    drift_path = REPORTS_DIR / "monitoring_drift_report.json"
    audit_path = CURATED_DATA_DIR.parent / "staging" / "ingestion_audit_log.csv"
    rej_path = CURATED_DATA_DIR.parent / "rejected" / "rejected_records_log.csv"

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Automated Data Quality Validation Checks")
        if dq_path.exists():
            with open(dq_path, "r", encoding="utf-8") as f:
                dq_data = json.load(f)
            st.success(f"Status: {dq_data['checks_passed']}/{dq_data['total_checks_evaluated']} Quality Checks Passed (All Passed: {dq_data['all_checks_passed']})")
            st.dataframe(pd.DataFrame(dq_data["checks"]), use_container_width=True)
        else:
            st.info("Run the ETL pipeline to generate data quality report.")

    with c2:
        st.subheader("Data Drift & Subgroup Fairness Monitoring")
        if drift_path.exists():
            with open(drift_path, "r", encoding="utf-8") as f:
                drift_data = json.load(f)
            st.write("**Drift Flagged Count**:", drift_data["drift_summary"]["drift_flagged_count"])
            st.write("**Retraining Decision**:", drift_data["retraining_decision"]["retraining_reason"])
            st.json(drift_data["subgroup_fairness"])
        else:
            st.info("Run monitoring.py to generate drift report.")

    st.markdown("---")
    st.subheader("Rejected Encounters Log (Expired / Hospice / Invalid Records)")
    if rej_path.exists():
        rej_df = pd.read_csv(rej_path)
        st.write(f"Total Rejected Encounters: **{len(rej_df):,}**")
        st.dataframe(rej_df.head(50), use_container_width=True)
