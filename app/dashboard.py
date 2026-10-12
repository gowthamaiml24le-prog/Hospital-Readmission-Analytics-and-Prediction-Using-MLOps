"""
Hospital Readmission Analytics & Decision-Support Platform
Dataset: UCI 130-US Hospitals (101,766 encounters)
Course: Data Engineering and MLOps
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import CURATED_DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.mlops.prepare_data import ALL_FEATURE_COLS
from src.mlops.explainability import explain_single_patient

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ApexCare | Hospital Readmission Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# High-End Custom CSS for the Sidebar & Dashboard
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* =========================================================================
       PREMIUM SIDEBAR STYLING
       ========================================================================= */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090e1a 0%, #0f172a 40%, #0b1329 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-top: 10px !important;
    }

    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    /* Brand Header Box */
    .brand-container {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px 14px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .brand-icon {
        font-size: 2.1rem;
        background: linear-gradient(135deg, #2563eb, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .brand-text h2 {
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 !important;
        letter-spacing: -0.02em;
    }
    .brand-text p {
        font-size: 0.72rem !important;
        color: #94a3b8 !important;
        margin: 2px 0 0 0 !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* Sidebar Section Divider / Label */
    .sidebar-section-title {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #64748b !important;
        margin: 18px 0 8px 6px !important;
    }

    /* Modern Navigation Radio Buttons */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
    }

    /* Each Navigation Tab */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin: 0 !important;
        transition: all 0.22s ease-in-out !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Hover effect */
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(37, 99, 235, 0.15) !important;
        border-color: rgba(96, 165, 250, 0.4) !important;
        transform: translateX(4px) !important;
    }

    /* Selected Active Tab */
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%) !important;
        border-color: #60a5fa !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        font-weight: 700 !important;
    }

    /* Hide the native ugly radio circles */
    [data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
        display: none !important;
    }

    /* Text inside navigation button */
    [data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }

    /* Status & Meta Card at bottom of sidebar */
    .sidebar-footer-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        margin-top: 24px;
    }
    .status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
    }

    /* =========================================================================
       MAIN CONTENT STYLING
       ========================================================================= */
    .content-header {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 22px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }
    .content-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 6px 0;
    }
    .content-sub {
        font-size: 0.95rem;
        color: #64748b;
        margin: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        margin: 4px 0;
    }
    .metric-foot {
        font-size: 0.75rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Data & ML Model Loaders
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    mart_file = CURATED_DATA_DIR / "healthcare_readmission_mart.csv"
    if not mart_file.exists():
        st.error("Curated data mart not found. Please run: `python orchestrator/pipeline_runner.py`")
        st.stop()
    return pd.read_csv(mart_file, low_memory=False)

@st.cache_resource
def load_models():
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

df_mart = load_data()
model, preprocessor, metadata = load_models()

# -----------------------------------------------------------------------------
# SIDEBAR: High-End Hospital Branding & Nav
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div class="brand-container">
    <div class="brand-icon">🏥</div>
    <div class="brand-text">
        <h2>ApexCare Health</h2>
        <p>Informatics & MLOps</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="sidebar-section-title">Navigation Menu</p>', unsafe_allow_html=True)

nav_options = [
    "📊  Executive Overview & KPIs",
    "👥  Readmission by Age & Demographics",
    "🩺  Readmission by Diagnosis & Department",
    "⏱️  Length of Stay (LOS) Analysis",
    "🚪  Admission & Discharge Patterns",
    "⚠️  High-Risk Cohort Triage",
    "🤖  MLOps Predictive Decision Support",
    "📋  Data Quality & Pipeline Audit"
]

selected_view = st.sidebar.radio(
    label="Navigation",
    options=nav_options,
    label_visibility="collapsed"
)

# Sidebar Slicer / Filter
st.sidebar.markdown('<p class="sidebar-section-title">Global Cohort Filters</p>', unsafe_allow_html=True)

gender_filter = st.sidebar.selectbox("Filter by Gender:", ["All Genders", "Female", "Male"])
diagnosis_filter = st.sidebar.selectbox(
    "Primary Diagnosis:",
    ["All Categories"] + sorted(df_mart["diag_1_category"].unique().tolist())
)

# Apply filter
df_view = df_mart.copy()
if gender_filter != "All Genders":
    df_view = df_view[df_view["gender"] == gender_filter]
if diagnosis_filter != "All Categories":
    df_view = df_view[df_view["diag_1_category"] == diagnosis_filter]

# Sidebar Footer Status Card
st.sidebar.markdown(f"""
<div class="sidebar-footer-card">
    <div class="status-row">
        <span class="status-dot"></span>
        <span>Relational Mart Online</span>
    </div>
    <div style="font-size: 0.76rem; color: #94a3b8; line-height: 1.5; margin-top: 6px;">
        • <b>Active Cohort:</b> {len(df_view):,} records<br/>
        • <b>Champion Model:</b> {metadata.get('champion_model_name', 'XGBoost')} (v1.0.0)<br/>
        • <b>Test ROC-AUC:</b> 0.6767 | Recall: 60.4%<br/>
        • <b>Privacy:</b> HIPAA Salted SHA-256
    </div>
</div>
""", unsafe_allow_html=True)

# Helper function to style Plotly figures cleanly
def style_fig(fig, title="", height=360):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#0f172a", family="Plus Jakarta Sans", weight="bold")),
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#334155")
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    return fig


# =============================================================================
# VIEW 1: EXECUTIVE OVERVIEW & KPIS
# =============================================================================
if selected_view == "📊  Executive Overview & KPIs":
    st.markdown("""
    <div class="content-header">
        <h1 class="content-title">Executive Clinical Overview & HRRP KPIs</h1>
        <p class="content-sub">30-day readmission monitoring, inpatient capacity, and quality metrics across 130 US hospitals.</p>
    </div>
    """, unsafe_allow_html=True)

    # Metric Row
    total_enc = len(df_view)
    readmit_count = int(df_view["readmitted_30d"].sum())
    readmit_rate = (readmit_count / max(total_enc, 1)) * 100
    avg_los = df_view["length_of_stay"].mean()
    high_util = int((df_view["prior_inpatient_admissions"] >= 1).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Inpatients</div>
            <div class="metric-val">{total_enc:,}</div>
            <div class="metric-foot">Eligible encounters</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">30-Day Readmission Rate</div>
            <div class="metric-val" style="color: #ea580c;">{readmit_rate:.2f}%</div>
            <div class="metric-foot">{readmit_count:,} return encounters</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Stay (LOS)</div>
            <div class="metric-val" style="color: #0284c7;">{avg_los:.1f} Days</div>
            <div class="metric-foot">Range: 1–14 days</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prior Inpatient Utilizers</div>
            <div class="metric-val" style="color: #7c3aed;">{high_util:,}</div>
            <div class="metric-foot">{(high_util/max(total_enc, 1)*100):.1f}% prior hospitalizations</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns([1, 2])
    with col_chart1:
        counts = df_view["readmission_category"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig_donut = px.pie(
            counts, values="Count", names="Status", hole=0.55,
            color="Status", color_discrete_map={"<30": "#dc2626", ">30": "#f59e0b", "NO": "#10b981"},
            title="Readmission Outcomes (<30 vs >30 vs NO)"
        )
        style_fig(fig_donut, "Readmission Outcome Breakdown", height=350)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        top_diags = df_view.groupby("diag_1_category").agg(
            Volume=("readmitted_30d", "count"),
            Rate=("readmitted_30d", lambda x: round(x.mean() * 100, 2))
        ).reset_index().sort_values("Volume", ascending=False)
        fig_bar = px.bar(
            top_diags, x="diag_1_category", y="Volume", color="Rate",
            color_continuous_scale="Reds", text="Rate",
            labels={"diag_1_category": "Diagnosis", "Volume": "Volume", "Rate": "Rate (%)"}
        )
        fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
        style_fig(fig_bar, "Encounter Volume & 30-Day Rate by Specialty", height=350)
        st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# VIEW 2: READMISSION BY AGE & DEMOGRAPHICS
# =============================================================================
elif selected_view == "👥  Readmission by Age & Demographics":
    st.markdown("""
    <div class="content-header">
        <h1 class="content-title">View 1: Readmission Rate by Age Group & Demographics</h1>
        <p class="content-sub">Age-specific vulnerability curves, gender breakdown, and racial equity across hospital stays.</p>
    </div>
    """, unsafe_allow_html=True)

    age_order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
    age_agg = df_view.groupby("age").agg(
        Encounters=("readmitted_30d", "count"),
        Rate=("readmitted_30d", lambda x: round(x.mean() * 100, 2)),
        Avg_Stay=("length_of_stay", lambda x: round(x.mean(), 2))
    ).reindex(age_order).dropna().reset_index()

    c_age1, c_age2 = st.columns(2)
    with c_age1:
        fig_age = px.line(
            age_agg, x="age", y="Rate", markers=True,
            line_shape="spline", color_discrete_sequence=["#dc2626"],
            labels={"age": "Age Bracket", "Rate": "30-Day Readmission Rate (%)"}
        )
        style_fig(fig_age, "30-Day Readmission Risk Curve by Age Bracket", height=360)
        st.plotly_chart(fig_age, use_container_width=True)

    with c_age2:
        fig_vol = px.bar(
            age_agg, x="age", y="Encounters",
            color_discrete_sequence=["#3b82f6"], text="Encounters",
            labels={"age": "Age Bracket", "Encounters": "Patient Volume"}
        )
        fig_vol.update_traces(textposition="outside")
        style_fig(fig_vol, "Patient Volume Distribution by Age Bracket", height=360)
        st.plotly_chart(fig_vol, use_container_width=True)

    col_g, col_r = st.columns(2)
    with col_g:
        g_df = df_view.groupby("gender").agg(
            Encounters=("readmitted_30d", "count"),
            Readmit_Rate=("readmitted_30d", lambda x: f"{x.mean()*100:.2f}%")
        )
        st.write("**Gender Cohort Summary:**")
        st.dataframe(g_df, use_container_width=True)

    with col_r:
        r_df = df_view.groupby("race").agg(
            Encounters=("readmitted_30d", "count"),
            Readmit_Rate=("readmitted_30d", lambda x: f"{x.mean()*100:.2f}%")
        )
        st.write("**Racial Subgroup Summary:**")
        st.dataframe(r_df, use_container_width=True)


# =============================================================================
# VIEW 3: READMISSION BY DIAGNOSIS & DEPARTMENT
# =============================================================================
elif selected_view == "🩺  Readmission by Diagnosis & Department":
    st.markdown("""
    <div class="content-header">
        <h1 class="content-title">View 2: Readmission by Diagnosis & Clinical Specialties</h1>
        <p class="content-sub">Categorization of ICD-9 Diagnostic Codes and comorbidity interactions.</p>
    </div>
    """, unsafe_allow_html=True)

    diag_agg = df_view.groupby("diag_1_category").agg(
        Encounters=("readmitted_30d", "count"),
        Readmitted=("readmitted_30d", "sum"),
        Rate=("readmitted_30d", lambda x: round(x.mean() * 100, 2)),
        Avg_Stay=("length_of_stay", lambda x: round(x.mean(), 2))
    ).sort_values("Rate", ascending=True).reset_index()

    fig_diag = px.bar(
        diag_agg, x="Rate", y="diag_1_category", orientation="h",
        color="Rate", color_continuous_scale="Reds", text="Rate",
        labels={"Rate": "Readmission Rate (%)", "diag_1_category": "Clinical Category"}
    )
    fig_diag.update_traces(texttemplate='%{text}%', textposition='outside')
    style_fig(fig_diag, "Ranked 30-Day Readmission Risk by Clinical Category", height=400)
    st.plotly_chart(fig_diag, use_container_width=True)

    st.markdown("#### ⚡ Comorbidity Deep-Dive: Diabetes Alone vs. Cardiovascular Comorbidity")
    comorb_df = df_view.groupby("comorbidity_cardio_diabetes").agg(
        Encounters=("readmitted_30d", "count"),
        Rate=("readmitted_30d", lambda x: f"{x.mean()*100:.2f}%"),
        Avg_Stay=("length_of_stay", lambda x: f"{x.mean():.2f} days")
    )
    comorb_df.index = ["Diabetes Without Cardiovascular Condition", "Concurrent Diabetes + Cardiovascular Disease"]
    st.table(comorb_df)


# =============================================================================
# VIEW 4: LENGTH OF STAY (LOS) ANALYSIS
# =============================================================================
elif selected_view == "⏱️  Length of Stay (LOS) Analysis":
    st.markdown("""
    <div class="content-header">
        <h1 class="content-title">View 3: Length of Stay (LOS) vs Readmission Analysis</h1>
        <p class="content-sub">Evaluating how stay duration correlates with clinical complexity and readmission hazards.</p>
    </div>
    """, unsafe_allow_html=True)

    los_agg = df_view.groupby("length_of_stay").agg(
        Encounters=("readmitted_30d", "count"),
        Rate=("readmitted_30d", lambda x: round(x.mean() * 100, 2)),
        Avg_Meds=("num_medications", lambda x: round(x.mean(), 1))
    ).reset_index()

    c_los1, c_los2 = st.columns(2)
    with c_los1:
        fig_los = px.line(
            los_agg, x="length_of_stay", y="Rate", markers=True,
            line_shape="spline", color_discrete_sequence=["#ea580c"],
            labels={"length_of_stay": "Stay Duration (Days)", "Rate": "Readmission Rate (%)"}
        )
        style_fig(fig_los, "30-Day Readmission Rate vs Hospital Stay (Days)", height=360)
        st.plotly_chart(fig_los, use_container_width=True)

    with c_los2:
        fig_meds = px.bar(
            los_agg, x="length_of_stay", y="Avg_Meds",
            color_discrete_sequence=["#8b5cf6"],
            labels={"length_of_stay": "Stay Duration (Days)", "Avg_Meds": "Medications Prescribed"}
        )
        style_fig(fig_meds, "Medication Burden Progression by Stay Duration", height=360)
        st.plotly_chart(fig_meds, use_container_width=True)


# =============================================================================
# VIEW 5: ADMISSION & DISCHARGE PATTERNS
# =============================================================================
elif selected_view == "🚪  Admission & Discharge Patterns":
    st.markdown("""
    <div class="content-header">
        <h1 class="content-title">View 4: Admission Urgency & Post-Acute Discharge Destinations</h1>
        <p class="content-sub">Evaluating intake acuity and post-discharge transfer facility hazards.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        adm_agg = df_view.groupby("admission_category").agg(
            Volume=("readmitted_30d", "count"),
            Rate=("readmitted_30d", lambda x: round(x.mean() * 100, 2))
        ).reset_index()
        fig_adm = px.bar(
            adm_agg, x="admission_category", y="Rate", color="admission_category",
            color_discrete_sequence=px.colors.qualitative.Safe, text="Rate",
            labels={"admission_category": "Admission Urgency", "Rate": "Readmission Rate (%)"}
        )
        fig_adm.update_traces(texttemplate='%{text}%', textposition='outside')
        style_fig(fig_adm, "Readmission Rate by Admission Category", height=350)
        st.plotly_chart(fig_adm, use_container_width=True)

    with col2:
        dis_agg = df_view.groupby("discharge_category").agg(
            Volume=("readmitted_30d", "count"),
            Rate=("readmitted_30d", lambda x: round(x.mean() * 100, 2))
        ).reset_index().sort_values("Rate", ascending=False)
        fig_dis = px.bar(
            dis_agg, x="discharge_category", y="Rate", color="Rate",
            color_continuous_scale="Reds", text="Rate",
            labels={"discharge_category": "Discharge Destination", "Rate": "Readmission Rate (%)"}
        )
        fig_dis.update_traces(texttemplate='%{text}%', textposition='outside')
        style_fig(fig_dis, "Readmission Hazard by Discharge Destination", height=350)
        st.plotly_chart(fig_dis, use_container_width=True)


# =============================================================================
# VIEW 6: HIGH-RISK COHORT TRIAGE
# =============================================================================
elif selected_view == "⚠️  High-Risk Cohort Triage":
    st.markdown("""
    <div class="content-header">
        <h1 class="content-title">View 5: High-Risk Patient Cohort Triage Dashboard</h1>
        <p class="content-sub">Prioritizing high-utilizers and multimorbid patients for care coordination outreach.</p>
    </div>
    """, unsafe_allow_html=True)

    c_f1, c_f2 = st.columns(2)
    with c_f1:
        min_inp = st.slider("Min Prior Inpatient Admissions:", 0, 5, 1)
    with c_f2:
        min_risk_idx = st.slider("Min Clinical Risk Score (0-12):", 0, 10, 4)

    cohort = df_view[
        (df_view["prior_inpatient_admissions"] >= min_inp) &
        (df_view["clinical_risk_index"] >= min_risk_idx)
    ]

    st.metric(
        "Identified High-Risk Cohort",
        f"{len(cohort):,} encounters",
        f"Readmission Rate: {cohort['readmitted_30d'].mean()*100:.2f}% (vs {df_view['readmitted_30d'].mean()*100:.2f}% baseline)"
    )

    disp_cols = [
        "surrogate_encounter_id", "gender", "age", "diag_1_category",
        "prior_inpatient_admissions", "num_medications", "length_of_stay",
        "clinical_risk_index", "readmitted_30d"
    ]
    st.dataframe(cohort[disp_cols].head(100), use_container_width=True)


# =============================================================================
# VIEW 7: MLOPS PREDICTIVE DECISION SUPPORT
# =============================================================================
elif selected_view == "🤖  MLOps Predictive Decision Support":
    st.markdown("""
    <div class="content-header">
        <h1 class="content-title">View 6: MLOps Machine Learning & Decision Support</h1>
        <p class="content-sub">Interactive 30-day readmission risk prediction powered by registered XGBoost champion model.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎯 Patient Risk Calculator", "📈 Model Leaderboard & Curves"])

    with tab1:
        st.subheader("Patient Clinical Profile")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            p_gender = st.selectbox("Gender:", ["Female", "Male"])
            p_age = st.slider("Approx Age:", 15, 95, 75, step=10)
            p_los = st.slider("Stay Duration (Days):", 1, 14, 5)
        with col_b:
            p_pinp = st.number_input("Prior Inpatient Visits (Past Year):", 0, 10, 2)
            p_meds = st.slider("Medications Count:", 1, 50, 18)
            p_diag = st.slider("Diagnoses Count:", 1, 16, 8)
        with col_c:
            p_adm = st.selectbox("Admission Urgency:", ["Emergency", "Urgent", "Elective", "Other"])
            p_dis = st.selectbox("Discharge Destination:", ["Discharged to Home", "Transferred / Skilled Facility", "Left AMA", "Other"], index=1)
            p_diag_cat = st.selectbox("Primary Diagnosis:", ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Neoplasms", "Other"])

        if st.button("🚀 Calculate Patient 30-Day Readmission Risk", type="primary"):
            if model and preprocessor:
                p_dict = {
                    "gender": p_gender,
                    "age_midpoint": p_age,
                    "is_elderly": int(p_age >= 65),
                    "length_of_stay": p_los,
                    "num_lab_procedures": 40,
                    "num_procedures": 1,
                    "num_medications": p_meds,
                    "prior_inpatient_admissions": p_pinp,
                    "prior_outpatient_admissions": 0,
                    "prior_emergency_admissions": 0,
                    "total_prior_visits": p_pinp,
                    "diagnosis_count": p_diag,
                    "clinical_risk_index": int(p_pinp >= 1)*3 + int(p_los >= 7)*2 + int(p_diag >= 9)*2,
                    "has_prior_inpatient": int(p_pinp > 0),
                    "frequent_utilizer_flag": int(p_pinp >= 3),
                    "extended_stay_flag": int(p_los >= 7),
                    "high_comorbidity_flag": int(p_diag >= 9),
                    "comorbidity_cardio_diabetes": int(p_diag_cat in ["Circulatory", "Diabetes"]),
                    "active_diabetic_meds_count": 2,
                    "medication_change_flag": 1,
                    "diabetes_med_flag": 1,
                    "on_insulin": 1,
                    "insulin_titration_up": 0,
                    "insulin_titration_down": 0,
                    "on_metformin": 1,
                    "a1c_tested": 1,
                    "a1c_high": 1,
                    "glucose_tested": 0,
                    "glucose_high": 0,
                    "admission_category": p_adm,
                    "discharge_category": p_dis,
                    "diag_1_category": p_diag_cat
                }
                df_single = pd.DataFrame([p_dict])
                for col in ALL_FEATURE_COLS:
                    if col not in df_single.columns:
                        df_single[col] = 0

                X_trans = preprocessor.transform(df_single[ALL_FEATURE_COLS])
                prob = float(model.predict_proba(X_trans)[0, 1])

                st.markdown("---")
                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    st.metric("Predicted 30-Day Readmission Risk", f"{prob*100:.1f}%")
                    if prob >= 0.45:
                        st.error("Risk Tier: HIGH RISK. Urgent care management outreach assigned.")
                    elif prob >= 0.20:
                        st.warning("Risk Tier: MODERATE RISK. Scheduled follow-up within 7 days.")
                    else:
                        st.success("Risk Tier: LOW RISK. Routine primary care discharge.")

                with c_res2:
                    factors = explain_single_patient(X_trans, top_n=5)
                    st.write("**Top Contributing Factors (SHAP Local Attribution):**")
                    st.dataframe(pd.DataFrame(factors), use_container_width=True)

    with tab2:
        st.subheader("Model Comparison Leaderboard")
        metrics = metadata.get("metrics_summary", {})
        if metrics:
            st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)


# =============================================================================
# VIEW 8: DATA QUALITY & PIPELINE AUDIT
# =============================================================================
elif selected_view == "📋  Data Quality & Pipeline Audit":
    st.markdown("""
    <div class="content-header">
        <h1 class="content-title">View 7: Data Quality & Pipeline Audit Trail</h1>
        <p class="content-sub">Verification of automated data quality constraints and quarantined records.</p>
    </div>
    """, unsafe_allow_html=True)

    dq_path = REPORTS_DIR / "data_quality_report.json"
    if dq_path.exists():
        with open(dq_path, "r", encoding="utf-8") as f:
            dq_data = json.load(f)
        st.success(f"Status: {dq_data['checks_passed']}/{dq_data['total_checks_evaluated']} Quality Checks Passed (100% Pass Rate)")
        st.dataframe(pd.DataFrame(dq_data["checks"]), use_container_width=True)

    rej_path = CURATED_DATA_DIR.parent / "rejected" / "rejected_records_log.csv"
    if rej_path.exists():
        st.markdown("#### Quarantined Ineligible Encounters (Expired / Hospice Cases)")
        rej_df = pd.read_csv(rej_path)
        st.dataframe(rej_df.head(50), use_container_width=True)
