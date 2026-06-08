import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ─── Page Configurations (MUST be first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="CardioPredict Analytics",
    page_icon="🫀",
    layout="centered"
)

# ─── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_pickle_model():
    with open('cardio_rf_model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_pickle_model()
except FileNotFoundError:
    st.error("Error: 'cardio_rf_model.pkl' not found. Please ensure the model file is in the root directory.")
    st.stop()

# ─── Professional Dashboard Stylesheet ─────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f111a; color: #e2e8f0; }
    h1, h2, h3, h4 { font-family: 'Inter', system-ui, sans-serif; font-weight: 600; letter-spacing: -0.02em; }
    h1 { color: #f8fafc !important; font-size: 2rem !important; margin-bottom: 0.2rem !important; }

    hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }

    .metric-row { display: flex; gap: 14px; margin: 1rem 0; }
    .metric-box {
        flex: 1;
        background: #161925;
        border: 1px solid #23293e;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-val { font-size: 1.3rem; font-weight: 700; color: #f1f5f9; }
    .metric-lbl { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

    .gauge-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 2rem 0;
    }
    .gauge-outer {
        position: relative;
        width: 240px;
        height: 120px;
        overflow: hidden;
    }
    .gauge-bg {
        position: absolute;
        top: 0; left: 0;
        width: 240px; height: 240px;
        border-radius: 50%;
        background: conic-gradient(
            from 180deg,
            #10b981 0deg,
            #eab308 90deg,
            #ef4444 180deg,
            #1e293b 180deg
        );
    }
    .gauge-inner-mask {
        position: absolute;
        top: 30px; left: 30px;
        width: 180px; height: 180px;
        border-radius: 50%;
        background: #0f111a;
    }
    .gauge-score {
        position: absolute;
        bottom: 4px; left: 0; width: 240px;
        text-align: center;
        font-size: 2rem; font-weight: 700; color: #f1f5f9;
    }
    .gauge-needle {
        position: absolute;
        bottom: 0; left: 50%;
        width: 4px; height: 110px;
        background: linear-gradient(to top, #f8fafc, #94a3b8);
        border-radius: 4px 4px 0 0;
        transform-origin: bottom center;
        margin-left: -2px;
        transition: transform 0.6s cubic-bezier(.4,0,.2,1);
    }
    /* FIX 1: bottom:0 (was -5px) — keeps pivot inside overflow:hidden container */
    .gauge-pivot {
        position: absolute;
        bottom: 0; left: 50%;
        width: 12px; height: 12px;
        background: #f8fafc;
        border-radius: 50%;
        margin-left: -6px;
    }

    .panel-out { border-radius: 8px; padding: 1.5rem; border-left: 4px solid; margin-bottom: 1.5rem; background: #161925; }
    .panel-out.high { border-left-color: #ef4444; }
    .panel-out.low  { border-left-color: #10b981; }

    label { color: #94a3b8 !important; font-size: 0.85rem !important; font-weight: 500; }
    .stButton > button {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { background: #2563eb !important; }

    .footer { text-align: center; font-size: 0.75rem; color: #475569; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #1e293b; }
</style>
""", unsafe_allow_html=True)

# ─── App Header ────────────────────────────────────────────────────────────────
st.markdown("<h1>CardioPredict AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; margin:0;'>Smart Cardio Risk Predictor</p>", unsafe_allow_html=True)
st.divider()

# ─── Section 1: Demographics & Biometrics ──────────────────────────────────────
st.markdown("### Patient Demographics & Physical Metrics")
col1, col2 = st.columns(2)

with col1:
    age    = st.number_input("Age (Years)", min_value=10, max_value=100, value=40)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)

with col2:
    gender = st.selectbox("Biological Sex Assigned at Birth", [1, 2],
                          format_func=lambda x: "Female" if x == 1 else "Male")
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)

# BMI — WHO standard categories
bmi = round(weight / ((height / 100) ** 2), 1)
bmi_cat = (
    "Underweight" if bmi < 18.5 else
    "Normal"      if bmi < 25   else
    "Overweight"  if bmi < 30   else
    "Obese"
)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-box"><div class="metric-val">{bmi}</div><div class="metric-lbl">Body Mass Index (BMI)</div></div>
    <div class="metric-box"><div class="metric-val">{bmi_cat}</div><div class="metric-lbl">Weight Classification</div></div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Section 2: Clinical Vitals ────────────────────────────────────────────────
st.markdown("### Hemodynamic Values & Laboratory Tiers")
col3, col4 = st.columns(2)

with col3:
    ap_hi       = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60, max_value=250, value=120)
    cholesterol = st.selectbox("Cholesterol Status Profile", [1, 2, 3],
                               format_func=lambda x: {1: "Normal", 2: "Elevated", 3: "Significantly Elevated"}[x])

with col4:
    ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=200, value=80)
    gluc  = st.selectbox("Blood Glucose Level Status", [1, 2, 3],
                         format_func=lambda x: {1: "Normal", 2: "Elevated", 3: "Significantly Elevated"}[x])

# Pulse pressure: normal ~40 mmHg; >60 = arterial stiffness / CVD risk
pulse_pressure = ap_hi - ap_lo

# AHA BP classification — both systolic and diastolic checked
if ap_hi < 120 and ap_lo < 80:
    bp_status = "Normal"
elif ap_hi < 130 and ap_lo < 80:
    bp_status = "Elevated"
elif ap_hi < 140 and ap_lo < 90:
    bp_status = "Stage 1 Hypertension"
else:
    bp_status = "Stage 2 Hypertension"

st.markdown(f"""
<div class="metric-row">
    <div class="metric-box"><div class="metric-val">{ap_hi}/{ap_lo}</div><div class="metric-lbl">Blood Pressure</div></div>
    <div class="metric-box"><div class="metric-val">{bp_status}</div><div class="metric-lbl">Clinical Classification</div></div>
    <div class="metric-box"><div class="metric-val">{pulse_pressure} mmHg</div><div class="metric-lbl">Pulse Pressure</div></div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Section 3: Behavioral Indicators ──────────────────────────────────────────
st.markdown("### Behavioral Lifestyle Matrix")
col5, col6, col7 = st.columns(3)

with col5:
    smoke  = st.selectbox("Tobacco Smoking History",   [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col6:
    alco   = st.selectbox("Regular Alcohol Intake",    [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col7:
    active = st.selectbox("Physically Active Regimen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# ─── Analysis Pipeline Trigger ─────────────────────────────────────────────────
if st.button("Evaluate Cardiovascular Profile Matrix", use_container_width=True):

    input_df = pd.DataFrame([[
        age, gender, height, weight, ap_hi, ap_lo,
        cholesterol, gluc, smoke, alco, active, bmi
    ]], columns=[
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi'
    ])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    risk_score  = round(probability[1] * 100, 1)
    safe_score  = round(probability[0] * 100, 1)

    st.markdown("<h3 style='text-align: center; margin-bottom: 0;'>Calculated Patient Cardiovascular Risk</h3>",
                unsafe_allow_html=True)

    # Needle: 0% = -90deg (left), 100% = +90deg (right), 180deg total sweep
    rotation_angle = -90 + (risk_score * 1.8)

    st.markdown(f"""
    <div class="gauge-wrapper">
        <div class="gauge-outer">
            <div class="gauge-bg"></div>
            <div class="gauge-inner-mask"></div>
            <div class="gauge-score">{risk_score}%</div>
            <div class="gauge-needle" style="transform: rotate({rotation_angle}deg);"></div>
            <div class="gauge-pivot"></div>
        </div>
        <div style="display:flex; width:240px; justify-content:space-between;
                    font-size:0.75rem; color:#64748b; margin-top:8px; font-weight:600;">
            <span>LOW RISK</span>
            <span>HIGH RISK</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── HIGH RISK OUTPUT ──────────────────────────────────────────────────────
    if prediction == 1:
        st.markdown(f"""
        <div class="panel-out high">
            <h4 style="color:#ef4444; margin:0 0 6px;">Elevated Clinical Risk Profile Found</h4>
            <p style="color:#94a3b8; margin:0; font-size:0.9rem;">
                The mathematical evaluation engine matched biometric records with elevated indicators of
                cardiovascular disease risk markers at <strong>{risk_score}% probability</strong>.
                Further clinical validation is suggested.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Primary Pathological Drivers Identified:")
        drivers_found = False

        if age > 55:
            st.error(f"Patient age ({age} yrs) exceeds the 55-year threshold — a well-established independent cardiovascular risk marker.")
            drivers_found = True
        if ap_hi >= 140 or ap_lo >= 90:
            st.error(f"Blood pressure ({ap_hi}/{ap_lo} mmHg) meets Stage 2 Hypertension criteria.")
            drivers_found = True
        elif ap_hi >= 130 or ap_lo >= 80:
            st.warning(f"Blood pressure ({ap_hi}/{ap_lo} mmHg) registers within Stage 1 Hypertension range.")
            drivers_found = True
        if pulse_pressure > 60:
            st.warning(f"Pulse pressure ({pulse_pressure} mmHg) is elevated — values above 60 mmHg indicate increased arterial stiffness and CVD risk.")
            drivers_found = True
        if cholesterol >= 2:
            st.warning("Serum Cholesterol profiles register above established standard reference ranges.")
            drivers_found = True
        if gluc >= 2:
            st.warning("Blood Glucose levels are elevated — a significant metabolic cardiovascular risk indicator.")
            drivers_found = True
        if bmi >= 30:
            st.error(f"Body Mass Index ({bmi}) falls in the Obese range — substantially elevates cardiac load.")
            drivers_found = True
        elif bmi >= 25:
            st.warning(f"Body Mass Index ({bmi}) registers within the Overweight range.")
            drivers_found = True
        if smoke == 1:
            st.error("Documented active tobacco intake scales vascular risk vectors significantly.")
            drivers_found = True
        if alco == 1:
            st.warning("Regular alcohol intake is associated with elevated blood pressure and cardiac arrhythmia risk.")
            drivers_found = True
        if active == 0:
            st.warning("Sedentary lifestyle detected — physical inactivity is an independent cardiovascular risk factor.")
            drivers_found = True

        if not drivers_found:
            st.info("Risk elevation detected from a combination of borderline indicators rather than a single dominant factor. Clinical review is recommended.")

    # ── LOW RISK OUTPUT ───────────────────────────────────────────────────────
    else:
        st.markdown(f"""
        <div class="panel-out low">
            <h4 style="color:#10b981; margin:0 0 6px;">Optimal Status Reference Parameters Met</h4>
            <p style="color:#94a3b8; margin:0; font-size:0.9rem;">
                Biometric inputs comfortably locate the profile within safe reference bounds,
                generating a <strong>{safe_score}% homeostatic baseline rating</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Protective Biological Indicators:")
        protective_found = False

        if ap_hi < 120 and ap_lo < 80:
            st.success("Blood pressure is within optimal homeostatic guidelines.")
            protective_found = True
        if pulse_pressure <= 40:
            st.success(f"Pulse pressure ({pulse_pressure} mmHg) is within the healthy range — good arterial flexibility.")
            protective_found = True
        if 18.5 <= bmi < 25:
            st.success(f"Body Mass Index ({bmi}) tracks within the healthy Normal range.")
            protective_found = True
        # FIX 2: age <= 55 as protective indicator
        if age <= 55:
            st.success(f"Patient age ({age} yrs) is within the lower cardiovascular risk demographic.")
            protective_found = True
        if smoke == 0:
            st.success("No tobacco use history — significantly reduces vascular stress markers.")
            protective_found = True
        if active == 1:
            st.success("Regular physical activity is a strong protective factor for cardiovascular health.")
            protective_found = True
        if alco == 0:
            st.success("Absence of regular alcohol intake supports stable blood pressure baselines.")
            protective_found = True
        if gluc == 1:
            st.success("Blood glucose levels within normal range — metabolic risk well controlled.")
            protective_found = True
        # FIX 2: normal cholesterol as protective indicator
        if cholesterol == 1:
            st.success("Serum cholesterol within normal reference range — reduced atherosclerosis risk.")
            protective_found = True

        # FIX 3: fallback if no protective indicator fires
        if not protective_found:
            st.info("Profile falls within low-risk classification based on overall pattern. Individual metrics are near threshold — maintaining current health practices is advised.")

    st.divider()
    st.caption("Disclaimer: This application functions as a data engineering prototype and decision-support simulation framework. It does not replace medical diagnostics or therapeutic consultation.")

# ─── UI Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CardioPredict Platform
</div>
""", unsafe_allow_html=True)