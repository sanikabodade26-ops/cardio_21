import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ─── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_pickle_model():
    # Safely reads the serialized backend model matrix
    with open('cardio_rf_model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_pickle_model()
except FileNotFoundError:
    st.error("Error: 'cardio_rf_model.pkl' not found. Please ensure the model file is in the root directory.")
    st.stop()

# ─── Page Configurations ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict Analytics",
    page_icon="🫀",
    layout="centered"
)

# ─── Professional Dashboard Stylesheet ──────────────────────────────────────────
st.markdown("""
<style>
    /* Global Canvas Styling */
    .stApp { background-color: #0f111a; color: #e2e8f0; }
    h1, h2, h3, h4 { font-family: 'Inter', system-ui, sans-serif; font-weight: 600; letter-spacing: -0.02em; }
    h1 { color: #f8fafc !important; font-size: 2rem !important; margin-bottom: 0.2rem !important; }
    
    /* Document Dividers */
    hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }
    
    /* Metric Display Blocks */
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

    /* Custom Modern Gauge System */
    .gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 2rem 0;
    }
    .gauge-track {
        position: relative;
        width: 240px;
        height: 120px;
        background: #1e293b;
        border-radius: 120px 120px 0 0;
        overflow: hidden;
    }
    .gauge-fill {
        position: absolute;
        top: 0; left: 0;
        width: 240px; height: 240px;
        border-radius: 50%;
        background: conic-gradient(from 270deg, #10b981 0%, #eab308 50%, #ef4444 100%);
        transform-origin: center center;
    }
    .gauge-cover {
        position: absolute;
        top: 20px; left: 20px;
        width: 200px; height: 100px;
        background: #0f111a;
        border-radius: 100px 100px 0 0;
        display: flex;
        align-items: flex-end;
        justify-content: center;
        padding-bottom: 10px;
    }
    .gauge-needle {
        position: absolute;
        bottom: 0; left: 50%;
        width: 6px; height: 90px;
        background: #f8fafc;
        border-radius: 4px;
        transform-origin: bottom center;
        margin-left: -3px;
        transition: transform 0.5s ease-in-out;
    }

    /* Output Results Clinical Panels */
    .panel-out { border-radius: 8px; padding: 1.5rem; border-left: 4px solid; margin-bottom: 1.5rem; background: #161925; }
    .panel-out.high { border-left-color: #ef4444; }
    .panel-out.low { border-left-color: #10b981; }
    
    /* Interactive Component Overrides */
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
    age = st.number_input("Age (Years)", min_value=10, max_value=100, value=40)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)

with col2:
    gender = st.selectbox("Biological Sex Assigned at Birth", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)

# Real-time Mathematical Baselines
bmi = round(weight / ((height / 100) ** 2), 1)
bmi_cat = "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"

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
    ap_hi = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60, max_value=250, value=120)
    cholesterol = st.selectbox("Cholesterol Status Profile", [1, 2, 3], 
                               format_func=lambda x: {1: "Normal", 2: "Elevated", 3: "Significantly Elevated"}[x])

with col4:
    ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=200, value=80)
    gluc = st.selectbox("Blood Glucose Level Status", [1, 2, 3], 
                        format_func=lambda x: {1: "Normal", 2: "Elevated", 3: "Significantly Elevated"}[x])

pulse_pressure = ap_hi - ap_lo
bp_status = "Normal" if ap_hi < 120 else "Elevated" if ap_hi < 130 else "Stage 1 Hypertension" if ap_hi < 140 else "Stage 2 Hypertension"

st.markdown(f"""
<div class="metric-row">
    <div class="metric-box"><div class="metric-val">{ap_hi}/{ap_lo}</div><div class="metric-lbl">Blood Pressure</div></div>
    <div class="metric-box"><div class="metric-val">{bp_status}</div><div class="metric-lbl">Clinical Classification</div></div>
    <div class="metric-box"><div class="metric-val">{pulse_pressure}</div><div class="metric-lbl">Pulse Pressure</div></div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Section 3: Behavioral Indicators ──────────────────────────────────────────
st.markdown("### Behavioral Lifestyle Matrix")
col5, col6, col7 = st.columns(3)

with col5:
    smoke = st.selectbox("Tobacco Smoking History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col6:
    alco = st.selectbox("Regular Alcohol Intake", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col7:
    active = st.selectbox("Physically Active Regimen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# ─── Analysis Pipeline Trigger ─────────────────────────────────────────────────
if st.button("Evaluate Cardiovascular Profile Matrix", use_container_width=True):

    # Process structured vector for prediction array mapping
    input_df = pd.DataFrame([[
        age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi
    ]], columns=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi'])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    risk_score = round(probability[1] * 100, 1)

    st.markdown("<h3 style='text-align: center; margin-bottom: 0;'>Calculated Patient Cardiovascular Risk</h3>", unsafe_allow_html=True)
    
    # Calculate needle rotation angle based on the probability score (0% = -90deg, 100% = 90deg)
    rotation_angle = -90 + (risk_score * 1.8)

    # Injecting Custom Animated Gauge Matrix Component
    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-track">
            <div class="gauge-fill"></div>
            <div class="gauge-cover">
                <span style="font-size: 2rem; font-weight: 700; color: #f1f5f9;">{risk_score}%</span>
            </div>
            <div class="gauge-needle" style="transform: rotate({rotation_angle}deg);"></div>
        </div>
        <div style="display: flex; width: 240px; justify-content: space-between; font-size: 0.75rem; color: #64748b; margin-top: 8px; font-weight: 600;">
            <span>LOW RISK</span>
            <span>HIGH RISK</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="panel-out high">
            <h4 style="color:#ef4444; margin:0 0 6px;">Elevated Clinical Risk Profile Found</h4>
            <p style="color:#94a3b8; margin:0; font-size:0.9rem;">
                The mathematical evaluation engine matched biometric records with elevated indicators of cardiovascular disease risk markers at <strong>{risk_score}% probability</strong>. Further clinical validation is suggested.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Primary Pathological Drivers Identified:")
        if ap_hi >= 140:
            st.error(f"Systolic Blood Pressure measurements ({ap_hi} mmHg) signify Stage 2 Hypertension baselines.")
        if cholesterol >= 2:
            st.warning("Serum Cholesterol profiles register above established standard reference ranges.")
        if bmi >= 25:
            st.warning(f"Calculated Body Mass Index profile data ({bmi}) registers within the {bmi_cat} range parameters.")
        if smoke == 1:
            st.error("Documented active tobacco intake scales risk vectors exponentially.")

    else:
        st.markdown(f"""
        <div class="panel-out low">
            <h4 style="color:#10b981; margin:0 0 6px;">Optimal Status Reference Parameters Met</h4>
            <p style="color:#94a3b8; margin:0; font-size:0.9rem;">
                Biometric inputs comfortably locate the profile within safe reference bounds, generating a <strong>{100 - risk_score}% homeostatic baseline rating</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Protective Biological Indicators:")
        if ap_hi < 120:
            st.success("Patient blood pressure is completely within optimal homeostatic guidelines.")
        if bmi < 25:
            st.success("Body Mass Index metrics track perfectly within targeted standard zones.")
        if smoke == 0:
            st.success("Negative historical smoking status drastically drops profile vascular stress values.")

    st.divider()
    st.caption("Disclaimer: This application functions as a data engineering prototype and decision-support simulation framework. It does not replace medical diagnostics or therapeutic consultation.")

# ─── UI Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CardioPredict Platform · Core Academic Project Integration · Built using Streamlit Engine + scikit-learn Framework
</div>
""", unsafe_allow_html=True)