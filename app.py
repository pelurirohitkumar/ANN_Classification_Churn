import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e6f0 !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #1a0a3a 0%, #0a0a0f 60%) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px !important; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 2px;
    background: linear-gradient(90deg, transparent, #7c3aed, #a78bfa, #7c3aed, transparent);
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    color: #7c3aed;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #6b6880;
    letter-spacing: 0.05em;
}

/* ── Section Label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    color: #7c3aed;
    text-transform: uppercase;
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1a2e;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: #7c3aed;
    border-radius: 50%;
}

/* ── Cards ── */
.card {
    background: #111118;
    border: 1px solid #1e1a2e;
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #7c3aed40, transparent);
}

/* ── Streamlit widget overrides ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input,
[data-testid="stSlider"] {
    font-family: 'DM Mono', monospace !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > label,
[data-testid="stNumberInput"] > label,
[data-testid="stSlider"] > label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    color: #9d8fc4 !important;
    text-transform: uppercase !important;
    margin-bottom: 0.35rem !important;
}

[data-testid="stSelectbox"] > div > div {
    background: #0d0d16 !important;
    border: 1px solid #2a2440 !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px #7c3aed20 !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    background: #0d0d16 !important;
    border: 1px solid #2a2440 !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.85rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px #7c3aed20 !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: #7c3aed !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #a78bfa !important;
    border: 2px solid #7c3aed !important;
    box-shadow: 0 0 12px #7c3aed60 !important;
}

/* Column gaps */
[data-testid="column"] { padding: 0 0.6rem !important; }

/* ── Result Panel ── */
.result-panel {
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 0.5rem;
    position: relative;
    overflow: hidden;
}
.result-safe {
    background: linear-gradient(135deg, #0a1f0a, #0d2b0d);
    border: 1px solid #1a4a1a;
}
.result-risk {
    background: linear-gradient(135deg, #1f0a0a, #2b0d0d);
    border: 1px solid #4a1a1a;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    opacity: 0.7;
}
.result-verdict {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.result-safe .result-verdict { color: #4ade80; }
.result-risk .result-verdict { color: #f87171; }

/* ── Icon ── */
.result-icon {
    font-size: 3rem;
    margin-bottom: 0.75rem;
    display: block;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #1e1a2e;
    margin: 2rem 0;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    box-shadow: 0 4px 24px #7c3aed40 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #8b4cf7, #7c3aed) !important;
    box-shadow: 0 6px 32px #7c3aed60 !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & encoders ──────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('one_hot_encoder_geography.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

model, label_encoder_gender, one_hot_encoder_geography, scaler = load_assets()


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Neural Churn Analysis</div>
    <div class="hero-title">Churn Intelligence</div>
    <div class="hero-sub">Deep learning · Real-time prediction · Customer retention</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Layout: inputs left, result right ─────────────────────────────────────────
left, right = st.columns([1.6, 1], gap="large")

with left:
    # ── Section 1: Demographics ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        geography = st.selectbox('Geography', one_hot_encoder_geography.categories_[0])
    with c2:
        gender = st.selectbox('Gender', label_encoder_gender.classes_)

    c3, c4 = st.columns(2)
    with c3:
        age = st.slider('Age', 18, 92, 35)
    with c4:
        tenure = st.slider('Tenure (years)', 0, 10, 3)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Section 2: Financials ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Financial Details</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
    with f2:
        balance = st.number_input('Balance', min_value=0.0, value=60000.0, step=500.0)
    with f3:
        estimated_salary = st.number_input('Est. Salary', min_value=0.0, value=50000.0, step=1000.0)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Section 3: Account ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Account Activity</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        num_of_products = st.slider('Products', 1, 4, 1)
    with a2:
        has_cr_card = st.selectbox('Credit Card', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    with a3:
        is_active_member = st.selectbox('Active Member', [0, 1], format_func=lambda x: 'Yes' if x else 'No')

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button('⚡  Run Churn Analysis')


# ── Right panel: result ────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

    if predict_btn:
        # ── UNCHANGED LOGIC ──────────────────────────────────────────────────
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })
        geo_encoded = one_hot_encoder_geography.transform([[geography]])
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        # ── END LOGIC ────────────────────────────────────────────────────────

        is_churn = prediction_proba > 0.5
        panel_cls = "result-risk" if is_churn else "result-safe"
        verdict   = "The customer is likely to churn." if is_churn else "The customer is not likely to churn."
        icon      = "⚠️" if is_churn else "✅"

        st.markdown(f"""
        <div class="result-panel {panel_cls}">
            <div class="result-label">Model Verdict</div>
            <span class="result-icon">{icon}</span>
            <div class="result-verdict">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div class="result-panel" style="background:#111118;border:1px dashed #2a2440;min-height:360px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.75rem;">
            <div style="font-size:2.5rem;opacity:0.15">⚡</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#3d3a52;text-align:center;letter-spacing:0.1em;text-transform:uppercase;">
                Fill in the customer details<br>and run the analysis
            </div>
        </div>
        """, unsafe_allow_html=True)