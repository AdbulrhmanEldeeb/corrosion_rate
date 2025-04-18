import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from utils.predictor import CorrosionClassifier
from utils.processors import  build_final_input
from utils.vars import environment, uns_nums
from config.config import SIDEBAR_IMAGE,PAGE_ICON

st.set_page_config(page_title="Corrosion Classifier", layout="wide", page_icon=PAGE_ICON)

clf = CorrosionClassifier()

# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.image(SIDEBAR_IMAGE, use_container_width=True)
    st.markdown("## 🧪 Corrosion Rate Predictor")
    st.markdown("Predict the **corrosion rate** based on material and environment conditions.")
    st.markdown("---")
    st.markdown("🔬 Powered by ML | 📊 PCA | 🧠 SciBERT")

# ------------------------ Page Header ------------------------
st.markdown("<h1 style='text-align: center;'>🔍 Corrosion Rate Prediction with AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Fill in the details below to predict the corrosion rate of a material.</p>", unsafe_allow_html=True)

# ------------------------ Input Form ------------------------
with st.form("corrosion_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        env = st.selectbox("🌍 Environment", options=environment,
                           help="Select the surrounding medium (e.g., seawater, acidic, etc.)")
        temp = st.number_input("🌡️ Temperature (°C)", step=1, value=25,
                               help="Temperature of the environment in Celsius")

    with col2:
        conc = st.number_input("🧪 Concentration (%)", min_value=0, max_value=100, value=50,
                               help="Concentration of the surrounding medium as a percentage")

    with col3:
        uns_input = st.selectbox("🧬 Alloy UNS", options=uns_nums,
                                 help="Select the Unified Numbering System (UNS) alloy code")

    comment = st.text_area(
        "💬 Describe the Condition in details",
        height=120,
        placeholder="e.g. acidic environment with high humidity..",
        help="Describe the environmental condition"
    )

    submitted = st.form_submit_button("🚀 Predict corrosion rate")

# ------------------------ Prediction & Output ------------------------
if submitted:
    prediction, _ = clf.predict(env, temp, conc, uns_input, comment)

    st.markdown("## 🗞 Prediction Result")
    st.success(f"✅ **Predicted Corrosion Class**: `{prediction}`")

    # Prepare raw input + prediction for download
    raw_input = pd.DataFrame([{
        "Environment": env,
        "Temperature (°C)": temp,
        "Concentration (%)": conc,
        "Alloy UNS": uns_input,
        "Condition Description": comment,
        "Predicted Corrosion Rate": prediction
    }])

    # Download raw input + prediction
    csv_bytes = raw_input.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Input + Prediction as CSV",
        data=csv_bytes,
        file_name="corrosion_prediction.csv",
        mime="text/csv",
    )

    st.markdown("---")

# ------------------------ Footer ------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("💪 Built with Streamlit | 🧠 Machine Learning | 👨‍🔬 SciBERT + PCA Model")
