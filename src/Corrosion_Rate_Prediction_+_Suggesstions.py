import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from utils.predictor import CorrosionClassifier
from utils.processors import build_final_input, remove_think_tags
from utils.vars import environment, uns_nums
from config.config import SIDEBAR_IMAGE, PAGE_ICON
from chat.chat import invoke_llm, get_main_prompt

st.set_page_config(
    page_title="Corrosion Rate Predictor", layout="wide", page_icon=PAGE_ICON
)

clf = CorrosionClassifier()

# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.image(SIDEBAR_IMAGE, use_container_width=True)
    st.markdown("## 🧪 Corrosion Rate Predictor")
    st.markdown(
        "Predict the **corrosion rate** based on material and environment conditions."
    )
    st.markdown("---")
    st.markdown("🔬 Powered by ML | 📊 PCA | 🧠 SciBERT")

# ------------------------ Page Header ------------------------
st.markdown(
    "<h1 style='text-align: center;'>🔍 Corrosion Rate Prediction with AI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Fill in the details below to predict the corrosion rate of a material.</p>",
    unsafe_allow_html=True,
)

# ------------------------ Input Form ------------------------
with st.form("corrosion_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        env = st.selectbox(
            "🌍 Environment",
            options=environment,
            help="Select the surrounding medium (e.g., seawater, acidic, etc.)",
        )
        temp = st.number_input(
            "🌡️ Temperature (°C)",
            step=1,
            value=25,
            help="Temperature of the environment in Celsius",
        )

    with col2:
        conc = st.number_input(
            "🧪 Concentration (%)",
            min_value=0,
            max_value=100,
            value=50,
            help="Concentration of the surrounding medium as a percentage",
        )

    with col3:
        uns_input = st.selectbox(
            "🧬 Alloy UNS",
            options=uns_nums,
            help="Select the Unified Numbering System (UNS) alloy code",
        )

    comment = st.text_area(
        "💬 Describe the Condition in details",
        height=120,
        placeholder="e.g. acidic environment with high humidity..",
        help="Describe the environmental condition",
    )

    submitted = st.form_submit_button("🚀 Predict corrosion rate")

if "prediction_data" in st.session_state:
    st.markdown("## 🗞 Prediction Result")
    st.success(
        f"✅ Predicted Corrosion Rate: **{st.session_state.prediction_data['Predicted Corrosion Rate'][0]}**"
    )

    st.markdown("### 🧠 AI Recommendations for Corrosion Control")
    st.markdown(st.session_state.llm_output)

    # CSV Download
    csv_bytes = st.session_state.prediction_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Input, Prediction and Recommendations as CSV",
        data=csv_bytes,
        file_name="corrosion_prediction.csv",
        mime="text/csv",
    )

    # TXT Download
    txt_content = "Corrosion Prediction Report\n\n"
    txt_content += "Input Parameters:\n"
    for col in st.session_state.prediction_data.columns:
        if col != "AI Recommendations":
            value = st.session_state.prediction_data[col].values[0]
            txt_content += f"{col}: {value}\n"
    txt_content += "\nAI Recommendations:\n"
    txt_content += st.session_state.llm_output
    txt_bytes = txt_content.encode("utf-8")

    st.download_button(
        label="📄 Download AI Recommendations as TXT",
        data=txt_bytes,
        file_name="corrosion_recommendations.txt",
        mime="text/plain",
    )

# ------------------------ Prediction & Output ------------------------
if submitted:
    prediction, _ = clf.predict(env, temp, conc, uns_input, comment)
    raw_input = pd.DataFrame(
        [
            {
                "Environment": env,
                "Temperature (°C)": temp,
                "Concentration (%)": conc,
                "Alloy UNS": uns_input,
                "Condition Description": comment,
                "Predicted Corrosion Rate": prediction,
            }
        ]
    )
    main_page_prompt = get_main_prompt(raw_input)
    llm_output = invoke_llm(main_page_prompt)
    llm_output = remove_think_tags(llm_output)
    raw_input["AI Recommendations"] = llm_output

    # Store in session state
    st.session_state.prediction_data = raw_input
    st.session_state.llm_output = llm_output

# ------------------------ Footer ------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("💪 Built with Streamlit | 🧠 Machine Learning | 👨‍🔬 SciBERT + PCA Model")
