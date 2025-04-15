import streamlit as st
import pandas as pd
import joblib
from utils.processors import clean_condition_text
from utils.vars import environment, targets
import os

# Page config
st.set_page_config(page_title="Corrosion Classifier", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.image("https://www.ddcoatings.co.uk/wp-content/uploads/2019/09/pipeline-corrosion.jpg", use_container_width=True)
    st.markdown("## 🧪 Corrosion Classifier")
    st.markdown("Predict the **corrosion class** based on material and environment conditions.")
    st.markdown("---")
    st.markdown("🔬 Powered by ML | 📊 PCA | 🌐 TF-IDF")

# --- Load Models & Encoders ---
base_path = "src"
pca = joblib.load(os.path.join(base_path, "decomposers", "pca_model.pkl"))
vectorizer = joblib.load(os.path.join(base_path, "vectorizers", "tfidf_vectorizer.pkl"))
model = joblib.load(os.path.join(base_path, "classifiers", "tunned_rf_model.pkl"))
uns_encoder = joblib.load(os.path.join(base_path, "encoders", "uns_encoder.pkl"))
temprature_scaler = joblib.load(os.path.join(base_path, "scalers", "temprature_scaler.pkl"))
environment_target_encoder = joblib.load(os.path.join(base_path, "encoders", "environment_target_encoder.pkl"))

# --- Constants ---
actual_uns = list(uns_encoder.classes_)
categorical_columns = ['Environment']
not_compose_columns = ['Environment', 'UNS', 'Temperature (deg C)', 'Concentration_clean']

# --- Page Title ---
st.markdown("<h1 style='text-align: center;'>🔍 Corrosion Rate Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Fill in the details below to predict the corrosion class of a material.</p>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("corrosion_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        env = st.selectbox("🌍 Environment", environment)
        temp = st.number_input("🌡️ Temperature (°C)", step=1, value=25)

    with col2:
        conc = st.number_input("🧪 Concentration (%)", min_value=0, max_value=100, value=50)

    with col3:
        uns_input = st.selectbox("📘 UNS Number", actual_uns)
    
    comment = st.text_area("💬 Describe the Condition", height=120, placeholder="e.g. acidic environment with high humidity")

    submitted = st.form_submit_button("🚀 Predict")

# --- Helper Functions ---
def get_tfidf_features(comment):
    tfidf_arr = vectorizer.transform([comment]).toarray()
    return pd.DataFrame(tfidf_arr, columns=vectorizer.get_feature_names_out())

# --- Prediction Pipeline ---
if submitted:
    st.markdown("### 🔄 Processing Input...")
    
    # Clean and encode text
    comment_clean = clean_condition_text(comment)
    tfidf_df = get_tfidf_features(comment_clean)

    main_data = pd.DataFrame([{
        "Environment": env,
        "UNS": uns_input,
        "Temperature (deg C)": temp,
        "Concentration_clean": conc,
    }])

    full_input = pd.concat([main_data, tfidf_df], axis=1)
    pca_input = pca.transform(full_input.drop(columns=not_compose_columns))
    pca_df = pd.DataFrame(pca_input, columns=[f"PCA_{i+1}" for i in range(pca_input.shape[1])], index=full_input.index)

    final_input = pd.concat([main_data[not_compose_columns], pca_df], axis=1)
    final_input["UNS"] = uns_encoder.transform(final_input["UNS"])
    final_input[categorical_columns] = environment_target_encoder.transform(final_input[categorical_columns])
    final_input['Temperature (deg C)'] = temprature_scaler.transform([final_input['Temperature (deg C)']])
    final_input = final_input.loc[:, ~final_input.columns.duplicated()]

    # --- Predict ---
    pred = model.predict(final_input)
    result = targets[str(int(pred[0]))]

    # --- Result ---
    st.markdown("## 🧾 Prediction Result")
    st.success(f"✅ **Predicted Corrosion Class**: `{result}`")
    st.markdown("---")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🛠️ Built with Streamlit | 🧠 Machine Learning powered | 🧑‍🔬 Data Science Demo")
