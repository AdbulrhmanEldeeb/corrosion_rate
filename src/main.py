import streamlit as st
import pandas as pd
import joblib
import os

from utils.processors import clean_condition_text
from utils.vars import environment, targets

st.set_page_config(page_title="Corrosion Classifier", layout="wide")
# ------------------------
# 📌 Constants & Config
# ------------------------
BASE_PATH = "src"
MODEL_PATHS = {
    "pca": os.path.join(BASE_PATH, "decomposers", "pca_model.pkl"),
    "vectorizer": os.path.join(BASE_PATH, "vectorizers", "tfidf_vectorizer.pkl"),
    "model": os.path.join(BASE_PATH, "classifiers", "tunned_rf_model.pkl"),
    "uns_encoder": os.path.join(BASE_PATH, "encoders", "uns_encoder.pkl"),
    "env_encoder": os.path.join(BASE_PATH, "encoders", "environment_target_encoder.pkl"),
    "temp_scaler": os.path.join(BASE_PATH, "scalers", "temprature_scaler.pkl"),
}

NOT_COMPOSE_COLUMNS = ['Environment', 'UNS', 'Temperature (deg C)', 'Concentration_clean']
CATEGORICAL_COLUMNS = ['Environment']

# ------------------------
# 🧠 Load Models
# ------------------------
@st.cache_resource
def load_models():
    return {
        "pca": joblib.load(MODEL_PATHS["pca"]),
        "vectorizer": joblib.load(MODEL_PATHS["vectorizer"]),
        "model": joblib.load(MODEL_PATHS["model"]),
        "uns_encoder": joblib.load(MODEL_PATHS["uns_encoder"]),
        "env_encoder": joblib.load(MODEL_PATHS["env_encoder"]),
        "temp_scaler": joblib.load(MODEL_PATHS["temp_scaler"]),
    }

models = load_models()

# ------------------------
# 🔧 Helper Functions
# ------------------------
def get_tfidf_features(comment: str) -> pd.DataFrame:
    """Transform comment into TF-IDF features."""
    tfidf_arr = models["vectorizer"].transform([comment]).toarray()
    return pd.DataFrame(tfidf_arr, columns=models["vectorizer"].get_feature_names_out())

def build_final_input(env: str, temp: float, conc: float, uns_input: str, comment: str) -> pd.DataFrame:
    """Preprocess and merge all input features."""
    comment_clean = clean_condition_text(comment)
    tfidf_df = get_tfidf_features(comment_clean)

    main_data = pd.DataFrame([{
        "Environment": env,
        "UNS": uns_input,
        "Temperature (deg C)": temp,
        "Concentration_clean": conc,
    }])

    # PCA Transformation
    non_pca_data = main_data.drop(columns=NOT_COMPOSE_COLUMNS)
    pca_input = models["pca"].transform(pd.concat([non_pca_data, tfidf_df], axis=1))
    pca_df = pd.DataFrame(pca_input, columns=[f"PCA_{i+1}" for i in range(pca_input.shape[1])])

    final_df = pd.concat([main_data[NOT_COMPOSE_COLUMNS], pca_df], axis=1)

    # Encode and Scale
    final_df["UNS"] = models["uns_encoder"].transform(final_df["UNS"])
    final_df[CATEGORICAL_COLUMNS] = models["env_encoder"].transform(final_df[CATEGORICAL_COLUMNS])
    final_df['Temperature (deg C)'] = models["temp_scaler"].transform([final_df['Temperature (deg C)']])
    return final_df.loc[:, ~final_df.columns.duplicated()]

def predict_corrosion_class(input_df: pd.DataFrame) -> str:
    """Run prediction pipeline."""
    pred = models["model"].predict(input_df)
    return targets.get(str(int(pred[0])), "Unknown")

# ------------------------
# 🌐 Page Setup
# ------------------------

with st.sidebar:
    st.image("https://www.ddcoatings.co.uk/wp-content/uploads/2019/09/pipeline-corrosion.jpg", use_container_width=True)
    st.markdown("## 🧪 Corrosion Classifier")
    st.markdown("Predict the **corrosion rate** based on material and environment conditions.")
    st.markdown("---")
    st.markdown("🔬 Powered by ML | 📊 PCA | 🌐 TF-IDF")

# ------------------------
# 🧾 App Title
# ------------------------
st.markdown("<h1 style='text-align: center;'>🔍 Corrosion Rate Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Fill in the details below to predict the corrosion rate of a material.</p>", unsafe_allow_html=True)

# ------------------------
# 📋 Form Input
# ------------------------
with st.form("corrosion_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        env = st.selectbox("🌍 Environment (the surrounding medium in which the material is exposed)", environment)
        temp = st.number_input("🌡️ Temperature (°C)", step=1, value=25)

    with col2:
        conc = st.number_input("🧪 Concentration (%)", min_value=0, max_value=100, value=50)

    with col3:
        uns_input = st.selectbox("📘 Alloy UNS Number (Unified Numbering System)", list(models["uns_encoder"].classes_))

    comment = st.text_area("💬 Describe the Condition", height=120, placeholder="e.g. acidic environment with high humidity")

    submitted = st.form_submit_button("🚀 Predict")

# ------------------------
# 🧠 Prediction
# ------------------------
if submitted:
    input_df = build_final_input(env, temp, conc, uns_input, comment)
    result = predict_corrosion_class(input_df)

    st.markdown("## 🧾 Prediction Result")
    st.success(f"✅ **Predicted Corrosion Class**: `{result}`")
    st.markdown("---")

# ------------------------
# 📎 Footer
# ------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🛠️ Built with Streamlit | 🧠 Machine Learning powered | 🧑‍🔬 Data Science Demo")
