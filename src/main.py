import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import numpy as np
from utils.processors import clean_condition_text, extract_features_from_comment
from utils.vars import environment, material, material_group, material_family,uns_encodings,targets
sidebar_image='https://www.ddcoatings.co.uk/wp-content/uploads/2019/09/pipeline-corrosion.jpg'
actual_uns=list(uns_encodings.keys())
categorical_columns=['Environment','Material Group', 'Material Family','Material']

# Load vectorizer and model
vectorizer = joblib.load("src/vectorizers/tfidf_vectorizer.pkl")
model = joblib.load("src/classifiers/lgb_model.pkl")
uns_encoder = joblib.load("src/encoders/uns_encoder.pkl")
temprature_scaler=joblib.load('src/scalers/temprature_scaler.pkl')
target_encoder=joblib.load('src/encoders/target_encoder.pkl')
# Features extracted from comment or condition
EXTRACTED_FEATURES = [
    "is_glacial",
    "aerated",
    "unaerated",
    "agitation_moderate",
    "agitation_static",
    "mill_annealed",
    "heat_treated",
    "cast_specimen",
    "lab_test",
    "plant_test",
    "evaporator",
    "diaphragm_cell",
    "mercury_cell",
    "ppm_cu2",
    "ppm_cl",
    "gpl_fe",
    "cu_so4_%",
    "na_cl_%",
    "hno3_%",
    "pH",
]

# UI inputs
st.title("Corrosion Rate Classifier")

env = st.selectbox("Environment (The material surrounding the sample being observed.)",environment)
mat_group = st.selectbox(
    "Material Group (The base element or predominant constituent of a sample; the alloy class. For example, “Stainless steels,” “Copper and alloys” or “Miscellaneous”.)",
    material_group,
    index=0,
)
mat_family = st.selectbox(
    "Material Family (A further narrowing of a Material Group. For example, “Austenitic,” “Tin brass” or “Nobel metals”.)",
    material_family,
    index=0
)
material_input = st.selectbox(
    "Material (The actual composition of the sample. This may be stated in words or by giving an alloy, or composition, designation. For example, “316LN,” “Naval brass” or “Silver”.)",
    material,
    index=0
)
uns_input = st.selectbox(
    "UNS (The Unified Numbering System (UNS) provides a correlation of many numbering systems separately administered by societies, associations and producers of metals and alloys. For example, “S30453,” “C46400” or “P07010”. http://www.astm.org/Standards/E527.html)",
    actual_uns,
    index=0
)
temp = st.number_input("Temperature (deg C)", step=1)
conc = st.number_input(
    "Concentration (Concentration of aqueous solution by volume percent (%).",
    step=1,
    min_value=0,
    max_value=100,
    value=50
)
comment = st.text_area(
    "Comment or Condition (Text summarization from the original source giving additional information about the observation. Examples include, “pH 8.9,” “Aged 500 h at 760 C” and “Ocean depth: 2 to 2070 m (6.5 to 6800 ft)”)"
)
comment = clean_condition_text(comment)


# TF-IDF
def get_tfidf_features(comment):
    tfidf_arr = vectorizer.transform([comment]).toarray()
    return pd.DataFrame(tfidf_arr, columns=vectorizer.get_feature_names_out())


if st.button("Predict"):
    # Extract condition features
    comment_features = extract_features_from_comment(comment)
    cond_df = pd.DataFrame([comment_features])

    # TF-IDF features
    tfidf_df = get_tfidf_features(comment)

    # Combine all features
    main_data = pd.DataFrame(
        [
            {
                "Environment": env,
                "Material Group": mat_group,
                "Material Family": mat_family,
                "Material": material_input,
                "UNS": uns_input,
                "Temperature (deg C)": temp,
                "Concentration_clean": conc,
            }
        ]
    )

    final_input = pd.concat([main_data, cond_df, tfidf_df], axis=1)
    final_input["UNS"] = uns_encoder.transform(final_input["UNS"])
    final_input[categorical_columns]=target_encoder.transform(final_input[categorical_columns])
    final_input['Temperature (deg C)']=temprature_scaler.transform([final_input['Temperature (deg C)']])
    print(final_input)
    final_input = final_input.loc[:, ~final_input.columns.duplicated()]

    pred = model.predict(final_input)
    result=targets[str(int(pred[0]))]
    st.success(f"Predicted corrosion class: {result}")
