import pandas as pd
import numpy as np
import joblib
from utils.processors import clean_condition_text, get_scibert_embedding
from config.config import BASE_PATH, MODEL_PATHS, NOT_COMPOSE_COLUMNS, CATEGORICAL_COLUMNS
from utils.vars import targets
import streamlit as st 


class CorrosionClassifier:
    def __init__(self):
        self.models = self._load_models()

    @staticmethod
    @st.cache_resource
    def _load_models():
        return {
            "pca": joblib.load(MODEL_PATHS["pca"]),
            "model": joblib.load(MODEL_PATHS["model"]),
            "uns_encoder": joblib.load(MODEL_PATHS["uns_encoder"]),
            "env_encoder": joblib.load(MODEL_PATHS["env_encoder"]),
            "temp_scaler": joblib.load(MODEL_PATHS["temp_scaler"]),
        }

    def preprocess_input(self, env: str, temp: float, conc: float, uns_input: str, comment: str):
        """Preprocess inputs into model-ready format."""
        # Build input DataFrame
        input_df = pd.DataFrame([{
            "Environment": env,
            "Temperature (deg C)": temp,
            "Concentration_clean": conc,
            "UNS": uns_input,
        }])

        # Encode categorical variables
        input_df["Environment"] = self.models["env_encoder"].transform(input_df["Environment"])
        input_df["UNS"] = self.models["uns_encoder"].transform(input_df["UNS"])

        # Scale temperature
        input_df["Temperature (deg C)"] = self.models["temp_scaler"].transform(input_df[["Temperature (deg C)"]])

        # Process condition text using SciBERT
        cleaned_comment = clean_condition_text(comment)
        scibert_embedding = np.squeeze(get_scibert_embedding(cleaned_comment))
        scibert_df = pd.DataFrame([scibert_embedding], columns=[f"scibert_{i}" for i in range(len(scibert_embedding))])

        # PCA transformation
        pca_emb = self.models["pca"].transform(scibert_df)
        pca_df = pd.DataFrame(pca_emb, columns=[f"PCA_{i+1}" for i in range(15)])

        # Final input
        full_input = pd.concat([input_df.reset_index(drop=True), pca_df], axis=1)
        ordered_columns = NOT_COMPOSE_COLUMNS + [f"PCA_{i+1}" for i in range(15)]
        full_input = full_input[ordered_columns]

        return full_input

    def predict(self, env: str, temp: float, conc: float, uns_input: str, comment: str):
        """Predict corrosion class and return it with the raw input."""
        full_input = self.preprocess_input(env, temp, conc, uns_input, comment)
        prediction = self.models["model"].predict(full_input)
        predicted_class = targets.get(str(int(prediction[0])), "Unknown")
        return predicted_class, full_input
