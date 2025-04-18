import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

def clean_condition_text(text):

    text = text.lower()
    text = re.sub(r"[\n\r]", " ", text)
    text = re.sub(r"[^a-z0-9%.\- ]+", "", text)
    return text


# ------------------------
# 🔧 Helper Functions
# ------------------------
def build_final_input(
    env: str, temp: float, conc: float, uns_input: str, comment: str
) -> pd.DataFrame:
    """Preprocess and merge all input features."""
    comment_clean = clean_condition_text(comment)
    tfidf_df = get_tfidf_features(comment_clean)

    main_data = pd.DataFrame(
        [
            {
                "Environment": env,
                "UNS": uns_input,
                "Temperature (deg C)": temp,
                "Concentration_clean": conc,
            }
        ]
    )

    # PCA Transformation
    non_pca_data = main_data.drop(columns=NOT_COMPOSE_COLUMNS)
    pca_input = models["pca"].transform(pd.concat([non_pca_data, tfidf_df], axis=1))
    pca_df = pd.DataFrame(
        pca_input, columns=[f"PCA_{i+1}" for i in range(pca_input.shape[1])]
    )

    final_df = pd.concat([main_data[NOT_COMPOSE_COLUMNS], pca_df], axis=1)

    # Encode and Scale
    final_df["UNS"] = models["uns_encoder"].transform(final_df["UNS"])
    final_df[CATEGORICAL_COLUMNS] = models["env_encoder"].transform(
        final_df[CATEGORICAL_COLUMNS]
    )
    final_df["Temperature (deg C)"] = models["temp_scaler"].transform(
        [final_df["Temperature (deg C)"]]
    )
    return final_df.loc[:, ~final_df.columns.duplicated()]


model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_scibert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()
