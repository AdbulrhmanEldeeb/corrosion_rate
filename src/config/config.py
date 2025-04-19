import os

# ------------------------ Constants & Config ------------------------
BASE_PATH = "src"
MODEL_PATHS = {
    "pca": os.path.join(BASE_PATH, "models", "decomposers", "pca.pkl"),
    "model": os.path.join(BASE_PATH, "models", "classifiers", "rf_all_data.pkl"),
    "uns_encoder": os.path.join(BASE_PATH, "models", "encoders", "uns_encoder.pkl"),
    "env_encoder": os.path.join(
        BASE_PATH, "models", "encoders", "env_target_encoder.pkl"
    ),
    "temp_scaler": os.path.join(
        BASE_PATH, "models", "scalers", "temprature_scaler.pkl"
    ),
}
NOT_COMPOSE_COLUMNS = [
    "Environment",
    "UNS",
    "Temperature (deg C)",
    "Concentration_clean",
]
CATEGORICAL_COLUMNS = ["Environment"]
PAGE_ICON = "src/assets/images/corrosive.png"
SIDEBAR_IMAGE = (
    "https://www.ddcoatings.co.uk/wp-content/uploads/2019/09/pipeline-corrosion.jpg"
)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "deepseek-r1-distill-llama-70b",
    "meta-llama/llama-4-maverick-17b-128e-instruct"
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "mistral-saba-24b",
    "qwen-qwq-32b",
]
