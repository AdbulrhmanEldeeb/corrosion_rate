# 🔍 Corrosion Rate Prediction with AI

This Streamlit web application predicts the **corrosion rate** of a material based on selected environmental and alloy conditions, and provides **AI-generated technical recommendations** for corrosion control using an LLM model (e.g., SciBERT).

---

## 🚀 Features

- 🌡️ Input temperature, concentration, environment type, and alloy code.
- 🔬 Predict corrosion rate using a pre-trained ML model.
- 🧠 Generate AI recommendations for corrosion mitigation.
- 💾 Download predictions and recommendations as **CSV** or **TXT** reports.
- ⚡ Fast and interactive UI powered by **Streamlit**.

---

## 🌐 Try the App

You can try the app by visiting the following link:  
[**Corrosion Rate Predictor**](https://corrosion-rate-09.streamlit.app/)

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Model:** SciBERT + PCA + Custom Classifier
- **Backend Tools:** Pandas, NumPy, Joblib
- **LLM Recommendations:** Local or API-based LLM (e.g., OpenAI, HuggingFace Transformers)

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/corrosion-ai-app.git
   cd corrosion-ai-app
    ```
2. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    Run the Streamlit app:
    ```
4. run the app 
    ```bash
    streamlit run app.py
    ```
## 📂 Project Structure

├── app.py                    # Main Streamlit application
├── utils/
│   ├── predictor.py          # CorrosionClassifier class
│   ├── processors.py         # Input processing utilities
│   └── vars.py               # Static values (environments, alloys)
├── chat/
│   └── chat.py               # invoke_llm function for AI recommendations
├── config/
│   └── config.py             # App-wide constants (e.g., logo, icons)
├── requirements.txt
└── README.md


---

## 📄 Download Options

- **CSV Report**: Includes all input parameters, predicted corrosion rate, and AI-generated recommendations.
- **TXT Report**: Simple plain-text report for sharing or documentation.

---

## ✍️ Example Use Case

You have a scenario with an acetaldehyde environment at 25°C and 50% concentration, using alloy `P04995`.  
The app predicts a **low corrosion rate** and recommends **protective coatings, inhibitors, and environmental controls** to maintain performance.

---

## 📌 Notes

- LLM recommendations require an API connection or a locally running model.
- The app currently supports a predefined list of alloys and environments.

### Future versions may include:
- 📊 Visualizations of corrosion trends
- 📁 Upload batch data as CSV
- 📘 Export as PDF or DOCX

---

## 🙌 Acknowledgments

- [Streamlit](https://streamlit.io/)
- [SciBERT](https://github.com/allenai/scibert)
- [Hugging Face Transformers](https://huggingface.co/)
- Domain knowledge and datasets for corrosion modeling

---

## 📬 Contact

For feedback or collaboration, reach out at [abdodebo3@gmail.com](mailto:abdodebo3@gmail.com)
