import streamlit as st
from chat.chat import invoke_llm
from utils.vars import environment
from config.config import PIPE_ICON
from utils.processors import remove_think_tags

st.set_page_config(
    page_title="Material Selector (LLM)", layout="wide", page_icon=PIPE_ICON
)

# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.markdown("## 🧱 Material Selector for Corrosion Resistance")
    st.image("src/assets/images/material_selection.jpeg")
    st.markdown(
        "Get material suggestions using an LLM based on your corrosion environment."
    )
    st.markdown("🧠 Powered by LLMs | 🔍 Intelligent Selection")

# ------------------------ Page Header ------------------------
st.markdown(
    "<h1 style='text-align: center;'>🧠 AI-Powered Corrosion Material Selector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Describe the corrosion environment and let the AI suggest suitable materials.</p>",
    unsafe_allow_html=True,
)

# ------------------------ Input Form ------------------------
with st.form("llm_material_selector"):
    col1, col2 = st.columns(2)

    with col1:
        env = st.selectbox(
            "🌍 Environment Type",
            options=environment,
            help="e.g., seawater, acidic, alkaline",
        )
        pH = st.number_input("🧪 pH Level", min_value=0.0, max_value=14.0, value=7.0)
        chloride = st.selectbox(
            "🧂 Chloride Presence", ["None", "Low", "Moderate", "High"]
        )
        temperature = st.number_input("🌡️ Operating Temperature (°C)", value=25)
        pressure = st.number_input("⚙️ Operating Pressure (bar)", value=1.0)

    with col2:
        flow = st.selectbox(
            "🌊 Flow Condition",
            ["Static", "Low velocity", "High velocity", "Turbulent"],
        )
        contact = st.selectbox("🔩 Galvanic Contact with Other Metals?", ["Yes", "No"])
        design_life = st.number_input("📅 Required Design Life (Years)", value=10)
        maintenance = st.selectbox(
            "🛠️ Maintenance Frequency", ["Low", "Moderate", "High"]
        )
        budget = st.selectbox("💰 Budget Constraint", ["None", "Low", "Medium", "High"])

    custom_notes = st.text_area(
        "📝 Additional Notes (Optional)",
        height=120,
        placeholder="Any extra details about the environment or design requirements...",
    )

    submitted = st.form_submit_button("🔎 Suggest Materials")

# ------------------------ LLM Output ------------------------
if submitted:
    user_prompt = f"""
You are a corrosion engineering assistant helping select optimal materials for corrosion resistance in industrial settings.

Based on the following operating and environmental conditions, recommend the **top 2–3 materials**:

- 🌍 Environment: {env}
- 🧪 pH Level: {pH}
- 🧂 Chloride Presence: {chloride}
- 🌡️ Temperature: {temperature}°C
- ⚙️ Pressure: {pressure} bar
- 💨 Flow Condition: {flow}
- 🔗 Galvanic Contact: {contact}
- 📆 Required Design Life: {design_life} years
- 🛠️ Maintenance Requirements: {maintenance}
- 💰 Budget Constraints: {budget}
- 📝 Additional Notes: {custom_notes}

Please provide your output in the following format:

1. **Material Name (UNS Code)**  
   - ✅ *Why it is suitable* (highlight corrosion resistance, mechanical properties, compatibility, etc.)  
   - ⚠️ *Limitations* or special handling considerations  
   - Suggestions: Suggested surface treatments or enhancements (if needed)  

Conclude with:
- 🎯 A final recommendation if one material clearly stands out for the given case.
- 🧠 Reminders or caveats (e.g., importance of site-specific testing, monitoring methods, etc.)

Use a **professional and concise tone**. Structure your response clearly with bullet points or short paragraphs to enhance readability for engineers in the field.
"""

    response = remove_think_tags(invoke_llm(user_prompt))
    st.markdown("## 🧪 Suggested Materials")
    st.success(response)

    # Combine inputs with LLM response for download
    txt_content = "Material Selection Report\n\n"
    txt_content += "Input Parameters:\n"
    txt_content += f"- Environment: {env}\n"
    txt_content += f"- pH Level: {pH}\n"
    txt_content += f"- Chloride Presence: {chloride}\n"
    txt_content += f"- Temperature: {temperature}°C\n"
    txt_content += f"- Pressure: {pressure} bar\n"
    txt_content += f"- Flow Condition: {flow}\n"
    txt_content += f"- Galvanic Contact: {contact}\n"
    txt_content += f"- Required Design Life: {design_life} years\n"
    txt_content += f"- Maintenance: {maintenance}\n"
    txt_content += f"- Budget: {budget}\n"
    txt_content += f"- Additional Notes: {custom_notes}\n\n"
    txt_content += "AI Recommendations:\n"
    txt_content += response

    txt_bytes = txt_content.encode("utf-8")
    st.download_button(
        label="📄 Download Recommendations as TXT",
        data=txt_bytes,
        file_name="material_recommendations.txt",
        mime="text/plain",
    )


# ------------------------ Footer ------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🔧 Built with Streamlit | 🤖 AI Material Selector | 🌐 LLM-Powered")
