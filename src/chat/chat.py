from dotenv import load_dotenv
from config.config import GROQ_MODELS
import os
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()

groq_api_key1 = os.getenv("GROQ_API_KEY_1")
groq_api_key2 = os.getenv("GROQ_API_KEY_2")

groq_api_keys = [groq_api_key1, groq_api_key2]


api_index = 0
model_index = 0


def get_next_api_and_model():
    """
    Retrieves the next available API key and model name in a round-robin fashion.
    This function cycles through the list of Groq API keys and model names,
    ensuring a balanced usage of available resources.
    """
    global api_index, model_index
    api_key = groq_api_keys[api_index]
    model_name = GROQ_MODELS[model_index]
    api_index = (api_index + 1) % len(groq_api_keys)
    model_index = (model_index + 1) % len(GROQ_MODELS)
    return api_key, model_name


def get_groq_llm():
    """
    Initializes and returns a ChatGroq LLM instance using the next available API key and model.
    The function fetches an API key and model name from the round-robin selection
    and creates a ChatGroq instance with predefined parameters.

    """
    api_key, model = get_next_api_and_model()
    return (
        ChatGroq(model_name=model, api_key=api_key, temperature=0.3, max_tokens=1024),
        model,
    )


def get_main_prompt(df):
    prompt = f"""
You are a corrosion control expert assisting engineers in preventing material degradation in industrial environments.

Given the following dataframe:
{df}

Interpret the corrosion severity using the following scale:
- A (Resistant): < 0.002 inches/year
- B (Good): < 0.020 inches/year
- C (Questionable): 0.020 - 0.050 inches/year
- D (Poor): > 0.050 inches/year

Generate a concise technical recommendation with a clear bullet-point structure, suitable for field engineers. Your output should include:

- The interpreted corrosion severity class and its implications.
- Likely causes based on the material and environment.
- Specific mitigation strategies (e.g., naming coating types, inhibitor types, or environmental controls).
- Suggested monitoring or tests (e.g., EIS, weight loss, visual inspection).
- Optional: Practical next steps such as documentation or sharing findings.

Ensure the tone is practical, professional, and clear. Respond in exactly 5 bullet points.
"""
    return prompt


def invoke_llm(prompt):
    try:
        groq_llm, llm_name = get_groq_llm()
        response = groq_llm.invoke(prompt)
        return response.content

    except Exception as e:
        # You can also log the error using `logging` if preferred
        return f"⚠️ An error occurred while generating LLM output: {e}"
