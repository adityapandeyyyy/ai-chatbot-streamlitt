import streamlit as st
import pandas as pd
import google.generativeai as genai

# Load Excel file
file_path = "ManpowerPython_BI.xlsx"
df = pd.read_excel(file_path)

documents = []

for _, row in df.iterrows():
    documents.append(str(row.to_dict()))

# Load API key from Streamlit Secrets
genai.configure(api_key = st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.0-flash")

st.title("AI Assistant")

query = st.text_input("Ask AI")

if query:
    context = "\n".join(documents)

    prompt = f"""
    You are an AI assistant, please do not hallucinate.
    Answer ONLY from the Excel data provided below.

    Context:
    {context}

    Question: {query}
    """

    response = model.generate_content(prompt)

    st.write("Answer")
    st.write(response.text)
