import streamlit as st
import pandas as pd
import google.generativeai as genai

file_path = r"C:\Users\aditya.pandey4\Downloads\ManpowerPython_BI.xlsx"

df = pd.read_excel(file_path)

documents = []

for _, row in df.iterrows():
    documents.append(str(row.to_dict()))


genai.configure(api_key = "AIzaSyAuo-SZs8jYVsFUuTkz0cpd6qNSIOU3wYY")
model = genai.GenerativeModel("gemini-3-flash-preview")

st.title("AI Assistant")

query = st.text_input("Ask AI")

if query:
    context  = "\n".join(documents)

    prompt = f"""
    You are an AI assistant, please do not halluciante the question
    Answer only from the below excel data for context:
    {context}

    question: {query}
"""
    response = model.generate_content(prompt)


    st.write("Answer")
    st.write(response.text)





