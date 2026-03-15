import streamlit as st
import pandas as pd
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("ManpowerPython_BI.xlsx")
    return df

df = load_data()

# Convert rows to text
documents = df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()

# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------
@st.cache_resource
def create_vector_store(docs):

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embed_model.encode(docs)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embed_model

index, embed_model = create_vector_store(documents)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Excel AI Assistant")

query = st.text_input("Ask a question about the Excel data")

# -----------------------------
# SEARCH RELEVANT ROWS
# -----------------------------
def retrieve_context(query):

    query_embedding = embed_model.encode([query])

    k = 5
    distances, indices = index.search(query_embedding, k)

    results = [documents[i] for i in indices[0]]

    return "\n".join(results)

# -----------------------------
# QUERY GEMINI
# -----------------------------
if st.button("Ask"):

    context = retrieve_context(query)

    prompt = f"""
    Answer ONLY using the context below.
    If the answer is not in the data, say "Not found in dataset".

    Context:
    {context}

    Question:
    {query}
    """

    response = model.generate_content(prompt)

    st.write("### Answer")
    st.write(response.text)
