import streamlit as st
import pandas as pd
from google import genai
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
MODEL_ID = "gemini-flash-latest"

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("ManpowerPython_BI.xlsx")
    df = df.fillna("")
    return df

df = load_data()

# Convert rows with schema
documents = df.astype(str).apply(
    lambda row: " | ".join([f"{col}: {row[col]}" for col in df.columns]),
    axis=1
).tolist()

# -----------------------------
# VECTOR STORE
# -----------------------------
@st.cache_resource
def create_vector_store(docs):

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embed_model.encode(docs)

    dimension = embeddings.shape[1]

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embed_model

index, embed_model = create_vector_store(documents)

# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve_context(query):

    query_embedding = embed_model.encode([query])

    faiss.normalize_L2(query_embedding)

    k = 15

    distances, indices = index.search(query_embedding, k)

    results = [documents[i] for i in indices[0]]

    return results

# -----------------------------
# STREAMLIT
# -----------------------------
st.title("Excel AI Assistant")

query = st.text_input("Ask question")

if st.button("Ask"):

    if query:

        rows = retrieve_context(query)

        st.write("### Retrieved Rows (debug)")
        st.write(rows)

        context = "\n".join(rows)

        prompt = f"""
        You are analyzing Excel data.

        Dataset columns:
        {", ".join(df.columns)}

        Context rows:
        {context}

        Question:
        {query}

        Answer using the context rows only.
        """

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )

        st.write("### Answer")
        st.write(response.text)
