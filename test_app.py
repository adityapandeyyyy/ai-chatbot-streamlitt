import streamlit as st
import pandas as pd
from google import genai  # Use the modern 2026 SDK
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
# The new SDK automatically picks up keys, but we pass it explicitly for Streamlit
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
MODEL_ID = "gemini-3-flash"  # Latest stable model for 2026

# -----------------------------
# 2. LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    # Ensure openpyxl is in your requirements.txt
    df = pd.read_excel("ManpowerPython_BI.xlsx")
    return df

df = load_data()

# Convert rows to text for retrieval
documents = df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()

# -----------------------------
# 3. CREATE VECTOR STORE (FAISS)
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
# 4. SEARCH LOGIC
# -----------------------------
def retrieve_context(query):
    query_embedding = embed_model.encode([query])
    k = 5  # Retrieve top 5 most relevant rows
    distances, indices = index.search(query_embedding, k)
    results = [documents[i] for i in indices[0]]
    return "\n".join(results)

# -----------------------------
# 5. STREAMLIT UI & LLM QUERY
# -----------------------------
st.title("Excel AI Assistant 2026")
st.write("Ask questions about your Manpower and Billing data.")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        with st.spinner("Searching and thinking..."):
            context = retrieve_context(query)

            prompt = f"""
            Answer ONLY using the context below. 
            Include specific billing amounts or names if requested.
            If the answer is not in the data, say "Not found in dataset".

            Context:
            {context}

            Question:
            {query}
            """

            # UPDATED: Use the new 'models.generate_content' syntax
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )

            st.write("### Answer")
            st.info(response.text)
    else:
        st.warning("Please enter a question first.")
