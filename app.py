import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()  # Must be before project imports so env vars are available at module init

from src.engine.ingestion import ingest_data, DATA_DIR
from src.engine.generation import generate_answer

st.set_page_config(page_title="Production-Grade RAG", layout="wide")

st.title("📚 Production-Grade RAG System")
st.markdown("A full-lifecycle RAG app built with Hybrid Search, Cross-Encoder Re-ranking, and Langfuse Observability.")

# Sidebar for Ingestion and settings
with st.sidebar:
    st.header("⚙️ Settings & Ingestion")
    
    st.write("Upload PDFs or Text files to build the knowledge base.")
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "txt"])
    
    if st.button("Process & Ingest"):
        if not uploaded_files:
            st.error("Please upload files first.")
        else:
            with st.spinner("Saving files and running ingestion pipeline..."):
                os.makedirs(DATA_DIR, exist_ok=True)
                for f in uploaded_files:
                    path = os.path.join(DATA_DIR, f.name)
                    with open(path, "wb") as output:
                        output.write(f.read())
                
                try:
                    ingest_data()
                    st.success("Ingestion complete! Chunks are stored in ChromaDB and serialized for BM25.")
                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")

    st.markdown("---")
    st.header("📊 Observability")
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        st.success("Langfuse tracing is enabled!")
    else:
        st.warning("Langfuse tracing is disabled. Set LANGFUSE_PUBLIC_KEY in .env.")
    st.markdown("[Open Langfuse Dashboard](http://localhost:3000)")

# Main Chat Interface
st.header("Ask the Documents")
query = st.text_input("Enter your question:")

if st.button("Generate Answer"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving (Hybrid vector+keyword) -> Reranking (Cross-Encoder) -> Generating..."):
            try:
                # Add environment check
                response = generate_answer(query)
                st.subheader("Answer with Citations")
                st.info(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
