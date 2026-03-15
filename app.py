import streamlit as st
import os

# Load environment variables (like API keys) from the .env file.
# This MUST happen before we import our tools so they have the keys when they initialize!
from dotenv import load_dotenv
load_dotenv()  

# Import our custom RAG functions (ingesting data and getting answers)
from src.engine.ingestion import ingest_data, DATA_DIR
from src.engine.generation import generate_answer

# ==========================================
# STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Production-Grade RAG", layout="wide")

st.title("📚 Production-Grade RAG System")
st.markdown("A full-lifecycle RAG app built with Hybrid Search, Cross-Encoder Re-ranking, and Langfuse Observability.")

# ==========================================
# SIDEBAR: SETTINGS & DATA UPLOAD
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings & Ingestion")
    
    st.write("Upload PDFs or Text files to build the knowledge base.")
    # Allow the user to upload multiple files
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "txt"])
    
    # "Process & Ingest" Button Logic
    if st.button("Process & Ingest"):
        if not uploaded_files:
            st.error("Please upload files first.")
        else:
            with st.spinner("Saving files and running ingestion pipeline..."):
                # Ensure the data folder exists
                os.makedirs(DATA_DIR, exist_ok=True)
                
                # Save each uploaded file to the local ./data folder
                for f in uploaded_files:
                    path = os.path.join(DATA_DIR, f.name)
                    with open(path, "wb") as output:
                        output.write(f.read())  # Read from browser, write to disk
                
                try:
                    # Run our deep backend function to chunk + embed everything
                    ingest_data()
                    st.success("Ingestion complete! Chunks are stored in ChromaDB and serialized for BM25.")
                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")

    st.markdown("---")
    
    # ==========================================
    # SIDEBAR: OBSERVABILITY STATUS
    # ==========================================
    st.header("📊 Observability")
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        st.success("Langfuse tracing is enabled!")
    else:
        st.warning("Langfuse tracing is disabled. Set LANGFUSE_PUBLIC_KEY in .env.")
    st.markdown("[Open Langfuse Dashboard](http://localhost:3000)")

# ==========================================
# MAIN INTERFACE: CHAT / Q&A
# ==========================================
st.header("Ask the Documents")
# A text box for the user's question
query = st.text_input("Enter your question:")

if st.button("Generate Answer"):
    if not query:
        st.warning("Please enter a query.")
    else:
        # Show a loading spinner during the heavy lifting
        with st.spinner("Retrieving (Hybrid vector+keyword) -> Reranking (Cross-Encoder) -> Generating..."):
            try:
                # Call our core backend RAG function
                response = generate_answer(query)
                st.subheader("Answer with Citations")
                # Display the AI's answer in a blue formatted box
                st.info(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
