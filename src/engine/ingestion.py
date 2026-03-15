import os
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
CHROMA_DB_DIR = "./chroma_db"
DATA_DIR = "./data"

def get_embeddings():
    """Initialize HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_documents(data_dir=DATA_DIR):
    """Load documents from the specified directory."""
    documents = []
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return documents
        
    # Load PDFs
    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents.extend(pdf_loader.load())
    
    # Load Text files
    text_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents.extend(text_loader.load())
    
    return documents

def chunk_documents(documents, chunk_size=800, chunk_overlap=100):
    """Chunk documents into smaller pieces for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def ingest_data(data_dir=DATA_DIR, persist_directory=CHROMA_DB_DIR):
    """Main ingestion pipeline."""
    print(f"Loading documents from {data_dir}...")
    documents = load_documents(data_dir)
    
    if not documents:
        print(f"No documents found to ingest in {data_dir}.")
        return None
        
    print(f"Loaded {len(documents)} document pages/sections. Chunking...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    print("Initializing embeddings model...")
    embeddings = get_embeddings()
    
    print(f"Storing chunks in Chroma DB at {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Save chunks for BM25 retrieval
    chunks_path = os.path.join(persist_directory, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
        
    print("Ingestion complete. Chunks saved for BM25.")
    return vectorstore

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR}. Please add some PDF/TXT files and run again.")
    else:
        ingest_data()
