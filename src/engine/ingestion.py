import os
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================
# CONFIGURATION
# ==========================================
# Directory where our vector database will be saved on your computer
CHROMA_DB_DIR = "./chroma_db"
# Directory where you should place your raw files (PDFs, TXT files)
DATA_DIR = "./data"

def get_embeddings():
    """
    Initialize the Embedding Model.
    Think of an embedding model as a translator that turns human text into numbers (vectors) 
    that a computer can understand and compare mathematically.
    We are using a small, fast, free model from HuggingFace called 'all-MiniLM-L6-v2'.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_documents(data_dir=DATA_DIR):
    """
    Load all documents (PDFs and Text files) from the specified folder.
    This reads the files and converts them into text that Python can process.
    """
    documents = []
    
    # If the folder doesn't exist yet, create an empty one and return
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return documents
        
    # 1. Load all PDF files in the folder and its subdirectories
    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents.extend(pdf_loader.load())
    
    # 2. Load all plain Text files (.txt) in the folder
    text_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents.extend(text_loader.load())
    
    # Return the combined list of all loaded pages/documents
    return documents

def chunk_documents(documents, chunk_size=800, chunk_overlap=100):
    """
    Chunk documents into smaller, manageable pieces (like paragraphs).
    Why do we do this? 
    1. AI models can't read entire books at once (they have a "context window" limit).
    2. We want to find the specific paragraph that answers a question, not the whole book.
    
    chunk_overlap=100 means the end of chunk 1 overlaps with the start of chunk 2. 
    This prevents sentences from being cut in half and losing their meaning.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # Max number of characters per chunk
        chunk_overlap=chunk_overlap,  # Number of characters to overlap between chunks
        separators=["\n\n", "\n", " ", ""] # Try to split by paragraphs first, then sentences, etc.
    )
    return text_splitter.split_documents(documents)

def ingest_data(data_dir=DATA_DIR, persist_directory=CHROMA_DB_DIR):
    """
    The Main Ingestion Pipeline. 
    This function brings it all together: Load -> Chunk -> Embed -> Save Database.
    """
    print(f"Loading documents from {data_dir}...")
    documents = load_documents(data_dir)
    
    # If folder is empty, stop here
    if not documents:
        print(f"No documents found to ingest in {data_dir}.")
        return None
        
    print(f"Loaded {len(documents)} document pages/sections. Chunking...")
    # Break long PDFs into small paragraphs
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    print("Initializing embeddings model...")
    # Get our model for turning text into numbers
    embeddings = get_embeddings()
    
    print(f"Storing chunks in Chroma DB at {persist_directory}...")
    # Create the Vector Database (Chroma). 
    # This automatically converts every text chunk into a number vector and saves it to a folder.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # We ALSO save the raw text chunks to a standard file (chunks.pkl).
    # Why? Because later we will do keyword searches (BM25) which requires raw text, not just number vectors!
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
