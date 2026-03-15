import os
import pickle
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Import functions and variables we defined in our ingestion file
from .ingestion import get_embeddings, CHROMA_DB_DIR

def get_vectorstore(persist_directory=CHROMA_DB_DIR):
    """
    Load the Chroma DB vector store that we created during the "ingestion" phase.
    Vector storage allows "Semantic Search" (finding meaning, rather than exact words).
    E.g., A search for "canine" will successfully find documents mentioning "dog".
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Chroma DB directory {persist_directory} not found. Please run ingestion first.")
        
    # We must use the exact same embedding model used to save the docs to load them!
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def get_bm25_retriever(persist_directory=CHROMA_DB_DIR):
    """
    Initialize a Keyword Search Retriever (BM25).
    Unlike semantic search (vectors), BM25 looks for exact word matches.
    It is great for finding specific acronyms, part numbers, or exact names.
    We load the raw text chunks we saved in the ingestion step.
    """
    chunks_path = os.path.join(persist_directory, "chunks.pkl")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"BM25 chunks file {chunks_path} not found.")
        
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
        
    # Create the keyword search engine and tell it to return the top 5 matches
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    return bm25_retriever

def get_ensemble_retriever():
    """
    Create a HYBRID Search system.
    This combines Vector Search (for meaning/context) + Keyword Search (for exact terms).
    Combining them almost always yields better results than using either one alone.
    """
    # 1. Setup Vector Retrieval (Semantic) to get top 5 docs
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 2. Setup Keyword Retrieval (BM25) to get top 5 docs
    bm25_retriever = get_bm25_retriever()
    
    # 3. Combine them using an Ensemble (give them a 50/50 vote weight)
    # This will return a larger pool of documents (up to 10)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], 
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def get_rerank_retriever():
    """
    Add a "Reranker" on top of our Hybrid Search.
    
    How it works:
    1. Our Ensemble (Hybrid) finds ~10 potentially relevant documents extremely fast.
    2. A Cross-Encoder model (a much smarter but slower AI) takes a deep look at the question 
       and those 10 documents side-by-side.
    3. It grades them all out of 100, sorts them, and keeps ONLY the absolute best Top 3.
    This ensures the final AI prompt only gets the most highly relevant context.
    """
    ensemble_retriever = get_ensemble_retriever()
    
    # Use a small cross encoder model specifically trained for question-answering
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Tell it to only return the absolute top 3 highest graded chunks
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    
    # Wrap our hybrid search with the reranker
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    return rerank_retriever

def advanced_retrieve(query):
    """Perform hybrid search + reranking."""
    retriever = get_rerank_retriever()
    return retriever.invoke(query)

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    print(f"Retrieving top chunks (Hybrid + Rerank) for query: '{query}'")
    try:
        chunks = advanced_retrieve(query)
        for i, c in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {c.metadata.get('source', 'Unknown')}")
            print(f"Content: {c.page_content[:200]}...")
    except Exception as e:
        print(f"Retrieval error: {e}")
