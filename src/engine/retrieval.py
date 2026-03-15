import os
import pickle
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from .ingestion import get_embeddings, CHROMA_DB_DIR

def get_vectorstore(persist_directory=CHROMA_DB_DIR):
    """Load the Chroma DB vector store."""
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Chroma DB directory {persist_directory} not found. Please run ingestion first.")
        
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def get_bm25_retriever(persist_directory=CHROMA_DB_DIR):
    """Load the saved chunks and initialize BM25 Retriever."""
    chunks_path = os.path.join(persist_directory, "chunks.pkl")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"BM25 chunks file {chunks_path} not found.")
        
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
        
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    return bm25_retriever

def get_ensemble_retriever():
    """Create a hybrid search retriever combining Vector and Keyword searches."""
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    bm25_retriever = get_bm25_retriever()
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], 
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def get_rerank_retriever():
    """Wrap the ensemble retriever with a cross-encoder for re-ranking."""
    ensemble_retriever = get_ensemble_retriever()
    
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    
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
