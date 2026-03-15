import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from .retrieval import get_rerank_retriever

# Initialize Langfuse client (reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env)
if os.environ.get("LANGFUSE_PUBLIC_KEY"):
    Langfuse()

# Set up Langfuse tracing
def get_langfuse_handler():
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        print("Warning: LANGFUSE_PUBLIC_KEY not set. Local tracking disabled.")
        return None
    
    return CallbackHandler()

def format_docs(docs):
    """Format retrieved documents and include their source metadata for citation."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", f"Chunk {i+1}")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def get_rag_chain():
    """Initialize the RAG generation chain with strict citation prompt."""
    
    # Using Llama-3.2 via Ollama (default local LLM)
    # If you prefer a different provider (e.g., Fireworks, OpenAI), update this.
    llm = ChatOllama(model="llama3.2:1b", temperature=0)

    # Prompt emphasizing citations and constraints
    system_prompt = """You are a precise, research assistant designed to answer user questions exclusively based on the provided context.

Context documents:
{context}

Instructions:
1. Answer the user's question using ONLY the context provided above.
2. For every claim or extracted piece of information, you MUST include an explicit citation pointing to the exact source paragraph/chunk provided in the context (e.g., "According to [Source: ...], ").
3. If the provided context does not contain sufficient information to answer the query, you MUST explicitly state: "I cannot answer this based on the provided documents." Do NOT hallucinate or use outside knowledge.

Question: """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # In Phase 2 we use hybrid advanced retriever with cross-encoder reranking
    retriever = get_rerank_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def generate_answer(query):
    """Generate answer for a query, tracing with Langfuse if configured."""
    chain = get_rag_chain()
    langfuse_handler = get_langfuse_handler()
    
    config = {}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
        
    print(f"Generating answer for: '{query}'")
    response = chain.invoke(query, config=config)
    return response

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the key points in the documents?"
    ans = generate_answer(query)
    print("\n--- Answer ---")
    print(ans)
