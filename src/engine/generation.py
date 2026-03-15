import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from .retrieval import get_rerank_retriever

# ==========================================
# OBSERVABILITY / TRACING SETUP (LANGFUSE)
# ==========================================
# This automatically connects to the Langfuse server if the keys are available.
# Langfuse tracks every step of the AI (time taken, cost, exact inputs/outputs).
if os.environ.get("LANGFUSE_PUBLIC_KEY"):
    Langfuse()

def get_langfuse_handler():
    """Returns a tracking handler we can attach to our AI requests."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        print("Warning: LANGFUSE_PUBLIC_KEY not set. Local tracking disabled.")
        return None
    
    return CallbackHandler()

def format_docs(docs):
    """
    Format retrieved documents and include their source metadata for citation.
    This takes the raw Documents retrieved from our DB and turns them into a clean 
    string block so the AI can read them easily.
    """
    formatted = []
    for i, doc in enumerate(docs):
        # We try to get the original filename, otherwise label it 'Chunk 1', etc.
        source = doc.metadata.get("source", f"Chunk {i+1}")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def get_rag_chain():
    """
    Initialize the complete RAG (Retrieval-Augmented Generation) chain.
    This connects the "Search Engine" to the "Brain".
    """
    
    # 1. Provide the LLM (The Brain)
    # Using Llama-3.2 via Ollama (a local, free model running on your machine)
    # temperature=0 makes it factual and less "creative" (prevents hallucination)
    llm = ChatOllama(model="llama3.2:1b", temperature=0)

    # 2. Provide the Instructions (The Prompt)
    # This is exactly what the AI sees when asked a question.
    system_prompt = """You are a precise, research assistant designed to answer user questions exclusively based on the provided context.

Context documents:
{context}

Instructions:
1. Answer the user's question using ONLY the context provided above.
2. For every claim or extracted piece of information, you MUST include an explicit citation pointing to the exact source paragraph/chunk provided in the context (e.g., "According to [Source: ...], ").
3. If the provided context does not contain sufficient information to answer the query, you MUST explicitly state: "I cannot answer this based on the provided documents." Do NOT hallucinate or use outside knowledge.

Question: """
    
    # Create a template that will automatically fill in the {context} and {question}
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # 3. Provide the Search Engine
    # We use our advanced hybrid retriever with cross-encoder reranking
    retriever = get_rerank_retriever()

    # 4. Chain everything together!
    # - Run the retriever, then format the docs into a string
    # - Pass the user's question straight through
    # - Feed both into the Prompt
    # - Feed the Prompt to the LLM
    # - Extract purely the string output from the LLM
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def generate_answer(query):
    """
    Generate answer for a query, tracing with Langfuse if it's configured.
    """
    chain = get_rag_chain()
    langfuse_handler = get_langfuse_handler()
    
    # If Langfuse is enabled, we attach our tracking "callback" here
    config = {}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
        
    print(f"Generating answer for: '{query}'")
    # Trigger the entire RAG chain!
    response = chain.invoke(query, config=config)
    return response

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the key points in the documents?"
    ans = generate_answer(query)
    print("\n--- Answer ---")
    print(ans)
