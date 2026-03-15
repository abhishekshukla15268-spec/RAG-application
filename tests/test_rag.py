import os
import sys
import json
import warnings
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Add project root to path so we can import our engine functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engine.generation import generate_answer
from src.engine.retrieval import get_rerank_retriever

# Suppress warnings that clutter the evaluation output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run_rag_pipeline_eval():
    """
    RAGAS Evaluation Loop.
    This acts like an automated grading system for our AI. 
    It asks the AI 20 pre-written questions and grades how well the AI answers them 
    based on two metrics:
    1. Faithfulness: Is the answer hallucinated, or strictly based on the text?
    2. Answer Relevancy: Did it actually answer the user's question, or ramble?
    """
    # 1. Load the "Golden Dataset" (A list of Questions + Ground Truth Answers)
    filepath = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(filepath, "r") as f:
        qa_dataset = json.load(f)

    # We need to collect 4 lists to pass to Ragas
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    # Get our Search Engine
    retriever = get_rerank_retriever()
    
    for item in qa_dataset:
        q = item["question"]
        questions.append(q)
        ground_truths.append(item["ground_truth"])
        
        # Step A: Ask our RAG system the question and get the answer
        ans = generate_answer(q)
        answers.append(ans)
        
        # Step B: Record exactly which paragraphs the system retrieved to answer it
        # (Ragas uses this to check if the AI "hallucinated" info from outside these docs)
        retrieved_docs = retriever.invoke(q)
        contexts.append([doc.page_content for doc in retrieved_docs])

    # Convert everything to a format the Ragas library understands
    data_samples = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_samples)
    
    # 2. Initialize the "Judge". 
    # Ragas uses an LLM to grade our LLM! We provide our local Llama 3.2 model to be the grader.
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2:1b"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    # Tell the grading metrics which "Judge" model to use
    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings
    
    metrics = [faithfulness, answer_relevancy]

    # 3. Run the Evaluation
    # We use a RunConfig to limit parallel requests so we don't crash our local Llama model
    run_config = RunConfig(max_workers=2, timeout=120)
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config
    )
    
    # 4. Process and Print Results
    # Convert scores to a table and calculate the average score for the whole test
    df = result.to_pandas()
    faithfulness_score = df["faithfulness"].mean()
    answer_relevancy_score = df["answer_relevancy"].mean()
    
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS:")
    print("="*50)
    print(f"Faithfulness:     {faithfulness_score:.4f}")
    print(f"Answer Relevancy: {answer_relevancy_score:.4f}")
    print("="*50 + "\n")
    
    # 5. Automated CI/CD Check
    # If the scores fall below these thresholds, the script crashes (assert fails). 
    # This prevents deploying bad code to production.
    assert faithfulness_score >= 0.8, f"Faithfulness metric failed! Score: {faithfulness_score}"
    assert answer_relevancy_score >= 0.7, f"Answer Relevancy failed! Score: {answer_relevancy_score}"
    print("✅ All metrics passed CI/CD thresholds!")

if __name__ == "__main__":
    run_rag_pipeline_eval()
