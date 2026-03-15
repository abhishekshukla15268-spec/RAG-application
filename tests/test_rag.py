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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engine.generation import generate_answer
from src.engine.retrieval import get_rerank_retriever

# Filter warnings globally at the module level for collection
# The original pytestmark was removed, but the intent to filter warnings might still be desired.
# For a native Python script, warnings can be filtered using the warnings module.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run_rag_pipeline_eval():
    """
    Simulated CI check measuring Faithfulness and Answer Relevancy using Ragas.
    """
    filepath = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(filepath, "r") as f:
        qa_dataset = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    retriever = get_rerank_retriever()
    
    for item in qa_dataset:
        q = item["question"]
        questions.append(q)
        ground_truths.append(item["ground_truth"])
        
        # 1. Generate answer
        ans = generate_answer(q)
        answers.append(ans)
        
        # 2. Extract contexts used by the RAG system to evaluate faithfulness
        retrieved_docs = retriever.invoke(q)
        contexts.append([doc.page_content for doc in retrieved_docs])

    data_samples = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data_samples)
    
    # Wrap LLM and embeddings for RAGAS (v0.2+ API)
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2:1b"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    # Use the base metrics (which are already instantiated objects in Ragas v0.4+)
    # and set their evaluators
    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings
    
    metrics = [faithfulness, answer_relevancy]

    # Evaluate with Ragas
    # Add RunConfig to prevent overwhelming the local LLM
    run_config = RunConfig(max_workers=2, timeout=120)
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config
    )
    
    # In Ragas v0.4+, result is an EvaluationResult object, not a dict.
    # Convert scores to pandas dataframe to calculate means
    df = result.to_pandas()
    faithfulness_score = df["faithfulness"].mean()
    answer_relevancy_score = df["answer_relevancy"].mean()
    
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS:")
    print("="*50)
    print(f"Faithfulness:     {faithfulness_score:.4f}")
    print(f"Answer Relevancy: {answer_relevancy_score:.4f}")
    print("="*50 + "\n")
    
    # CI/CD Regression Gating - Strict assertions
    assert faithfulness_score >= 0.8, f"Faithfulness metric failed! Score: {faithfulness_score}"
    assert answer_relevancy_score >= 0.7, f"Answer Relevancy failed! Score: {answer_relevancy_score}"
    print("✅ All metrics passed CI/CD thresholds!")

if __name__ == "__main__":
    run_rag_pipeline_eval()
