# 📚 Production-Grade RAG System

Welcome to the Production-Grade Retrieval-Augmented Generation (RAG) system! This project is a complete, scalable backend and frontend designed to allow users to securely chat with their own documents (PDFs and Text files).

It goes far beyond a simple "Hello World" RAG tutorial by implementing **advanced retrieval strategies**, **deep observability tracing**, and **automated LLM evaluation**.

---

## 🌟 Key Features

1. **Advanced Hybrid Retrieval Pipeline**
   - **Vector Search (Semantic):** Uses `ChromaDB` and HuggingFace embeddings (`all-MiniLM-L6-v2`) to find documents based on *meaning and context*.
   - **Keyword Search (BM25):** Uses a traditional exact-match algorithm to pinpoint exact names, IDs, or acronyms.
   - **Cross-Encoder Re-ranking:** A specialized AI model (`ms-marco-MiniLM-L-6-v2`) acts as a judge to deeply re-evaluate and sort the hybrid search results, ensuring only the absolute most relevant top 3 chunks are sent to the LLM.

2. **Strict LLM Generation & Citation**
   - Employs a local LLM via Ollama (`llama3.2:1b`) with `temperature=0` to ensure factual responses.
   - The generation prompt is strictly engineered to prevent hallucination. Every fact generated **must** include an explicit `[Source: X]` citation pointing back to the specific uploaded document.

3. **Enterprise Observability (Langfuse)**
   - Fully integrated with Langfuse. Every user query traces:
     - How long retrieval took.
     - What exact documents were fetched.
     - The exact prompt sent to the LLM.
     - Token usage and latency.

4. **Automated LLM Evaluation (Ragas)**
   - Includes a continuous testing script (`test_rag.py`) that uses the **Ragas** framework.
   - It iterates over a "Golden Dataset" of 20 questions, automatically generates answers, and grades the AI based on **Faithfulness** (no hallucinations) and **Answer Relevancy** (actually answering the prompt).
   - Simulates a CI/CD pipeline by crashing the build if score thresholds drop.

---

## 🛠️ Prerequisites & Setup

This application is designed to run entirely locally, meaning **your data never leaves your machine.**

### 1. Clone the Repository
```bash
git clone https://github.com/abhishekshukla15268-spec/RAG-application.git
cd RAG-application
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed, then run:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 2. Install & Start Ollama (The Brain)
You need to install [Ollama](https://ollama.com/) to run the local LLM.
Once installed, open a terminal and download the model:
```bash
ollama run llama3.2:1b
```
*(Keep the Ollama app running in the background).*

### 3. Setup Langfuse (Observability - Optional but Recommended)
For local tracing, this repo includes a Docker Compose file.
```bash
# Start the Langfuse server locally on port 3000
docker-compose up -d
```
1. Go to `http://localhost:3000`
2. Create an account and a new project.
3. Generate API Keys in the Langfuse settings.
4. Copy `.env.example` to `.env` and paste your keys:
```env
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
```

---

## 🚀 How to Run the App

### 1. The Web UI (Streamlit)
Start the user interface by running:
```bash
streamlit run app.py
```
- A browser window will open.
- **Step 1:** Use the sidebar to upload a PDF or text file. Click **Process & Ingest**. 
- **Step 2:** Ask a question in the chat box! The AI will retrieve the data, re-rank it, and provide an answer with citations.

### 2. The Evaluation Suite (Ragas)
To test if the RAG system is actually providing good answers, run the evaluation script:
```bash
python tests/test_rag.py
```
This will run the framework against the `golden_dataset.json` and print out the average Faithfulness and Answer Relevancy scores.

---

## 📁 Code Architecture Breakdown

If you are new to coding, here is exactly how the files work together:

- **`app.py`**: The frontend. This file builds the buttons, text boxes, and layout using Streamlit.
- **`src/engine/ingestion.py`**: The "FileReader". It takes your uploaded PDFs, cuts them into small paragraphs (chunking), translates those paragraphs into numbers (embeddings), and saves them into the `chroma_db/` folder.
- **`src/engine/retrieval.py`**: The "Search Engine". When a user asks a question, this file does the Hybrid Search (Math + Keywords) and Re-ranking to find the 3 best paragraphs.
- **`src/engine/generation.py`**: The "Talker". It takes the 3 paragraphs found by retrieval, packages them with strict instructions ("don't hallucinate, use citations"), and sends them to the Ollama LLM to get a human-readable answer.
- **`tests/test_rag.py`**: The "Grader". It runs the AI through a gauntlet of questions to mathematically score how smart and accurate it is.

*All source code contains extensive beginner-friendly comments detailing exactly how the functions work line-by-line!*
