# 🤖 AI-Powered Retail Assistant — Conversational Business Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20Streamlit-red.svg)](https://flask.palletsprojects.com/)
[![RAG Pipeline](https://img.shields.io/badge/RAG-LangChain%20%7C%20FAISS-green.svg)](https://python.langchain.com/)
[![LLM Provider](https://img.shields.io/badge/LLM-Hugging%20Face%20Serverless-yellow.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/Tests-45%20Passed-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An enterprise-grade, hybrid **Conversational Business Intelligence (BI)** assistant built on the **UCI Online Retail dataset** (~540,000 transactions). 

This project solves a fundamental challenge in applying generative AI to business intelligence: **Large Language Models are notorious for hallucinating arithmetic and failing at precise calculations, while traditional BI dashboards lack conversational flexibility.** 

The **AI-Powered Retail Assistant** bridges this gap using a **hybrid routing architecture**:
1. **Deterministic Analytics**: Python & Pandas compute 100% accurate financial KPIs and aggregations across the complete dataset.
2. **Contextual RAG**: LangChain, local Sentence Transformers (`all-MiniLM-L6-v2`), and FAISS vector indexing retrieve relevant transaction records.
3. **Generative Explanation**: Hugging Face serverless LLMs (`meta-llama/Llama-3.1-8B-Instruct`) provide qualitative explanations grounded strictly in retrieved context.

---

## 📸 Application Showcase

![AI-Powered Retail Assistant Demo](assets/Demo.png)

### Key Interface Capabilities Shown Above:
* **Real-Time Health Status**: Automatically pings the backend Flask API and displays a live connection indicator (`API Online`).
* **One-Click Query Presets**: Pre-configured prompts for common executive questions (Revenue, Top Products, Leading Markets, Purchasing Trends).
* **Rich Visual Responses**:
  * **Metric Highlight Cards**: High-visibility summary widgets (e.g., Total Revenue: **£10,666,684.54**).
  * **Structured Data Tables**: Clean tabular rankings (e.g., Top 3 Products by Revenue).
  * **Response Tagging**: Visual badges distinguishing deterministic `Analytics`, semantic `RAG`, and combined `Hybrid` queries.

---

## 📐 System Architecture

```text
                                  ┌───────────────────────────┐
                                  │   Streamlit Web UI        │
                                  │   (frontend/app.py)       │
                                  └─────────────┬─────────────┘
                                                │ REST API (JSON)
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Flask API Gateway       │
                                  │   (backend/main.py)       │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Intent Router & Parser  │
                                  │   (backend/router.py)     │
                                  └──────┬──────┬──────┬──────┘
                                         │      │      │
                     ┌───────────────────┘      │      └───────────────────┐
                     │ "analytics"              │ "hybrid"                 │ "rag"
                     ▼                          ▼                          ▼
        ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
        │    Pandas Engine        │ │   Hybrid Coordinator    │ │   LangChain RAG Pipeline│
        │  (backend/analytics.py) │ │   Metric + Qualitative  │ │   (backend/rag.py)      │
        └────────────┬────────────┘ └───────────┬─────────────┘ └────────────┬────────────┘
                     │                          │                            │
                     ▼                          ▼                            ▼
        ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
        │ Exact Math Computation  │ │ 1. Deterministic Metric │ │ Local MiniLM Embedding  │
        │ Complete Dataset        │ │ 2. FAISS Similarity     │ │ FAISS Vector Search     │
        │ (530,000+ Transactions) │ │ 3. LLM Synthesis        │ │ Hugging Face LLM (Llama)│
        └─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
```

---

## 🛠 Tech Stack & Design Choices

| Component | Technology | Role & Justification |
| :--- | :--- | :--- |
| **Frontend UI** | **Streamlit** | Rapid, reactive data application framework featuring custom CSS glassmorphism, responsive chat feeds, and session state persistence. |
| **Backend Gateway**| **Flask & Flask-CORS** | Lightweight, stateless REST API handling query routing, health checks, and cross-origin communication. |
| **Analytics Core** | **Pandas & NumPy** | In-memory vectorized analytics computing deterministic sums, groupings, and rankings without database overhead. |
| **Embedding Model** | **Sentence Transformers** (`all-MiniLM-L6-v2`) | Local, CPU-efficient embeddings (384 dimensions) with zero external API latency or token cost. |
| **Vector Index** | **FAISS (CPU)** | High-performance similarity search for dense vector clustering and low-latency top-$k$ nearest neighbor retrieval. |
| **LLM Inference** | **Hugging Face Serverless Router** | Zero-cost inference using `meta-llama/Llama-3.1-8B-Instruct` with built-in multi-model failover resilience. |
| **Orchestration** | **LangChain** | Document transformations, vectorstore integration, and prompt template pipelining. |

---

## 💡 Query Handling Methodology

### 1. Analytics Route (Deterministic)
* **Trigger**: Queries matching numerical metrics, revenue calculations, or product/country rankings (e.g., *"What is total revenue?"*, *"Top 3 products by sales"*).
* **Pipeline**: Handled exclusively by Pandas operating over the full, cleaned 530,000+ transaction dataset.
* **Guarantee**: 0% hallucination risk, exact decimal precision, instantaneous response.

### 2. RAG Route (Semantic & Pattern Discovery)
* **Trigger**: Open-ended questions concerning behavior, product nuances, or sales trends (e.g., *"Describe customer purchasing behavior"*, *"What trends do you observe?"*).
* **Pipeline**: User query is embedded via `all-MiniLM-L6-v2` $\rightarrow$ FAISS retrieves top-$k$ most similar transaction narratives $\rightarrow$ Formats strict context prompt $\rightarrow$ LLM synthesizes observed patterns.
* **Guardrail**: Strict system prompt prohibiting the LLM from inventing figures or fabricating transactions outside the retrieved documents.

### 3. Hybrid Route (Synthesized Intelligence)
* **Trigger**: Inquiries asking "Why" or combining a quantitative figure with a request for explanation (e.g., *"Why is the UK leading?"*).
* **Pipeline**:
  1. Pandas computes the deterministic metric (e.g., UK total revenue: £9,025,222.08).
  2. FAISS retrieves representative transaction records from the target market.
  3. The LLM generates a qualitative breakdown explaining the factors driving that figure.
  4. The response seamlessly presents the verified number alongside the contextual explanation.

---

## 📦 Dataset Overview

* **Source**: [UCI Machine Learning Repository — Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
* **Scope**: Transnational dataset containing ~541,000 transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based non-store online retail firm.
* **Cleaning Pipeline** (`backend/data_loader.py`):
  * Removal of cancelled transactions (invoices starting with 'C').
  * Filtering out invalid/negative quantities and zero unit prices.
  * Imputation/normalization of missing customer identifiers.
  * Automatic feature engineering: `Revenue = Quantity * UnitPrice`.

---

## 🚀 Quick Start Guide

### Prerequisites
* **Python 3.10, 3.11, 3.12, or 3.13**
* **Git**
* Free **Hugging Face Account** ([Get access token](https://huggingface.co/settings/tokens))

---

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd "MINI AI Assistant"
```

### 2. Set Up Virtual Environment

**Windows (PowerShell / Git Bash):**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env     # macOS / Linux / Git Bash
copy .env.example .env   # Windows Command Prompt
```

Open `.env` and add your free Hugging Face token:
```ini
HF_TOKEN=hf_your_token_here
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
TOP_K=5
RAG_SAMPLE_SIZE=5000
DATA_PATH=data/Online Retail.xlsx
VECTORSTORE_PATH=vectorstore/
FLASK_PORT=5000
```
> *(Note: The deterministic Analytics features work immediately even without a Hugging Face token).*

### 5. Download Dataset & Build Vector Index

1. **Download Data**:
   ```bash
   python scripts/download_data.py
   ```
   *(Or place `Online Retail.xlsx` directly into the `data/` folder).*

2. **Build FAISS Vector Index**:
   ```bash
   python scripts/build_index.py
   ```
   This loads the dataset, performs stratified sampling (5,000 transactions), encodes document representations using Sentence Transformers, and creates the FAISS index in `vectorstore/`.

---

### 6. Launch the Application

Start the backend and frontend in two separate terminals:

#### Terminal 1 — Backend API:
```bash
python backend/main.py
```
* API Server starts at: `http://localhost:5000`
* Health Check: `http://localhost:5000/health`

#### Terminal 2 — Frontend UI:
```bash
streamlit run frontend/app.py
```
* Web Dashboard opens automatically at: `http://localhost:8501`

---

## 📡 REST API Reference

### `GET /health`
Verifies backend connectivity and service availability.

**Sample Response:**
```json
{
  "status": "ok"
}
```

---

### `POST /ask`
Primary inference endpoint. Automatically routes the query to the correct execution pipeline.

**Request Payload:**
```json
{
  "question": "What is total revenue?"
}
```

#### Example 1: Analytics Response
```json
{
  "question": "What is total revenue?",
  "type": "analytics",
  "answer": "The total revenue is £10,666,684.54.",
  "data": {
    "metric": "total_revenue",
    "value": 10666684.54,
    "currency": "GBP"
  }
}
```

#### Example 2: RAG Response
```json
{
  "question": "What trends do you observe?",
  "type": "rag",
  "answer": "Based on the retrieved transactions, purchasing patterns show strong wholesale and bulk buying for decorative giftware, with seasonal peaks in homeware accessories and consistent reordering from repeat corporate accounts.",
  "retrieved_documents": 5
}
```

#### Example 3: Hybrid Response
```json
{
  "question": "Why is UK leading?",
  "type": "hybrid",
  "answer": "United Kingdom Revenue: £9,025,222.08\n\nExplanation:\nThe retrieved transaction records demonstrate that the UK constitutes the core domestic market with high-frequency B2B orders, recurring wholesale shipments, and substantially lower per-unit freight costs compared to international buyers.",
  "data": {
    "country": "United Kingdom",
    "revenue": 9025222.08
  },
  "retrieved_documents": 5
}
```

---

## 🧪 Automated Testing Suite

The repository includes a comprehensive automated test suite covering analytics accuracy, API routing, exception handling, and offline fallbacks.

Run all tests via `pytest`:
```bash
pytest tests/ -v
```

### Test Coverage Highlights:
* **`tests/test_analytics.py`**: Tests calculations (revenue, top products, top country, regional breakdown, edge cases).
* **`tests/test_api.py`**: Integration tests for Flask endpoints, verifying request validation, JSON schemas, and error responses.
* **`tests/test_rag.py`**: Validates document formatting, stratified sampling, and graceful degradation when the LLM service is offline or tokens are missing.

---

## 📂 Repository Structure

```text
MINI AI Assistant/
│
├── assets/
│   └── screenshot.png          # UI showcase image used in README
│
├── backend/
│   ├── __init__.py
│   ├── main.py                 # Flask REST API server (/health, /ask)
│   ├── router.py               # Question classification & intent detection
│   ├── analytics.py            # Deterministic Pandas KPI functions
│   ├── rag.py                  # LangChain FAISS retrieval & HF router LLM
│   ├── data_loader.py          # Excel dataset ingestion & cleaning
│   └── config.py               # Environment configuration & defaults
│
├── frontend/
│   └── app.py                  # Streamlit chat application
│
├── scripts/
│   ├── build_index.py          # Script to generate FAISS vector index
│   └── download_data.py        # Dataset downloader script
│
├── tests/
│   ├── test_analytics.py       # Unit tests for Pandas metrics
│   ├── test_api.py             # Integration tests for Flask endpoints
│   └── test_rag.py             # RAG fallback & document generation tests
│
├── data/                       # Dataset directory (Online Retail.xlsx)
├── vectorstore/                # FAISS index files (index.faiss, index.pkl)
│
├── .env.example                # Template for environment variables
├── .gitignore                  # Git exclusion rules (protects .env & venv)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🛡️ Security & Privacy Guardrails

* **API Key Protection**: `.env` is explicitly included in `.gitignore` to prevent secret leakage.
* **Virtual Environment Safety**: Virtual environment folders (`venv/`, `.venv/`) are excluded from version control.
* **Local Embedding Execution**: Embeddings are computed strictly on the local CPU; customer records are never dispatched to third parties for embedding generation.
* **Data Sanitization**: Internal error tracebacks are logged locally and suppressed from client-facing JSON payloads to prevent information disclosure.

---

## 🔮 Future Roadmap

- [ ] **Dynamic Visualization**: Automatically render interactive Plotly bar/line charts inside Streamlit for analytical questions.
- [ ] **Multi-Turn Memory**: Add conversation buffer memory to support multi-step follow-up queries.
- [ ] **SQL Database Integration**: Port raw Excel transactions to DuckDB / SQLite for sub-millisecond analytical queries on larger datasets.
- [ ] **Export Reports**: One-click PDF / Excel executive summary export for generated insights.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
