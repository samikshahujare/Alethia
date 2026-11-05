# Alethia

A minimal, deterministic, and self-correcting Retrieval-Augmented Generation (RAG) engine designed for high-accuracy document intelligence and factual verification.

Alethia combines dense semantic search, sparse lexical retrieval, intelligent question decomposition, and an agentic verification loop to deliver grounded, hallucination-free answers with verifiable citations.

---

## Key Capabilities

- **Hybrid Retrieval**: Combines dense vector similarity (`all-MiniLM-L6-v2` via ChromaDB) with sparse lexical search (Okapi BM25) using Reciprocal Rank Fusion (RRF).
- **Targeted Document Resolution**: Detects queries referencing specific files (e.g. `abc.txt`, `report.pdf`) and scopes retrieval to avoid noise from unrelated documents.
- **Self-Correcting Verification Loop**: Orchestrated via LangGraph to evaluate factual faithfulness against retrieved passages and reformulate queries if evidence is missing.
- **Multi-Format Ingestion**: Ingests `.pdf`, `.txt`, and direct text notes with automatic sentence-aware chunking and duplicate prevention.
- **Minimalist Interface**: Dark-first Streamlit UI featuring stepped execution status, citation cards with similarity scores, and execution diagnostics.

---

## Architecture Overview

```
User Query
   │
   ├──> Document Target Detection (Optional Scoped Search)
   │
   ├──> Hybrid Retrieval (ChromaDB Vector Store + Okapi BM25 + RRF)
   │
   ├──> LangGraph Cyclic Workflow:
   │       ├── Generate Draft (Groq Llama 3)
   │       ├── Judge Faithfulness (Strict Fact Checking)
   │       └── Targeted Rewrite & Re-retrieval (Max 2 Attempts)
   │
   └──> Verified Response + Source Citations + Execution Diagnostics
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Orchestration** | LangGraph (Cyclic State Machine) |
| **LLM Inference** | Groq (`llama-3.3-70b-versatile`) |
| **Vector Store** | ChromaDB (Cosine similarity metric) |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`, 384d) |
| **Sparse Retrieval** | Okapi BM25 (Pure-Python implementation) |
| **PDF Extraction** | `pypdf` |
| **Frontend UI** | Streamlit (Custom Dark SaaS Theme) |

---

## Getting Started

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/alethia.git
cd alethia

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```ini
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Start the Web App

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

---

## Project Structure

```
Alethia/
├── app.py                     # Streamlit application entrypoint
├── src/
│   ├── config.py              # Configuration and environment validation
│   ├── document_loader.py     # Document loaders (PDF, TXT, text input)
│   ├── text_cleaner.py        # Text normalization and formatting
│   ├── chunker.py             # Sentence-aware document chunking
│   ├── store.py               # ChromaDB vector store wrapper
│   ├── bm25.py                # Okapi BM25 index and tokenizer
│   ├── hybrid_retriever.py    # Hybrid sparse-dense fusion via RRF
│   ├── document_inspection.py # Targeted document reference resolution
│   ├── decomposition.py       # Multi-intent question decomposition
│   ├── prompts.py             # LLM system and evaluation prompts
│   ├── nodes.py               # LangGraph workflow nodes
│   ├── graph.py               # Compiled cyclic graph pipeline
│   └── llm.py                 # Groq client wrapper with error handling
├── ui/
│   ├── styles.py              # Minimalist dark theme CSS
│   ├── sidebar.py             # Ingestion, document manager & specs
│   ├── chat.py                # Chat interface & progressive status
│   ├── sources.py             # Citation cards with match percentages
│   └── trace.py               # Execution diagnostics timeline
└── tests/                     # Unit and integration test suite
```

---

## Running Tests

Execute the automated test suite with pytest:

```bash
pytest tests/
```

---

## License

MIT License.
