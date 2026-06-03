# ⚙️ Setup and Execution Instructions

**Project:** StayChatAI — Hotel RAG Chatbot  
**Repository:** https://github.com/Prranith/StayChatAI-Rag-Chatbot

---

## Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | 3.9+ | 3.11 or 3.12 recommended |
| pip | latest | Run `python -m pip install --upgrade pip` |
| Google Gemini API Key | Free tier | [Get yours here](https://aistudio.google.com/app/apikey) |
| Git | any | For cloning the repository |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Prranith/StayChatAI-Rag-Chatbot.git
cd StayChatAI-Rag-Chatbot
```

---

## Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
```
google-generativeai>=0.8.3   # Gemini LLM + embedding API
faiss-cpu>=1.9.0             # Vector similarity search
numpy>=1.24.0                # Array operations for FAISS
python-dotenv>=1.0.1         # .env file loader
pydantic>=2.0.0              # Data validation / schemas
pytest>=8.0.0                # Unit testing framework
fastapi>=0.111.0             # Web API framework
uvicorn[standard]>=0.29.0    # ASGI web server
```

---

## Step 3 — Configure Environment Variables

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` and fill in your API key:

```env
GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=models/gemini-embedding-001
LOG_LEVEL=INFO
```

> ⚠️ **Critical model names** (use exactly these):
> - ✅ `gemini-2.5-flash` — correct
> - ❌ `gemini-3.5-flash` — does not exist
> - ✅ `models/gemini-embedding-001` — correct
> - ❌ `models/embedding-001` — deprecated/wrong
> - ❌ `models/text-embedding-004` — may not be accessible on all accounts

---

## Step 4 — Run the Ingestion Pipeline

This step reads the hotel knowledge base, generates embeddings for all 38 documents,
and saves the FAISS index to disk. **Run this once before starting the chatbot.**

```bash
python app/ingest.py
```

**Expected output:**
```
[INFO] Starting ingestion pipeline...
[INFO] Successfully generated 41 chunks from hotel_kb.json
[INFO] Generating embeddings for 41 documents using models/gemini-embedding-001
[INFO] FAISS index built successfully with dimension 3072.
[INFO] FAISS index written to artifacts/faiss_index/index.faiss
[INFO] Chunks metadata written to artifacts/faiss_index/chunks.json
[INFO] Ingestion pipeline completed successfully!
```

> 💡 Re-run `python app/ingest.py` any time you modify `data/raw/hotel_kb.json`
> to add or update hotel documents.

---

## Step 5A — Run the Web App (Recommended)

This starts the FastAPI backend which also serves the premium chat UI.

```bash
python app/api.py
```

Then open your browser at:

```
http://localhost:8000
```

**API endpoints available:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the chat UI |
| `/health` | GET | Bot status + KB chunk count |
| `/chat` | POST | Send a message, get a full RAG response |
| `/reset` | POST | Clear conversation memory |
| `/docs` | GET | Auto-generated Swagger API documentation |

---

## Step 5B — Run the CLI Interface (Alternative)

For a terminal-based interaction without a browser:

```bash
python app/cli.py
```

**CLI commands:**
- Type any question in English, Hindi, or Hinglish and press Enter
- Type `clear` to reset conversation memory
- Type `exit` to quit

---

## Step 6 — Run the Evaluation Benchmark

Tests the bot across 15 curated test cases (including trap questions):

```bash
python app/evaluation.py
```

**Metrics reported:**
- Overall pass rate (intent + grounding + key points)
- Intent classifier accuracy
- Guardrail correctness (correct grounded/refused decisions)

Results are also saved to `data/eval/eval_results.json`.

> ⏳ The evaluation takes 5–15 minutes due to API rate limit sleep delays between cases.

---

## Step 7 — Run Unit Tests

All 29 unit tests run completely **offline** — no API key required (Gemini is mocked):

```bash
pytest tests/ -v
```

**Test files:**
| File | Tests | Coverage |
|------|-------|----------|
| `test_guardrails.py` | 12 | Regex detection, LLM mocking, grounded email pass-through |
| `test_retriever.py` | 8 | FAISS build/search/save/load, chunking, edge cases |
| `test_intents.py` | 9 | Intent labels, fallback, memory, multi-turn |

---

## Project Directory Structure

```
StayChatAI-Rag-Chatbot/
├── app/
│   ├── __init__.py       # Python package marker
│   ├── api.py            # FastAPI server (main web entrypoint)
│   ├── cli.py            # Terminal interface
│   ├── ingest.py         # Knowledge base ingestion pipeline
│   ├── rag.py            # HotelRAGBot — pipeline orchestrator
│   ├── chunking.py       # JSON KB → Chunk objects
│   ├── embeddings.py     # Gemini embedding wrapper
│   ├── vectorstore.py    # FAISS index (build/save/load/search)
│   ├── retriever.py      # Embed + search + threshold filter
│   ├── guardrails.py     # Anti-hallucination (regex + LLM)
│   ├── intents.py        # Intent classification (5 labels)
│   ├── memory.py         # Multi-turn conversation memory
│   ├── multilingual.py   # Language detection + translation
│   ├── config.py         # Paths, env vars, directory creation
│   └── utils.py          # Logger, retry_on_quota_limit
├── data/
│   ├── raw/hotel_kb.json         # 38-document hotel knowledge base
│   └── eval/eval_set.json        # 15 evaluation test cases
├── artifacts/
│   └── faiss_index/              # Generated by ingest.py (gitignored)
│       ├── index.faiss
│       └── chunks.json
├── frontend/
│   ├── index.html                # Premium chat UI (served by FastAPI)
│   └── hero_bg.png               # Hotel hero background image
├── tests/
│   ├── test_guardrails.py
│   ├── test_retriever.py
│   └── test_intents.py
├── submission/                   # This folder
│   ├── SUBMISSION_OVERVIEW.md
│   ├── ARCHITECTURE.md
│   ├── SETUP_AND_EXECUTION.md
│   └── ASSUMPTIONS_AND_NOTES.md
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `400 API key not valid` | Wrong or expired API key | Get a new key from aistudio.google.com |
| `403 Your project has been denied access` | Google Cloud project disabled | Create a new project and generate a fresh key |
| `404 model not found` | Wrong model name in `.env` | Use `gemini-2.5-flash` and `models/gemini-embedding-001` |
| `429 Quota exceeded` | Free tier rate limit hit | Wait 60 seconds and retry; the retry logic handles it automatically |
| `Index files missing` | Ingestion not run | Run `python app/ingest.py` first |
| `ModuleNotFoundError: app` | Running from wrong directory | Always run commands from the project root, not from `app/` |
