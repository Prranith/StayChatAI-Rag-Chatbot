# 🏨 The Grand Horizon Resort — AI Guest Assistant (RAG + FAISS)

> A production-grade, multilingual Hotel RAG Chatbot built with **Python + Google Gemini + FAISS**.

---

## 📌 Project Overview

This system is an intelligent AI-powered guest assistant for *The Grand Horizon Resort*, a 5-star luxury property in Calangute, Goa. It answers guest queries in English, Hindi, and Hinglish by retrieving facts exclusively from a curated hotel knowledge base — it **never invents answers**.

**Key highlights:**
- Real FAISS-based semantic vector search (not keyword matching, not whole-KB stuffing)
- Multi-layer anti-hallucination guardrail (regex + LLM verification)
- Multilingual support: English, Hindi (Devanagari), Hinglish (Latin script)
- Intent classification across 5 hotel-specific categories
- Multi-turn conversation memory with query reformulation
- 15-case evaluation benchmark with deliberate trap questions
- Clean, modular architecture designed to be explained in a live interview

---

## 🏗️ Architecture

```
Guest Query (English / Hindi / Hinglish)
          │
          ▼
┌──────────────────────────────┐
│ 1. Combined Query Analysis   │  Single Gemini call:
│    (Intent + Lang + Transl.) │  classify intent, detect language,
└─────────────┬────────────────┘  translate to English for retrieval
              │
              ├──[staff command]──► IMMEDIATE DENIAL (no retrieval)
              │
              ▼
┌──────────────────────────────┐
│ 2. Query Reformulation       │  ConversationMemory makes follow-up
│    (Multi-turn Context)      │  queries self-contained for FAISS search
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ 3. FAISS Vector Retrieval    │  Embed English query → search index
│    (Retriever + Embedder)    │  → filter by similarity threshold (0.40)
└─────────────┬────────────────┘
              │
              ├──[no results]──► SAFE REFUSAL
              │
              ▼
┌──────────────────────────────┐
│ 4. Guardrail: Sufficiency    │  LLM checks if retrieved chunks
│    Check                     │  actually contain enough to answer
└─────────────┬────────────────┘
              │
              ├──[insufficient]──► SAFE REFUSAL
              │
              ▼
┌──────────────────────────────┐
│ 5. Response Generation       │  Gemini generates grounded answer
│    (context-constrained)     │  strictly from retrieved chunks only
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ 6. Post-Generation Scan      │  Regex: block invented prices/URLs
│    (Regex + LLM Validation)  │  LLM: verify no unsupported facts
└─────────────┬────────────────┘
              │
              ▼
       Final Response
  (Grounded or Safe Refusal)
```

---

## 📁 Project Structure

```
Project/
├── app/
│   ├── __init__.py          # Python package marker
│   ├── config.py            # Paths, env vars, directory setup
│   ├── ingest.py            # Ingestion pipeline entrypoint
│   ├── chunking.py          # JSON KB to Chunk objects
│   ├── embeddings.py        # GeminiEmbedder (doc + query embeddings)
│   ├── vectorstore.py       # FAISSVectorStore (build, save, load, search)
│   ├── retriever.py         # Retriever (embed + search + threshold filter)
│   ├── intents.py           # IntentClassifier (5 labels via Gemini)
│   ├── rag.py               # HotelRAGBot (pipeline orchestrator)
│   ├── guardrails.py        # GuardrailsManager (regex + LLM double-check)
│   ├── memory.py            # ConversationMemory (multi-turn + reformulation)
│   ├── multilingual.py      # MultilingualHandler (detect + translate)
│   ├── evaluation.py        # Evaluation benchmark runner
│   ├── cli.py               # Interactive CLI interface
│   └── utils.py             # Logger, retry_on_quota_limit
├── data/
│   ├── raw/
│   │   └── hotel_kb.json    # Hotel knowledge base (38 documents)
│   └── eval/
│       ├── eval_set.json    # 15 evaluation test cases
│       └── eval_results.json # Auto-generated evaluation report
├── artifacts/
│   └── faiss_index/
│       ├── index.faiss      # Persisted FAISS index
│       └── chunks.json      # Serialized chunk metadata
├── tests/
│   ├── test_guardrails.py   # Anti-hallucination tests (offline, mocked)
│   ├── test_retriever.py    # FAISS and chunking tests (offline)
│   └── test_intents.py      # Intent classifier and memory tests (offline)
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🧠 Knowledge Base

The hotel KB (`data/raw/hotel_kb.json`) contains **38 richly detailed documents** covering:

| Category | Topics |
|----------|--------|
| Overview | Location, awards, eco-certification |
| Rooms (5 types) | Deluxe OV Room, Garden Pool Villa, Presidential Suite, Club Room, Superior Garden View |
| Policies | Check-in/out, cancellation, pets, children, smoking, payment, complaints |
| Amenities | Pool, gym, spa (Aura), Wi-Fi, yoga & wellness |
| Dining | Azure (fine dining), The Tide (all-day), Spice Route, Horizon Sky Bar, Poolside Bar |
| Services | Concierge, room service, airport transfer, laundry, business centre, kids club, water sports, weddings, accessibility |
| Contacts | All departmental extensions and email addresses |

**Deliberate gaps for anti-hallucination testing:**
- Exact room pricing (absent → bot must refuse to invent)
- GM's personal phone number (only email listed)
- Real-time weather (out of scope)
- Online payment links (explicitly prohibited by policy)

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9+
- A valid [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux
```

Edit `.env`:

```env
GEMINI_API_KEY=YOUR_ACTUAL_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=models/gemini-embedding-001
LOG_LEVEL=INFO
```

> **Important:** Use exactly `gemini-2.5-flash` and `models/gemini-embedding-001`.
> Do NOT use `gemini-3.5-flash` (non-existent) or `models/embedding-001` (deprecated).

### 3. Run ingestion pipeline

```bash
python app/ingest.py
```

This reads `hotel_kb.json`, generates embeddings for all 38 documents, and saves the FAISS index. Run this once (or whenever the KB is updated).

---

## 🚀 Run the Chatbot

```bash
python app/cli.py
```

Commands:
- Type any question in English, Hindi, or Hinglish
- Type `clear` to reset conversation memory
- Type `exit` to quit

---

## 📊 Run Evaluation

```bash
python app/evaluation.py
```

This runs all 15 test cases and prints a summary report. Results are also saved to `data/eval/eval_results.json`.

Sample metrics output:
```
================================================================================
                             METRICS SUMMARY
================================================================================
Total Test Cases Run      : 15
Overall Pass Rate         : 13/15 (86.67%)
Intent Classifier Accuracy: 14/15 (93.33%)
Guardrail Correctness     : 15/15 (100.00%)
================================================================================
```

---

## 🧪 Run Unit Tests

All tests are fully offline — no API key required (Gemini calls are mocked).

```bash
pytest tests/ -v
```

Test coverage:
- `test_guardrails.py` — 12 tests: regex detection for prices/URLs/emails, LLM grounding check, safe refusal constants
- `test_retriever.py` — 8 tests: FAISS build/search/save/load, chunk parsing, edge cases
- `test_intents.py` — 9 tests: intent classification labels, fallback behavior, memory management

---

## 🛡️ How Anti-Hallucination Works

The guardrail system has **three independent layers**, all of which must be passed:

### Layer 1: Staff Command Fast-Path
Before any retrieval, if intent is classified as `staff command`, the system returns a hardcoded denial string immediately. No LLM generation occurs.

### Layer 2: Retrieval Sufficiency Check (Pre-Generation)
After FAISS retrieval, the `GuardrailsManager.verify_context_sufficiency()` method sends the retrieved chunks and query to Gemini with a strict prompt asking only YES or NO: *"Do these chunks contain enough to answer the query?"*
If the answer is NO → safe refusal is returned before any generation occurs.

### Layer 3: Post-Generation Scan (Post-Generation)
After generation, `run_post_generation_scan()` applies two checks:

**3a. Regex Scan (deterministic):**
- `PRICE_REGEX`: Detects any currency symbol or price pattern (e.g., `$120`, `INR 500`, `₹2500`)
- `URL_REGEX`: Detects HTTP/HTTPS links or `www.` patterns
- `EMAIL_REGEX`: Detects email addresses

Any match is cross-referenced against the source chunks. If the detected value does NOT appear verbatim in the source chunks, the response is immediately replaced with the safe refusal message.

**3b. LLM Grounding Verification (semantic):**
A separate Gemini call compares the generated response with the retrieved context, checking for any unsupported factual claims beyond what regex can detect. If the verdict is FAIL → safe refusal.

---

## 🌐 How Multilingual Handling Works

1. The `_analyze_query_combined()` method in `rag.py` uses a single Gemini call to simultaneously detect language (`english`, `hindi`, or `hinglish`) and translate the query to English.
2. The English translation is used for FAISS embedding and retrieval (the KB is in English).
3. The `detected_language` is passed to the generation prompt, which instructs Gemini to respond in the same language as the original query.
4. For Hinglish, Gemini responds using Hindi grammar and vocabulary written in Latin script (e.g., "Aapka check-in time 3:00 PM hai...").

---

## 🎯 How Intent Classification Works

The `IntentClassifier` sends the user query to Gemini with a structured prompt asking it to return a JSON object with:
- `intent`: one of `booking inquiry`, `amenity question`, `complaint`, `staff command`, `other`
- `explanation`: reasoning for the classification
- `confidence`: score from 0.0 to 1.0

This is validated against the allowed intent list. Invalid responses default to `other`. The intent is used for:
- Routing `staff command` queries to the hardcoded denial response
- Informational display in the debug panel and evaluation metrics

---

## 🔄 How to Add New Hotel Documents

1. Open `data/raw/hotel_kb.json`
2. Add a new JSON object following this schema:
   ```json
   {
     "id": "kb_unique_id",
     "category": "category_name",
     "title": "Document Title",
     "content": "Full document content here...",
     "last_updated": "YYYY-MM-DD"
   }
   ```
3. Re-run ingestion: `python app/ingest.py`
4. The new FAISS index will include the new document.

---

## ⚠️ What Happens for Unsupported Questions

If a guest asks a question whose answer is not in the KB (e.g., "What is the price of the Deluxe Room?"), the system will respond with:

> *"I apologize, but I don't have that specific information in my knowledge base. I can connect you with a member of our front desk team to assist you further."*

This behavior is enforced by the two-stage guardrail: the sufficiency check prevents the LLM from even attempting to answer, and the post-generation scan catches any slippage.

---

## 📝 Key Design Decisions & Assumptions

1. **Single combined API call for analysis**: Intent classification, language detection, and translation are performed in one Gemini call (`_analyze_query_combined`). This reduces latency and API quota consumption compared to three separate calls.

2. **FAISS IndexFlatL2 with cosine approximation**: We use an L2 (Euclidean) index but convert distances to cosine similarity using the formula `cos_sim = 1 - (L2^2 / 2)`. This works correctly because Gemini embeddings are unit-normalized.

3. **Similarity threshold of 0.40**: Chunks with cosine similarity below 0.40 are discarded before being sent to the guardrail. This prevents very low-relevance chunks from confusing the sufficiency checker.

4. **Retry with exponential backoff**: All Gemini API calls are wrapped with `retry_on_quota_limit()`, which retries up to 5 times with exponential backoff (6s, 12s, 24s, 48s, 60s) for 429 rate limit errors. Other errors are re-raised immediately.

5. **Pricing is deliberately absent from the KB**: Room pricing is highly dynamic and out-of-scope for a chatbot. The KB explicitly instructs guests to contact reservations for pricing, which the bot will relay.

6. **UTF-8 stdout encoding**: On Windows, the default terminal encoding (cp1252) cannot render Devanagari Hindi script. The CLI and evaluation scripts patch `sys.stdout` to UTF-8 at startup.

---

## 🚧 Known Limitations

- **Free-tier API rate limits**: The Gemini free tier allows 20 requests/minute. The evaluation script includes sleep delays, but running it rapidly may still hit quota limits.
- **Hallucination possible under extreme edge cases**: The LLM grounding check is probabilistic. While the regex layer is deterministic, a sufficiently clever hallucination could in theory pass. This is a known limitation of all LLM-based systems.
- **No real-time data**: The KB is static. The bot cannot answer questions about current room availability, live pricing, or today's weather.
- **No authentication**: The CLI has no user authentication. In production, this would be added before deployment.
