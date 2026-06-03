# 🏗️ Model Architecture & Approach Explanation

**Project:** StayChatAI — Hotel RAG Chatbot  
**Stack:** Python · Google Gemini (LLM + Embeddings) · FAISS · FastAPI

---

## 1. High-Level System Architecture

```
                         ┌──────────────────────────────────────────────────┐
                         │           KNOWLEDGE BASE (hotel_kb.json)         │
                         │  38 JSON documents covering rooms, dining, spa,  │
                         │  policies, services, contacts, and commands       │
                         └────────────────────┬─────────────────────────────┘
                                              │  python app/ingest.py
                                              ▼
                         ┌──────────────────────────────────────────────────┐
                         │         INGESTION PIPELINE (one-time)            │
                         │  chunk_kb_file() → GeminiEmbedder (3072-dim)     │
                         │  → FAISSVectorStore.build_index() → .save()       │
                         └────────────────────┬─────────────────────────────┘
                                              │  artifacts/faiss_index/
                                              ▼
┌─────────────┐    HTTP    ┌─────────────────────────────────────────────────┐
│  Web Browser│ ◄────────► │         FastAPI Server (app/api.py)             │
│  Chat UI    │  POST/chat │  /health · /chat · /reset · GET / (serve HTML)  │
└─────────────┘            └────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                           ┌────────────────────────────────────────────────┐
                           │           HotelRAGBot (app/rag.py)             │
                           │         Main pipeline orchestrator             │
                           └──┬────────┬────────┬──────────┬────────────────┘
                              │        │        │          │
                    ┌─────────┘  ┌─────┘  ┌────┘   ┌──────┘
                    ▼            ▼         ▼         ▼
             IntentClassifier  Memory   Retriever  GuardrailsManager
             + Multilingual   (multi-  (FAISS +   (pre + post checks)
             (combined call)   turn)   Embedder)
```

---

## 2. RAG Pipeline — Step by Step

Every guest query passes through **9 sequential stages**:

### Stage 1 — Combined Query Analysis (Single LLM Call)
**Module:** `app/rag.py → _analyze_query_combined()`

Instead of making 3 separate API calls (classify → detect language → translate), all three tasks are combined into **one structured Gemini call** that returns a single JSON object:

```json
{
  "intent": "booking inquiry",
  "explanation": "Query is about room availability rules",
  "confidence": 0.97,
  "detected_language": "hinglish",
  "translated_query": "What are the check-in and check-out times?"
}
```

This optimization reduces API quota consumption by 67% compared to separate calls.

**Why combine?** These three tasks are semantically interdependent — language detection informs how to translate, and intent classification is aided by the English-normalized form. Combining them into one prompt produces more coherent results.

---

### Stage 2 — Staff Command Fast-Path
**Module:** `app/rag.py`, `app/guardrails.py`

If `intent == "staff command"`, the pipeline **immediately returns** a hardcoded denial string without performing any retrieval or generation:

```
Access Denied: Operational commands require secure staff credentials.
```

This is a deliberate architectural decision: **never let an LLM reason about a staff command** — the denial must be 100% deterministic and not subject to prompt injection manipulation.

---

### Stage 3 — Query Reformulation for Multi-Turn
**Module:** `app/memory.py → reformulate_query()`

For follow-up questions like *"What about breakfast?"* after asking about restaurants, the `ConversationMemory` module sends the full conversation history to Gemini and asks it to rewrite the follow-up as a self-contained query:

```
"What about breakfast?" → "What are the breakfast timings at The Grand Horizon Resort?"
```

This reformulated query (not the original) is used for FAISS retrieval, dramatically improving multi-turn accuracy.

---

### Stage 4 — Embedding and FAISS Retrieval
**Modules:** `app/embeddings.py`, `app/vectorstore.py`, `app/retriever.py`

1. The **English query** (translated + reformulated) is embedded using `models/gemini-embedding-001` with `task_type="retrieval_query"` — optimized for asymmetric retrieval where queries are short and documents are long.
2. The embedding (3072-dimensional float32 vector) is searched against the **FAISS IndexFlatL2** index, which returns top-`k` neighbors by Euclidean distance.
3. L2 distances are converted to cosine similarity: `cos_sim = 1 - (L2² / 2)` — valid because Gemini embeddings are unit-normalized.
4. Results below the **similarity threshold of 0.40** are discarded.

**Why FAISS IndexFlatL2?** For a 41-document index, exact search is fast enough (sub-millisecond). For larger KBs, `IndexIVFFlat` (inverted file index) would be preferred for speed.

---

### Stage 5 — Pre-Generation Sufficiency Guardrail
**Module:** `app/guardrails.py → verify_context_sufficiency()`

Before any generation occurs, the retrieved chunks are sent to Gemini with a strict compliance-auditor prompt:

> *"Do these retrieved chunks contain sufficient, direct facts to answer this query? Answer YES or NO only."*

If `NO` → safe refusal is returned immediately. This prevents the generation model from hallucinating when context is weak. Temperature is set to `0.0` for deterministic decisions.

---

### Stage 6 — Grounded Response Generation
**Module:** `app/rag.py → process_query()`

The generation prompt is carefully structured with:
- **System instructions** specifying language (respond in the guest's detected language)
- **Hard rules** against inventing prices, links, or unverified specifics
- **Retrieved context** (only the top-k filtered chunks — not the full KB)
- **Conversation history** for reference (multi-turn context)
- **Dual query** (original + English translation) for language-aware grounding

---

### Stage 7 — Regex Post-Generation Scan
**Module:** `app/guardrails.py → run_post_generation_scan()`

Three deterministic regex patterns scan the generated response:

| Pattern | Catches |
|---------|---------|
| `PRICE_REGEX` | `$120`, `INR 500`, `₹2500`, `Rs. 800` |
| `URL_REGEX` | `http://...`, `https://...`, `www....` |
| `EMAIL_REGEX` | Any `x@y.z` pattern |

For each match found, the system checks if that exact string appears verbatim in the source chunks. If not → response is immediately replaced with the safe refusal. This is a **deterministic, zero-false-negative layer** for the most critical hallucination types (prices and payment links).

---

### Stage 8 — LLM Grounding Verification
**Module:** `app/guardrails.py → run_post_generation_scan()`

A second Gemini call (temperature 0.0) compares the generated response against the retrieved chunks, looking for any unsupported factual claims the regex cannot catch (e.g., invented policies, wrong times, made-up room features). If verdict is `FAIL` → safe refusal is returned.

---

### Stage 9 — Memory Update and Response Return
**Module:** `app/memory.py`, `app/rag.py`

The original guest query and the final response are stored in the `ConversationMemory` (capped at `max_turns=5` to prevent context overflow). The result object returned includes the response, intent, language, grounding status, refused flag, and retrieved chunks — all visible in the debug panel and the web UI.

---

## 3. Anti-Hallucination Architecture (3 Layers)

```
                  Guest Query
                       │
             ┌─────────▼──────────┐
             │  LAYER 0: Intent   │  Staff Command → HARDCODED DENY
             │  Fast-Path Guard   │  (never reaches LLM)
             └─────────┬──────────┘
                       │
             ┌─────────▼──────────┐
             │  LAYER 1: Pre-Gen  │  "Is retrieved context sufficient?"
             │  Sufficiency Check │  YES → continue / NO → SAFE REFUSAL
             └─────────┬──────────┘
                       │
             ┌─────────▼──────────┐
             │  LAYER 2a: Regex   │  Price/URL/Email found in response
             │  Deterministic     │  but not in source → SAFE REFUSAL
             └─────────┬──────────┘
                       │
             ┌─────────▼──────────┐
             │  LAYER 2b: LLM     │  Semantic grounding check
             │  Verification      │  FAIL → SAFE REFUSAL / PASS → return
             └─────────┬──────────┘
                       │
                Final Response
              (Grounded or Refused)
```

**Why 3 layers?**
- Layer 0 is needed because LLMs can be tricked into reasoning about staff commands.
- Layer 1 is needed to prevent generation from starting on weak evidence.
- Layer 2a (regex) is needed because it's deterministic — prices/URLs are binary violations.
- Layer 2b (LLM) is needed to catch semantic hallucinations that regex cannot.

---

## 4. Intent Classifier Design

**Approach:** LLM-based (Gemini with JSON-mode output)

**Why LLM-based instead of rule-based?**  
The 5 intent categories have fuzzy boundaries — e.g., *"Can I bring my cat?"* is a booking inquiry (pet policy), not just general chat. Rule-based classifiers would require exhaustive keyword lists and would still fail on multilingual queries. An LLM understands semantics across all three languages natively.

**Validation:** The returned intent is validated against the allowed 5 labels. Any invalid label defaults to `other`. API exceptions also default to `other` with confidence 0.5.

**Labels:**
| Intent | Examples |
|--------|---------|
| `booking inquiry` | Room specs, check-in/out, cancellations, pet policy, capacity |
| `amenity question` | Pool/gym/spa times, Wi-Fi, parking, airport transfers, restaurants |
| `complaint` | Dirty room, slow service, billing issue, escalation request |
| `staff command` | `/system`, `/staff`, "set thermostat", "print guest list" |
| `other` | Greetings, general trivia, out-of-scope questions |

---

## 5. Multilingual Pipeline

```
Guest Query (any language)
        │
        ▼
_analyze_query_combined()
  ├── Detect: "hinglish"
  ├── Translate: "Kya hotel me wifi hai?" → "Is there WiFi in the hotel?"
  └── Intent: "amenity question"
        │
        ▼ (English translation used for FAISS retrieval)
   GeminiEmbedder.embed_query("Is there WiFi in the hotel?")
        │
        ▼ (FAISS returns English KB chunks)
   GuardrailsManager + Generation
        │
        ▼ (generation prompt instructs: "Respond in HINGLISH")
   "Ji haan! Hotel mein free Wi-Fi available hai. Network ka naam
    'GrandHorizon_Guest' hai..."
```

**Key insight:** Retrieval always happens in English (since the KB is English), but response generation is instructed to match the guest's detected language. This means the system works even if the KB were entirely in English while the guest speaks Hindi.

---

## 6. Knowledge Base Design

**Format:** JSON array of structured documents  
**Size:** 38 documents → 41 chunks after parsing  
**Embedding model:** `models/gemini-embedding-001` (3072 dimensions)  
**Chunk format:**
```
Category: AMENITIES
Title: Aura Spa - Services and Timings
Content: The Aura Spa is open daily from 9:00 AM to 8:00 PM...
```

The `Category + Title + Content` prefix is prepended to each chunk before embedding. This adds semantic context that improves embedding quality — the model understands "this is an amenities document about the spa" rather than just "spa open 9am".

**Deliberate KB gaps for trap questions:**
- No room pricing (prevents price hallucination)
- No GM's phone number (only email exists)
- No weather data (out of scope)
- No online payment links (explicitly prohibited by policy)

---

## 7. Technology Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| LLM | Google Gemini (`gemini-2.5-flash`) | Fast, cost-efficient, JSON-mode output, multilingual |
| Embeddings | `models/gemini-embedding-001` | 3072-dim, task-type-aware (doc vs query), same API |
| Vector DB | FAISS `IndexFlatL2` | Lightweight, no server required, persists to disk |
| Backend | FastAPI | Async, auto-generates Swagger docs, type-safe via Pydantic |
| Frontend | Vanilla HTML/CSS/JS | No build step, zero dependencies, instantly portable |
| Validation | Pydantic v2 | Schema validation for chunks, API schemas, intent models |
| Testing | pytest + `unittest.mock` | Fully offline (all Gemini calls mocked) |
