# 📝 Assumptions, Design Decisions & Known Limitations

**Project:** StayChatAI — Hotel RAG Chatbot

---

## Design Assumptions

### 1. Knowledge Base is Ground Truth
The hotel knowledge base (`data/raw/hotel_kb.json`) is treated as the **sole source of truth**. The bot will never use its pre-trained world knowledge to answer hotel-specific questions — even if the answer seems obvious. This is intentional: a real hotel deployment would need the system to reflect the hotel's specific policies, not generic hotel industry norms.

### 2. Room Pricing is Deliberately Absent
Room pricing is highly dynamic (seasonal rates, occupancy-based pricing, promotional packages). Including static prices would create a hallucination risk (outdated prices) or customer expectation issues. The KB explicitly instructs guests to contact reservations for pricing, and the bot relays this.

### 3. Single Hotel Instance
The system is built for one hotel property (The Grand Horizon Resort). The architecture supports multiple hotels — one would simply maintain separate FAISS indexes per property and instantiate separate `HotelRAGBot` instances. This was not implemented since the brief specified a single property.

### 4. English KB with Multilingual Queries
The knowledge base is authored in English. Multilingual support works by translating queries to English for retrieval, then generating responses back in the detected language. This is a deliberate trade-off: the KB only needs to be maintained in one language, while guests can still interact naturally in Hindi or Hinglish.

### 5. Free-Tier Gemini API Constraints
All development and testing was done on the **Gemini API free tier**, which allows 20 requests/minute. The retry system (`retry_on_quota_limit` in `utils.py`) and sleep delays in the evaluation script exist specifically to handle this constraint. A paid-tier deployment would remove these bottlenecks.

### 6. Similarity Threshold of 0.40
The minimum cosine similarity threshold for chunk retrieval was set to 0.40 after empirical testing on the eval set. A lower threshold would admit too much noise (weakly relevant chunks confusing the sufficiency guardrail). A higher threshold would over-refuse valid queries where the question uses different vocabulary than the KB.

### 7. Multi-turn Memory Capped at 5 Turns
Conversation memory is capped at 5 turns (10 total messages: 5 user + 5 bot). This prevents context windows from growing too large (which would increase cost and latency) while covering the vast majority of realistic guest conversations.

### 8. FAISS IndexFlatL2 for This Scale
`IndexFlatL2` (brute-force exact search) is used because the KB has 41 documents — exact search takes microseconds at this scale. For a KB of 10,000+ documents, `IndexIVFFlat` with a coarse quantizer would be more appropriate.

---

## Additional Design Decisions

### Combined Query Analysis (Single API Call)
Intent classification, language detection, and English translation are combined into **one Gemini API call** that returns a structured JSON. This was chosen over separate calls because:
- Reduces API quota consumption by ~67%
- The three tasks are semantically interdependent (language detection informs translation accuracy)
- Reduces latency (one network round-trip vs. three)

### Regex Layer Before LLM Validation
The post-generation guardrail applies a deterministic regex scan **before** the LLM grounding check. This is because:
- Regex is zero-cost, instantaneous, and has zero false negatives for prices/URLs
- If regex already blocks the response, we skip the LLM check (saving one API call)
- The LLM check catches semantic hallucinations that regex cannot

### Hardcoded Staff Command Denial
Staff command denial is entirely hardcoded — no LLM reasoning is involved. This prevents prompt injection attacks that might convince the model that a staff command is legitimate. The only possible output for a `staff command` intent is the exact hardcoded string.

### UTF-8 Stdout Patch for Windows
Windows terminals use cp1252 encoding by default, which cannot render Devanagari script (Hindi). The CLI and evaluation scripts patch `sys.stdout` to UTF-8 at startup to prevent crash errors when printing Hindi responses.

### Chunk Text Enrichment
Each chunk is formatted as:
```
Category: AMENITIES
Title: Aura Spa - Services and Timings
Content: [raw content]
```
Before embedding. This metadata prefix improves embedding quality because the model understands the document's semantic category and title, not just its content. This is sometimes called "rich text embeddings" or "document header augmentation."

---

## Known Limitations

### 1. No Real-Time Data
The KB is a static JSON file. The bot cannot answer questions that require real-time information:
- Current room availability
- Today's weather in Goa
- Live event schedules
- Current exchange rates at the currency desk

**Mitigation:** The guardrail correctly refuses these questions rather than inventing answers.

### 2. Hallucination is Not Eliminated — Only Minimized
The three-layer guardrail system significantly reduces hallucination, but no system based on LLMs can guarantee zero hallucination. Specifically:
- A sufficiently complex hallucination could pass the regex layer (if it doesn't mention prices/URLs) and the LLM grounding check (if it is similar enough to the context)
- The LLM grounding check is itself an LLM and is subject to its own reasoning errors

**Mitigation:** The pre-generation sufficiency check (Layer 1) is the most effective guard — it prevents generation entirely when context is insufficient, eliminating the hallucination opportunity.

### 3. Free-Tier Rate Limits Affect Demo Speed
On the free tier (20 requests/minute), each guest query involves 2–4 Gemini API calls. In back-to-back rapid usage, rate limits can cause 60-second delays. The retry system handles this gracefully but creates a slow user experience during demo.

**Mitigation in production:** A paid Gemini API tier would eliminate this. Additionally, the combined query analysis optimization reduces calls from 4 to 2 per query.

### 4. No User Authentication
The web UI has no login system. In a real hotel deployment, the bot would be embedded in the hotel's guest portal (where the guest is already authenticated) or would use session tokens for conversation isolation between multiple simultaneous guests.

### 5. Single Conversation Context Per Server Instance
The `HotelRAGBot` instance is a singleton on the server. In the current implementation, all users accessing `localhost:8000` share the same `ConversationMemory`. Calling `POST /reset` clears the memory for all users.

**Fix for production:** The memory should be session-scoped (e.g., stored per session token in Redis or a database).

### 6. Evaluation Scores Are API-Key Dependent
Because intent classification and grounding verification both use live Gemini API calls, evaluation scores will vary slightly between runs (LLMs are non-deterministic at temperature > 0). The evaluation script uses temperature 0.0 only for the guardrail decisions, not for intent classification, which has some natural variance.

### 7. No Voice / Accessibility Interface
The current implementation is text-only. A production-grade hotel assistant would benefit from voice input (speech-to-text) and text-to-speech output for accessibility and phone-channel support.

---

## Additional Notes

- The project was developed and tested entirely on **Windows 11** with Python 3.12. All path handling uses `pathlib.Path` for cross-platform compatibility.
- The `frontend/hero_bg.png` image was AI-generated specifically for this project.
- All 29 unit tests are fully offline and do not require an API key — they mock all Gemini calls using `unittest.mock.patch`.
- The submission folder (`submission/`) is committed to the repository as part of the project documentation, not as deployment code.
