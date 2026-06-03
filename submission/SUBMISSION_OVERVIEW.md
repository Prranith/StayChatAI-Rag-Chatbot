# 📦 StayChatAI — Submission Package

**Candidate Repository:** https://github.com/Prranith/StayChatAI-Rag-Chatbot  
**Hotel Property:** The Grand Horizon Resort, Calangute, Goa  
**Stack:** Python · Google Gemini · FAISS · FastAPI · Vanilla HTML/CSS/JS  
**Submission Date:** June 2026

---

## 📋 Table of Contents

| # | Document | Description |
|---|----------|-------------|
| 1 | [ARCHITECTURE.md](./ARCHITECTURE.md) | Full model architecture and RAG pipeline explanation |
| 2 | [SETUP_AND_EXECUTION.md](./SETUP_AND_EXECUTION.md) | Step-by-step setup and run instructions |
| 3 | [ASSUMPTIONS_AND_NOTES.md](./ASSUMPTIONS_AND_NOTES.md) | Design assumptions, known limitations, and considerations |
| 4 | README (repo root) | Project overview, folder structure, evaluation guide |

---

## 🗂️ What Was Built

A **complete end-to-end Hotel RAG Chatbot** with:

- ✅ Real FAISS vector search over **41 knowledge base chunks**
- ✅ **Multi-layer anti-hallucination guardrail** (regex + LLM verification)
- ✅ **Intent classification** (5 labels: booking, amenity, complaint, staff command, other)
- ✅ **Multilingual support** (English · Hindi · Hinglish)
- ✅ **Multi-turn conversation memory** with context-aware query reformulation
- ✅ **15-case evaluation benchmark** with deliberate trap questions
- ✅ **29 offline unit tests** (guardrails, FAISS store, intent classifier, memory)
- ✅ **FastAPI REST backend** (`/chat`, `/health`, `/reset`)
- ✅ **Premium hotel-themed chat UI** (dark navy + gold design, source viewer, intent tags)

---

## 🔗 Quick Links

- **GitHub Repository:** https://github.com/Prranith/StayChatAI-Rag-Chatbot
- **Live Demo:** Run locally at `http://localhost:8000` after setup (see SETUP_AND_EXECUTION.md)
- **API Docs (auto-generated):** `http://localhost:8000/docs` (Swagger UI via FastAPI)
