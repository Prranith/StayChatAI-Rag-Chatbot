"""
FastAPI backend for The Grand Horizon Resort AI Guest Assistant.

Run with:
    python app/api.py
    -- or --
    uvicorn app.api:app --reload --port 8000
"""
import sys
import io
from pathlib import Path

# UTF-8 fix for Windows console output
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on sys.path when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.rag import HotelRAGBot
from app.utils import get_logger

logger = get_logger("api_server")

# ──────────────────────────────────────────────
# App initialization
# ──────────────────────────────────────────────
app = FastAPI(
    title="Grand Horizon Resort – AI Guest Assistant API",
    description="RAG-powered hotel chatbot with anti-hallucination guardrails",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Bot singleton (loaded once at startup)
# ──────────────────────────────────────────────
bot: Optional[HotelRAGBot] = None
bot_load_error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    global bot, bot_load_error
    logger.info("Loading HotelRAGBot at startup...")
    try:
        bot = HotelRAGBot()
        logger.info("HotelRAGBot loaded successfully.")
    except Exception as e:
        bot_load_error = str(e)
        logger.error(f"Failed to load HotelRAGBot: {e}")


# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class ChunkInfo(BaseModel):
    title: str
    snippet: str
    similarity: float


class ChatResponse(BaseModel):
    response: str
    intent: str
    intent_explanation: str
    language: str
    grounded: bool
    refused: bool
    retrieved_chunks: List[ChunkInfo]


class HealthResponse(BaseModel):
    status: str
    bot_loaded: bool
    kb_chunks: int
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns the health status of the API and the number of indexed KB chunks."""
    if bot is None:
        return HealthResponse(
            status="degraded",
            bot_loaded=False,
            kb_chunks=0,
            error=bot_load_error or "Bot not initialized"
        )
    kb_size = len(bot.retriever.vectorstore.chunks) if bot.retriever.vectorstore else 0
    return HealthResponse(
        status="ok",
        bot_loaded=True,
        kb_chunks=kb_size
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Process a guest query through the full RAG pipeline and return a structured response."""
    if bot is None:
        raise HTTPException(
            status_code=503,
            detail=f"Bot not initialized. {bot_load_error or 'Run ingestion first: python app/ingest.py'}"
        )

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    logger.info(f"Received chat request: '{request.message[:80]}'")

    result = bot.process_query(request.message)

    # Serialize retrieved chunks for the frontend
    serialized_chunks: List[ChunkInfo] = []
    for res in result.get("retrieved_chunks", []):
        chunk = res["chunk"]
        serialized_chunks.append(ChunkInfo(
            title=chunk.metadata.get("title", "Unknown"),
            snippet=chunk.metadata.get("content", "")[:200] + "...",
            similarity=round(res["similarity"], 4)
        ))

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        intent_explanation=result["intent_explanation"],
        language=result["language"],
        grounded=result["grounded"],
        refused=result["refused"],
        retrieved_chunks=serialized_chunks,
    )


@app.post("/reset", tags=["Chat"])
async def reset_conversation():
    """Clears the bot's conversation memory for a fresh session."""
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot not initialized.")
    bot.memory.clear()
    logger.info("Conversation memory reset via API.")
    return {"status": "ok", "message": "Conversation memory cleared."}


# ──────────────────────────────────────────────
# Serve static frontend
# ──────────────────────────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", tags=["Frontend"])
    async def serve_frontend():
        """Serves the chat UI frontend."""
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Frontend not found.")


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
