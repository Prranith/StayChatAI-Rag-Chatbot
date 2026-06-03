import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Project directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "hotel_kb.json"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVAL_DATA_PATH = DATA_DIR / "eval" / "eval_set.json"
EVAL_RESULTS_PATH = DATA_DIR / "eval" / "eval_results.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FAISS_INDEX_DIR = ARTIFACTS_DIR / "faiss_index"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "eval").mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
