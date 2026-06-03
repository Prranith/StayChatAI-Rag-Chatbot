import sys
from pathlib import Path
# Add project root to sys.path to allow running as script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import RAW_DATA_PATH, FAISS_INDEX_DIR
from app.chunking import chunk_kb_file
from app.embeddings import GeminiEmbedder
from app.vectorstore import FAISSVectorStore
from app.utils import get_logger

logger = get_logger("ingest_pipeline")

def run_ingestion() -> None:
    """Executes the complete knowledge base ingestion pipeline."""
    logger.info("Starting ingestion pipeline...")
    
    # 1. Chunk knowledge base
    try:
        chunks = chunk_kb_file(RAW_DATA_PATH)
    except FileNotFoundError:
        logger.error(f"Raw data file not found at {RAW_DATA_PATH}. Make sure to create it first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to chunk knowledge base: {e}")
        sys.exit(1)
        
    # 2. Initialize embedder and generate embeddings
    try:
        embedder = GeminiEmbedder()
        texts_to_embed = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_documents(texts_to_embed)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        sys.exit(1)
        
    # 3. Create FAISS vector store, build, and save index
    try:
        vectorstore = FAISSVectorStore(FAISS_INDEX_DIR)
        vectorstore.build_index(chunks, embeddings)
        vectorstore.save()
        logger.info(f"Ingestion pipeline completed successfully! Index saved to {FAISS_INDEX_DIR}")
    except Exception as e:
        logger.error(f"Failed to build or save vector store: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_ingestion()
