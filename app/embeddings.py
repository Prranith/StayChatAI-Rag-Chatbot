import time
from typing import List
import google.generativeai as genai
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL
from app.utils import get_logger

logger = get_logger(__name__)

class GeminiEmbedder:
    """Wrapper for generating vector embeddings using Google Generative AI API."""
    
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = EMBEDDING_MODEL):
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set. Embedding calls will fail if not using local fallback.")
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: List[str], batch_size: int = 16, retries: int = 3) -> List[List[float]]:
        """Generates embeddings for a list of document strings with batching and retries."""
        logger.info(f"Generating embeddings for {len(texts)} documents using {self.model_name}")
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(retries):
                try:
                    # Task type 'retrieval_document' is optimized for indexing
                    response = genai.embed_content(
                        model=self.model_name,
                        content=batch,
                        task_type="retrieval_document"
                    )
                    embeddings.extend(response.get("embedding", []))
                    break
                except Exception as e:
                    logger.warning(f"Embedding batch failed (attempt {attempt + 1}/{retries}): {e}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)  # exponential backoff
                    else:
                        logger.error("Failed to generate embeddings after all retries.")
                        raise e
                        
        return embeddings

    def embed_query(self, text: str, retries: int = 3) -> List[float]:
        """Generates embedding for a single query string with retries."""
        logger.debug(f"Generating query embedding for: '{text[:50]}...'")
        for attempt in range(retries):
            try:
                # Task type 'retrieval_query' is optimized for search queries
                response = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_query"
                )
                embeddings = response.get("embedding", [])
                if embeddings:
                    # embed_content returns a list of float for a single text, or a list of list.
                    # Let's ensure we return a flat list.
                    if isinstance(embeddings[0], list):
                        return embeddings[0]
                    return embeddings
                raise ValueError("No embedding returned from API.")
            except Exception as e:
                logger.warning(f"Query embedding failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Failed to generate query embedding after all retries.")
                    raise e
