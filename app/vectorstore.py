import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss

from app.chunking import Chunk
from app.config import FAISS_INDEX_DIR
from app.utils import get_logger

logger = get_logger(__name__)

class FAISSVectorStore:
    """Manages the creation, saving, loading, and querying of the local FAISS index."""
    
    def __init__(self, index_dir: Path = FAISS_INDEX_DIR):
        self.index_dir = index_dir
        self.index_path = index_dir / "index.faiss"
        self.metadata_path = index_dir / "chunks.json"
        self.index = None
        self.chunks: List[Chunk] = []

    def build_index(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Builds a new FAISS index from scratch using embeddings and chunks."""
        logger.info(f"Building FAISS index with {len(chunks)} documents")
        if not chunks or not embeddings:
            raise ValueError("Chunks and embeddings lists cannot be empty.")
            
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks, but {len(embeddings)} embeddings.")
            
        self.chunks = chunks
        
        # Convert embeddings to float32 numpy array
        emb_array = np.array(embeddings).astype('float32')
        dimension = emb_array.shape[1]
        
        # Build IndexFlatL2 (simple Euclidean distance index)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(emb_array)
        
        logger.info(f"FAISS index built successfully with dimension {dimension}.")

    def save(self) -> None:
        """Persists the FAISS index and the corresponding chunks list to local files."""
        if self.index is None or not self.chunks:
            raise ValueError("No index built to save.")
            
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"FAISS index written to {self.index_path}")
        
        # Save chunk metadata as JSON
        serialized_chunks = [chunk.model_dump() for chunk in self.chunks]
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(serialized_chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Chunks metadata written to {self.metadata_path}")

    def load(self) -> bool:
        """Loads a persisted FAISS index and metadata chunks list from disk."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning(f"Index or metadata files missing in {self.index_dir}. Load skipped.")
            return False
            
        try:
            # Read index
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index from {self.index_path}")
            
            # Read chunks
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                serialized_chunks = json.load(f)
            self.chunks = [Chunk(**c) for c in serialized_chunks]
            logger.info(f"Loaded {len(self.chunks)} chunks from {self.metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}")
            self.index = None
            self.chunks = []
            return False

    def search(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """Searches the index for the top-k closest matches to the query embedding."""
        if self.index is None or not self.chunks:
            raise ValueError("Vector store is not loaded or has no indexed documents.")
            
        # Convert embedding to numpy array with shape (1, dimension)
        query_np = np.array([query_embedding]).astype('float32')
        
        # Query FAISS index
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx]
            # Convert L2 distance to Cosine Similarity score.
            # L2 distance for normalized vectors is d^2 = 2 * (1 - cos_sim)
            # So, cos_sim = 1 - (d^2 / 2)
            cosine_similarity = float(1.0 - (dist / 2.0))
            
            results.append({
                "chunk": chunk,
                "distance": float(dist),
                "similarity": cosine_similarity
            })
            
        logger.debug(f"Vector search returned {len(results)} matches.")
        return results
