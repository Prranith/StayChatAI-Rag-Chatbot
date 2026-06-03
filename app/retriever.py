from typing import List, Dict, Any
from app.embeddings import GeminiEmbedder
from app.vectorstore import FAISSVectorStore
from app.config import FAISS_INDEX_DIR
from app.utils import get_logger

logger = get_logger(__name__)

class Retriever:
    """Coordinates embedding generation and vector search with similarity filtering."""
    
    def __init__(self, vectorstore: FAISSVectorStore = None, embedder: GeminiEmbedder = None):
        if vectorstore is None:
            self.vectorstore = FAISSVectorStore(FAISS_INDEX_DIR)
            self.vectorstore.load()
        else:
            self.vectorstore = vectorstore
            
        self.embedder = embedder or GeminiEmbedder()

    def retrieve(self, query: str, k: int = 3, threshold: float = 0.40) -> List[Dict[str, Any]]:
        """Retrieves and filters documents based on query similarity.
        
        Args:
            query: The English query to search for.
            k: Number of documents to retrieve.
            threshold: Minimum cosine similarity score (1.0 is identical, 0.0 is orthogonal).
            
        Returns:
            List of dicts containing the chunk, distance, and similarity score, 
            filtered by the threshold.
        """
        logger.info(f"Retrieving top {k} chunks for query: '{query}' with threshold {threshold}")
        
        try:
            # 1. Embed query
            query_embedding = self.embedder.embed_query(query)
            
            # 2. Search FAISS vector store
            all_results = self.vectorstore.search(query_embedding, k=k)
            
            # 3. Filter results by threshold
            filtered_results = []
            for res in all_results:
                similarity = res["similarity"]
                chunk_id = res["chunk"].id
                logger.info(f"Retrieved chunk {chunk_id} | Similarity: {similarity:.4f} | Title: '{res['chunk'].metadata.get('title')}'")
                
                if similarity >= threshold:
                    filtered_results.append(res)
                else:
                    logger.debug(f"Filtered out chunk {chunk_id} due to low similarity: {similarity:.4f} < {threshold}")
                    
            logger.info(f"Retrieved {len(filtered_results)} chunks after similarity threshold filtering.")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
