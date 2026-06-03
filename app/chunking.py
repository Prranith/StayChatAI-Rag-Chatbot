import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from app.utils import get_logger

logger = get_logger(__name__)

class Chunk(BaseModel):
    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Normalized text content for vector indexing")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary (source, category, title)")

def format_chunk_text(category: str, title: str, content: str) -> str:
    """Formats the chunk text to include metadata for richer semantic embedding."""
    return f"Category: {category.upper()}\nTitle: {title}\nContent: {content}"

def chunk_kb_file(file_path: Path) -> List[Chunk]:
    """Loads a JSON knowledge base and parses it into Chunks."""
    logger.info(f"Loading knowledge base file from {file_path}")
    if not file_path.exists():
        logger.error(f"Knowledge base file does not exist at {file_path}")
        raise FileNotFoundError(f"Knowledge base file not found at {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    chunks = []
    for item in data:
        item_id = item.get("id", "")
        category = item.get("category", "general")
        title = item.get("title", "")
        content = item.get("content", "")
        
        # Construct raw chunk text
        chunk_text = format_chunk_text(category, title, content)
        
        # Build metadata
        metadata = {
            "id": item_id,
            "category": category,
            "title": title,
            "content": content,  # original content without styling prefix
            "last_updated": item.get("last_updated", "")
        }
        
        chunk = Chunk(
            id=item_id,
            text=chunk_text,
            metadata=metadata
        )
        chunks.append(chunk)
        
    logger.info(f"Successfully generated {len(chunks)} chunks from {file_path}")
    return chunks
