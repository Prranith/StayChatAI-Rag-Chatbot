"""
Unit tests for the chunking pipeline and FAISS vector store.
These tests are fully offline — no API calls are made.
"""
import pytest
import json
from pathlib import Path
from app.vectorstore import FAISSVectorStore
from app.chunking import Chunk, chunk_kb_file, format_chunk_text


# ---------------------------------------------------------------------------
# Chunk formatting tests
# ---------------------------------------------------------------------------
def test_format_chunk_text_structure():
    """Verifies that format_chunk_text produces the expected prefixed format."""
    result = format_chunk_text("amenities", "Pool Hours", "Open 7am to 9pm.")
    assert "Category: AMENITIES" in result
    assert "Title: Pool Hours" in result
    assert "Content: Open 7am to 9pm." in result


def test_chunk_model_fields():
    """Verifies that a Chunk model holds fields correctly."""
    chunk = Chunk(
        id="test_001",
        text="Category: ROOMS\nTitle: Deluxe Room\nContent: Spacious room with balcony.",
        metadata={"title": "Deluxe Room", "category": "rooms"}
    )
    assert chunk.id == "test_001"
    assert "Deluxe Room" in chunk.text
    assert chunk.metadata["category"] == "rooms"


def test_chunk_kb_file_loads_correctly(tmp_path):
    """Verifies that chunk_kb_file correctly parses a sample JSON knowledge base."""
    # Write a mini KB JSON file
    kb_data = [
        {
            "id": "kb_test_001",
            "category": "amenities",
            "title": "Test Gym",
            "content": "Gym is open from 6am to 10pm.",
            "last_updated": "2026-01-01"
        },
        {
            "id": "kb_test_002",
            "category": "policies",
            "title": "Test Policy",
            "content": "No pets allowed.",
            "last_updated": "2026-01-01"
        }
    ]
    kb_path = tmp_path / "test_kb.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f)

    chunks = chunk_kb_file(kb_path)
    assert len(chunks) == 2

    # Check first chunk
    assert chunks[0].id == "kb_test_001"
    assert "Gym is open" in chunks[0].text
    assert chunks[0].metadata["category"] == "amenities"

    # Check second chunk
    assert chunks[1].id == "kb_test_002"
    assert chunks[1].metadata["title"] == "Test Policy"


def test_chunk_kb_file_raises_on_missing_file():
    """Verifies FileNotFoundError is raised when the KB file does not exist."""
    with pytest.raises(FileNotFoundError):
        chunk_kb_file(Path("/nonexistent/path/to/kb.json"))


# ---------------------------------------------------------------------------
# FAISSVectorStore tests
# ---------------------------------------------------------------------------
def test_vectorstore_build_and_search():
    """Builds a tiny FAISS index and verifies search returns ranked results."""
    chunks = [
        Chunk(id="c1", text="Gym is open from 6am to 10pm.", metadata={"title": "Gym"}),
        Chunk(id="c2", text="Pool is open from 7am to 9pm.", metadata={"title": "Pool"})
    ]
    # Use 3D unit vectors for predictable L2 distances
    embeddings = [
        [1.0, 0.0, 0.0],  # c1
        [0.0, 1.0, 0.0],  # c2
    ]

    store = FAISSVectorStore()
    store.build_index(chunks, embeddings)

    # Query near c1
    results = store.search([1.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    assert results[0]["chunk"].id == "c1"
    assert abs(results[0]["similarity"] - 1.0) < 1e-5

    # c2 should be the second result
    assert results[1]["chunk"].id == "c2"
    assert abs(results[1]["similarity"] - 0.0) < 1e-5


def test_vectorstore_mismatch_raises():
    """Verifies that mismatched chunk/embedding counts raise ValueError."""
    chunks = [Chunk(id="c1", text="text", metadata={})]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]  # 2 embeddings for 1 chunk

    store = FAISSVectorStore()
    with pytest.raises(ValueError, match="Mismatch"):
        store.build_index(chunks, embeddings)


def test_vectorstore_empty_raises():
    """Verifies that empty inputs raise ValueError."""
    store = FAISSVectorStore()
    with pytest.raises(ValueError):
        store.build_index([], [])


def test_vectorstore_save_and_load(tmp_path):
    """Builds, saves, then loads a FAISS index and verifies fidelity."""
    chunks = [
        Chunk(id="c1", text="Wifi is GrandHorizon_Guest.", metadata={"title": "Wi-Fi", "content": "Wifi is free."})
    ]
    embeddings = [[0.5, 0.5, 0.707]]

    store = FAISSVectorStore(index_dir=tmp_path)
    store.build_index(chunks, embeddings)
    store.save()

    # Verify files written
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "chunks.json").exists()

    # Load in a fresh instance
    loaded_store = FAISSVectorStore(index_dir=tmp_path)
    success = loaded_store.load()

    assert success
    assert len(loaded_store.chunks) == 1
    assert loaded_store.chunks[0].id == "c1"
    assert loaded_store.chunks[0].metadata["title"] == "Wi-Fi"


def test_vectorstore_search_before_load_raises():
    """Verifies that searching an unloaded store raises ValueError."""
    store = FAISSVectorStore()
    with pytest.raises(ValueError, match="not loaded"):
        store.search([1.0, 0.0, 0.0], k=3)
