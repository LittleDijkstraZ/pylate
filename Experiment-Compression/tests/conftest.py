"""Shared test fixtures and configuration."""

import json
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture
def tmp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    return [
        torch.randn(10, 128),  # 10 tokens, 128 dims
        torch.randn(8, 128),   # 8 tokens, 128 dims
        torch.randn(12, 128),  # 12 tokens, 128 dims
    ]


@pytest.fixture
def sample_documents():
    """Create sample documents."""
    return [
        {"id": "doc_1", "text": "This is the first document."},
        {"id": "doc_2", "text": "This is the second document."},
        {"id": "doc_3", "text": "This is the third document."},
    ]


@pytest.fixture
def sample_queries():
    """Create sample queries."""
    return {
        "q_1": "What is the first?",
        "q_2": "What is the second?",
    }


@pytest.fixture
def sample_qrels():
    """Create sample qrels (query relevance)."""
    return {
        "q_1": {"doc_1": 2, "doc_2": 1},
        "q_2": {"doc_2": 2, "doc_3": 1},
    }


@pytest.fixture
def sample_beir_dataset(tmp_cache_dir):
    """Create a temporary BEIR-format dataset."""
    # Create directory structure
    corpus_dir = tmp_cache_dir / "corpus"
    queries_dir = tmp_cache_dir / "queries"
    qrels_dir = tmp_cache_dir / "qrels"
    
    corpus_dir.mkdir()
    queries_dir.mkdir()
    qrels_dir.mkdir()
    
    # Create corpus.jsonl
    corpus_file = corpus_dir / "corpus.jsonl"
    with open(corpus_file, "w") as f:
        for doc in [
            {"_id": "doc_1", "title": "Title 1", "text": "Document content 1"},
            {"_id": "doc_2", "title": "Title 2", "text": "Document content 2"},
            {"_id": "doc_3", "title": "Title 3", "text": "Document content 3"},
        ]:
            f.write(json.dumps(doc) + "\n")
    
    # Create queries.jsonl
    queries_file = queries_dir / "queries.jsonl"
    with open(queries_file, "w") as f:
        for qid, text in [("q_1", "query one"), ("q_2", "query two")]:
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")
    
    # Create qrels/test.txt
    qrels_file = qrels_dir / "test.txt"
    with open(qrels_file, "w") as f:
        f.write("q_1 Q0 doc_1 2\n")
        f.write("q_1 Q0 doc_2 1\n")
        f.write("q_2 Q0 doc_2 2\n")
        f.write("q_2 Q0 doc_3 1\n")
    
    return tmp_cache_dir
