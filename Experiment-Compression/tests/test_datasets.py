"""Tests for compression_eval_iterative.data_loading module."""

from pathlib import Path

import pytest

from src.compression_eval_iterative.data_loading import (
    _load_dataset_local,
    load_dataset,
)


class TestLoadDatasetLocal:
    """Test local BEIR dataset loading."""

    def test_load_local_beir_dataset(self, sample_beir_dataset):
        """Test loading a local BEIR-format dataset."""
        documents, queries, qrels = _load_dataset_local(str(sample_beir_dataset), lowercase=False)
        
        assert len(documents) == 3
        assert len(queries) == 2
        assert len(qrels) == 2
        
        # Check document structure
        assert documents[0]["id"] == "doc_1"
        assert "Title 1" in documents[0]["text"] or "Document content 1" in documents[0]["text"]
        
        # Check queries
        assert "q_1" in queries
        assert "q_2" in queries
        
        # Check qrels
        assert "q_1" in qrels
        assert "doc_1" in qrels["q_1"]

    def test_load_with_lowercase(self, sample_beir_dataset):
        """Test loading with lowercase conversion."""
        documents, queries, qrels = _load_dataset_local(str(sample_beir_dataset), lowercase=True)
        
        # Text should be lowercased
        for doc in documents:
            assert doc["text"] == doc["text"].lower()
        
        for q in queries.values():
            assert q == q.lower()


class TestLoadDataset:
    """Test dataset loader dispatch."""

    def test_load_local_path(self, sample_beir_dataset):
        """Test loading from local path."""
        documents, queries, qrels = load_dataset(str(sample_beir_dataset))
        
        assert len(documents) > 0
        assert len(queries) > 0
        assert len(qrels) > 0

    def test_load_nonexistent_path_raises(self):
        """Test loading nonexistent local path raises error."""
        with pytest.raises(Exception):  # pylate evaluation might raise different error
            load_dataset("/nonexistent/path/123456")


class TestDatasetStructure:
    """Test dataset structure and content validation."""

    def test_documents_have_required_fields(self, sample_beir_dataset):
        """Test documents have id and text fields."""
        documents, _, _ = load_dataset(str(sample_beir_dataset))
        
        for doc in documents:
            assert "id" in doc, "Document missing 'id' field"
            assert "text" in doc, "Document missing 'text' field"
            assert isinstance(doc["id"], str)
            assert isinstance(doc["text"], str)

    def test_queries_dict_structure(self, sample_beir_dataset):
        """Test queries dict has string keys and values."""
        _, queries, _ = load_dataset(str(sample_beir_dataset))
        
        assert isinstance(queries, dict)
        for qid, text in queries.items():
            assert isinstance(qid, str)
            assert isinstance(text, str)

    def test_qrels_nested_dict_structure(self, sample_beir_dataset):
        """Test qrels has nested dict structure with int relevance."""
        _, _, qrels = load_dataset(str(sample_beir_dataset))
        
        assert isinstance(qrels, dict)
        for qid, doc_rels in qrels.items():
            assert isinstance(qid, str)
            assert isinstance(doc_rels, dict)
            for doc_id, relevance in doc_rels.items():
                assert isinstance(doc_id, str)
                assert isinstance(relevance, int)
