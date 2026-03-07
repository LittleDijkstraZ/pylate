"""Tests for compression_eval_iterative.cache module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.compression_eval_iterative.cache import (
    build_cache_paths,
    pack_embeddings,
    unpack_embeddings,
)
from src.compression_eval_iterative.constants import CachePaths


class TestBuildCachePaths:
    """Test cache paths builder."""

    def test_build_cache_paths(self, tmp_cache_dir):
        """Test building cache paths."""
        cfg = MagicMock()
        cfg.cache.dir = str(tmp_cache_dir)
        cfg.cache.schema_version = 1
        cfg.cache.embedding_dtype = "fp16"
        cfg.model.dtype = "bf16"
        
        cache_paths = build_cache_paths(
            cfg=cfg,
            dataset_id="beir/nfcorpus/test",
            model_name="lightonai/GTE-ColBERT",
            doc_length=300,
            query_length=32,
            lowercase=False,
        )
        
        assert isinstance(cache_paths, CachePaths)
        assert cache_paths.cache_dir.exists()
        assert "beir_nfcorpus_test" in str(cache_paths.cache_dir)
        assert "doclen300" in str(cache_paths.cache_dir)
        assert "qlen32" in str(cache_paths.cache_dir)

    def test_cache_paths_sanitization(self, tmp_cache_dir):
        """Test dataset/model name sanitization in paths."""
        cfg = MagicMock()
        cfg.cache.dir = str(tmp_cache_dir)
        cfg.cache.schema_version = 1
        cfg.cache.embedding_dtype = "fp32"
        cfg.model.dtype = "fp16"
        
        cache_paths = build_cache_paths(
            cfg=cfg,
            dataset_id="beir/nfcorpus/test",
            model_name="models/my-model",
            doc_length=300,
            query_length=32,
            lowercase=False,
        )
        
        path_str = str(cache_paths.cache_dir)
        assert "/" not in path_str.split(str(tmp_cache_dir))[1]  # No more slashes after base


class TestCachePathsMetadata:
    """Test cache metadata files."""

    def test_doc_meta_path(self, tmp_cache_dir):
        """Test doc_meta file path."""
        cfg = MagicMock()
        cfg.cache.dir = str(tmp_cache_dir)
        cfg.cache.schema_version = 1
        cfg.cache.embedding_dtype = "fp16"
        cfg.model.dtype = "bf16"
        
        cache_paths = build_cache_paths(
            cfg=cfg,
            dataset_id="test",
            model_name="model",
            doc_length=300,
            query_length=32,
            lowercase=False,
        )
        
        assert cache_paths.doc_meta.name == "doc_meta.json"
        assert cache_paths.query_meta.name == "query_meta.json"

    def test_shard_patterns(self, tmp_cache_dir):
        """Test shard filename patterns."""
        cfg = MagicMock()
        cfg.cache.dir = str(tmp_cache_dir)
        cfg.cache.schema_version = 1
        cfg.cache.embedding_dtype = "fp16"
        cfg.model.dtype = "bf16"
        
        cache_paths = build_cache_paths(
            cfg=cfg,
            dataset_id="test",
            model_name="model",
            doc_length=300,
            query_length=32,
            lowercase=False,
        )
        
        emb_0 = cache_paths.cache_dir / cache_paths.doc_emb_shard_pattern.format(0)
        emb_5 = cache_paths.cache_dir / cache_paths.doc_emb_shard_pattern.format(5)
        
        assert "shard_000_emb.pt" in str(emb_0)
        assert "shard_005_emb.pt" in str(emb_5)


class TestEmbeddingPackingIntegration:
    """Integration tests for packing/unpacking."""

    def test_pack_unpack_preserves_structure(self):
        """Test that pack/unpack preserves tensor structure."""
        embeddings = [
            torch.randn(10, 128),
            torch.randn(5, 128),
            torch.randn(15, 128),
        ]
        
        packed = pack_embeddings(embeddings)
        unpacked = unpack_embeddings(packed)
        
        assert len(unpacked) == 3
        for orig, recovered in zip(embeddings, unpacked):
            assert orig.shape == recovered.shape
            assert torch.allclose(orig, recovered, rtol=1e-5)

    def test_pack_unpack_dtype_preservation(self):
        """Test that packing preserves dtype."""
        embeddings = [
            torch.randn(10, 128, dtype=torch.float16),
            torch.randn(5, 128, dtype=torch.float16),
        ]
        
        packed = pack_embeddings(embeddings)
        unpacked = unpack_embeddings(packed)
        
        for orig, recovered in zip(embeddings, unpacked):
            assert orig.dtype == recovered.dtype
