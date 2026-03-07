"""Tests for compression_eval_iterative.constants module."""

from pathlib import Path

import pytest

from src.compression_eval_iterative.constants import QUERY_LEN, CachePaths


class TestQueryLen:
    """Test query length mappings."""

    def test_beir_datasets(self):
        """Test BEIR dataset query lengths."""
        assert QUERY_LEN["beir/nfcorpus/test"] == 32
        assert QUERY_LEN["beir/scifact/test"] == 48
        assert QUERY_LEN["beir/scidocs"] == 48

    def test_short_names(self):
        """Test short name mappings."""
        assert QUERY_LEN["quora"] == 32
        assert QUERY_LEN["scifact"] == 48
        assert QUERY_LEN["nq"] == 32

    def test_all_lengths_positive(self):
        """Test all query lengths are positive."""
        for dataset, length in QUERY_LEN.items():
            assert length > 0, f"Invalid length for {dataset}: {length}"


class TestCachePaths:
    """Test CachePaths dataclass."""

    def test_cache_paths_creation(self, tmp_cache_dir):
        """Test creating CachePaths instance."""
        cache_paths = CachePaths(
            cache_dir=tmp_cache_dir,
            doc_meta=tmp_cache_dir / "doc_meta.json",
            query_meta=tmp_cache_dir / "query_meta.json",
            doc_emb_shard_pattern="shard_{:03d}_emb.pt",
            doc_art_shard_pattern="shard_{:03d}_art.pt",
            doc_idf_stats_file=tmp_cache_dir / "idf_stats.pt",
            query_file=tmp_cache_dir / "query.pt",
        )
        
        assert cache_paths.cache_dir == tmp_cache_dir
        assert cache_paths.doc_meta == tmp_cache_dir / "doc_meta.json"
        assert cache_paths.query_file == tmp_cache_dir / "query.pt"

    def test_cache_paths_frozen(self, tmp_cache_dir):
        """Test CachePaths is immutable."""
        cache_paths = CachePaths(
            cache_dir=tmp_cache_dir,
            doc_meta=tmp_cache_dir / "doc_meta.json",
            query_meta=tmp_cache_dir / "query_meta.json",
            doc_emb_shard_pattern="shard_{:03d}_emb.pt",
            doc_art_shard_pattern="shard_{:03d}_art.pt",
            doc_idf_stats_file=tmp_cache_dir / "idf_stats.pt",
            query_file=tmp_cache_dir / "query.pt",
        )
        
        with pytest.raises(AttributeError):
            cache_paths.cache_dir = Path("/new/path")

    def test_shard_pattern_formatting(self, tmp_cache_dir):
        """Test shard pattern formatting."""
        cache_paths = CachePaths(
            cache_dir=tmp_cache_dir,
            doc_meta=tmp_cache_dir / "doc_meta.json",
            query_meta=tmp_cache_dir / "query_meta.json",
            doc_emb_shard_pattern="shard_{:03d}_emb.pt",
            doc_art_shard_pattern="shard_{:03d}_art.pt",
            doc_idf_stats_file=tmp_cache_dir / "idf_stats.pt",
            query_file=tmp_cache_dir / "query.pt",
        )
        
        assert cache_paths.doc_emb_shard_pattern.format(0) == "shard_000_emb.pt"
        assert cache_paths.doc_emb_shard_pattern.format(5) == "shard_005_emb.pt"
        assert cache_paths.doc_emb_shard_pattern.format(123) == "shard_123_emb.pt"
