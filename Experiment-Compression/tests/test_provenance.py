"""Tests for compression_eval_iterative.provenance module."""

from unittest.mock import MagicMock

import pytest
import torch

from src.compression_eval_iterative.provenance import (
    build_provenance,
    get_git_info,
)


class TestGetGitInfo:
    """Test git information retrieval."""

    def test_git_info_structure(self):
        """Test that git info has correct structure."""
        info = get_git_info()
        
        assert isinstance(info, dict)
        assert "commit" in info
        assert "dirty" in info

    def test_git_info_with_repo(self):
        """Test git info when in a git repo."""
        info = get_git_info()
        
        # Should be able to get commit hash or None
        if info["commit"] is not None:
            assert isinstance(info["commit"], str)
            assert len(info["commit"]) == 40  # SHA-1 hash length


class TestBuildProvenance:
    """Test provenance metadata building."""

    def test_build_provenance_structure(self):
        """Test provenance has all required fields."""
        cfg = MagicMock()
        cfg = MagicMock()
        OmegaConf = MagicMock()
        OmegaConf.to_container = MagicMock(return_value={})
        
        # Patch OmegaConf before importing build_provenance
        import sys
        from unittest.mock import patch
        
        with patch("sys.modules"):
            provenance = build_provenance(
                cfg=cfg,
                dataset_id="test/dataset",
                model_name="test-model",
                compression_config=None,
                run_id="test_run_123",
                stats={"num_documents": 100, "num_queries": 10},
            )
        
        assert "run_id" in provenance
        assert "timestamp" in provenance
        assert "dataset" in provenance
        assert "model" in provenance
        assert "compression_config" in provenance
        assert "environment" in provenance
        assert "git" in provenance
        assert "stats" in provenance

    def test_provenance_values(self):
        """Test provenance contains correct values."""
        cfg = MagicMock()
        cfg_container = {
            "cache": {"dir": "/cache"},
            "encode": {"batch_size": 512},
        }
        
        with MagicMock() as mock_omegaconf:
            provenance = build_provenance(
                cfg=cfg,
                dataset_id="beir/nfcorpus/test",
                model_name="lightonai/GTE-ColBERT",
                compression_config=None,
                run_id="test_run_123",
                stats={"num_documents": 100},
            )
        
        assert provenance["run_id"] == "test_run_123"
        assert provenance["dataset"] == "beir/nfcorpus/test"
        assert provenance["model"] == "lightonai/GTE-ColBERT"
        assert provenance["stats"]["num_documents"] == 100

    def test_provenance_environment(self):
        """Test provenance contains environment info."""
        cfg = MagicMock()
        
        provenance = build_provenance(
            cfg=cfg,
            dataset_id="test",
            model_name="test",
            compression_config=None,
            run_id="run1",
            stats={},
        )
        
        env = provenance["environment"]
        assert "python" in env
        assert "platform" in env
        assert "torch" in env
        assert "cuda_available" in env
        assert "hostname" in env
        assert isinstance(env["cuda_available"], bool)
