"""Tests for compression_eval_iterative.configs module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.compression_eval_iterative.configs import (
    create_default_configs,
    serialize_config_for_storage,
)


class TestSerializeConfig:
    """Test config serialization."""

    def test_serialize_baseline(self):
        """Test serializing baseline (None) config."""
        serialized = serialize_config_for_storage(None)
        
        assert serialized["type"] == "baseline"
        assert serialized["description"] == "No compression"

    def test_serialize_with_config(self):
        """Test serializing actual compression config."""
        # Mock a compression config
        mock_config = MagicMock()
        mock_config.serialize.return_value = {
            "type": "compression",
            "strategies": ["random_pruning"],
        }
        
        serialized = serialize_config_for_storage(mock_config)
        
        assert "type" in serialized
        assert mock_config.serialize.called


class TestCreateDefaultConfigs:
    """Test default config creation."""

    def test_creates_baseline(self):
        """Test that baseline is first config."""
        mock_model = MagicMock()
        mock_model.tokenizer.added_tokens_decoder.keys.return_value = [101, 102]
        
        configs = create_default_configs(mock_model)
        
        assert len(configs) > 0
        assert configs[0] is None, "First config should be baseline (None)"

    def test_creates_multiple_configs(self):
        """Test that multiple configs are created."""
        mock_model = MagicMock()
        mock_model.tokenizer.added_tokens_decoder.keys.return_value = [101, 102]
        
        configs = create_default_configs(mock_model)
        
        # Should have baseline + multiple strategy configs
        assert len(configs) > 30, f"Expected many configs, got {len(configs)}"

    def test_config_descriptions(self):
        """Test that configs have descriptions."""
        mock_model = MagicMock()
        mock_model.tokenizer.added_tokens_decoder.keys.return_value = [101, 102]
        
        configs = create_default_configs(mock_model)
        
        # Check non-baseline configs have descriptions
        for config in configs[1:]:  # Skip baseline
            assert config is not None
            assert hasattr(config, "description")
            assert len(config.description) > 0

    def test_config_strategies(self):
        """Test that configs have strategies."""
        mock_model = MagicMock()
        mock_model.tokenizer.added_tokens_decoder.keys.return_value = [101, 102]
        
        configs = create_default_configs(mock_model)
        
        # Check non-baseline configs have strategies
        for config in configs[1:]:  # Skip baseline
            assert config is not None
            assert hasattr(config, "strategies")
            assert len(config.strategies) > 0

    def test_random_configs_present(self):
        """Test that random pruning/pooling configs are created."""
        mock_model = MagicMock()
        mock_model.tokenizer.added_tokens_decoder.keys.return_value = [101, 102]
        
        configs = create_default_configs(mock_model)
        descriptions = [c.description if c else "Baseline" for c in configs]
        
        # Should have random pruning configs
        random_pruning = [d for d in descriptions if "Random pruning" in d]
        assert len(random_pruning) > 0, "No random pruning configs found"
        
        # Should have random pooling configs
        random_pooling = [d for d in descriptions if "Random pooling" in d]
        assert len(random_pooling) > 0, "No random pooling configs found"

    def test_attention_configs_present(self):
        """Test that attention-based configs are created."""
        mock_model = MagicMock()
        mock_model.tokenizer.added_tokens_decoder.keys.return_value = [101, 102]
        
        configs = create_default_configs(mock_model)
        descriptions = [c.description if c else "Baseline" for c in configs]
        
        # Should have attention pruning
        attention_prune = [d for d in descriptions if "Attention score pruning" in d]
        assert len(attention_prune) > 0
        
        # Should have attention pooling
        attention_pool = [d for d in descriptions if "Attention score pooling" in d]
        assert len(attention_pool) > 0

    def test_pooling_configs_present(self):
        """Test that clustering-based pooling configs are created."""
        mock_model = MagicMock()
        mock_model.tokenizer.added_tokens_decoder.keys.return_value = [101, 102]
        
        configs = create_default_configs(mock_model)
        descriptions = [c.description if c else "Baseline" for c in configs]
        
        # Should have clustering pooling
        clustering = [d for d in descriptions if "Pooling" in d and any(m in d for m in ["Spherical", "Hierarchical"])]
        assert len(clustering) > 0, "No clustering pooling configs found"
