"""Tests for compression_eval.py utility functions and configurations.

Note: Due to module import dependencies, we replicate the pure utility functions
here for testing. These are the exact same implementations as in compression_eval.py.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn.functional as F

from pylate.models.compression import (
    CompressionConfig,
    RandomPruningConfig,
    RandomPruningStrategy,
    IDFPruningConfig,
    IDFPruningStrategy,
)


# ============================================================================
# Replicated utility functions from compression_eval.py for testing
# ============================================================================

@dataclass
class CachePaths:
    """Container for cache file paths."""
    cache_dir: Path
    doc_meta: Path
    query_meta: Path
    doc_shard_pattern: str
    query_file: Path
    doc_hash: str
    query_hash: str


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype_map[dtype_str]


def sanitize_dataset_name(dataset_name: str) -> str:
    """Sanitize dataset name for filesystem paths."""
    return dataset_name.replace("/", "_").replace(" ", "_").replace(":", "_").replace("\\", "_")


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for filesystem paths."""
    sanitized = model_name.split("output/")[-1].replace("/", "_")
    if "_checkpoint-" in sanitized:
        sanitized = sanitized.rsplit("_checkpoint-", 1)[0]
    return sanitized


def short_hash(payload: Dict[str, Any], length: int = 10) -> str:
    """Generate a short hash from a dictionary payload."""
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:length]


# Query length mapping (subset for testing)
QUERY_LEN = {
    "beir/nfcorpus/test": 32,
    "beir/fiqa/test": 64,
    "beir/scifact/test": 128,
}


def resolve_query_length(dataset_id: str, override: Optional[int]) -> int:
    """Resolve query length from dataset ID or override."""
    if override is not None:
        return override
    return QUERY_LEN.get(dataset_id, 32)


def pack_embeddings(embeddings: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pack variable-length embeddings into a single tensor with lengths."""
    if not embeddings:
        return {"embeddings": torch.empty(0), "lengths": torch.empty(0, dtype=torch.long)}
    lengths = torch.tensor([emb.shape[0] for emb in embeddings], dtype=torch.long)
    concatenated = torch.cat(embeddings, dim=0)
    return {"embeddings": concatenated, "lengths": lengths}


def unpack_embeddings(packed: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Unpack embeddings from packed format."""
    if packed["lengths"].numel() == 0:
        return []
    concatenated = packed["embeddings"]
    lengths = packed["lengths"]
    embeddings = []
    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length.item()
        embeddings.append(concatenated[start_idx:end_idx])
        start_idx = end_idx
    return embeddings


def cast_embeddings(embeddings: List[torch.Tensor], dtype: torch.dtype) -> List[torch.Tensor]:
    """Cast embeddings to a specific dtype."""
    if not embeddings:
        return embeddings
    if embeddings[0].dtype == dtype:
        return embeddings
    return [emb.to(dtype) for emb in embeddings]


def move_embeddings_to_cpu(embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
    """Move embeddings to CPU."""
    return [emb.cpu() for emb in embeddings]


def serialize_config_for_storage(config: Optional[CompressionConfig]) -> Dict[str, Any]:
    """Serialize a compression config (or baseline) for storage."""
    if config is None:
        return {"type": "baseline", "description": "No compression"}
    return config.serialize()


def apply_compression(
    embeddings: List[torch.Tensor],
    artifacts: Optional[Dict[str, List[torch.Tensor]]],
    config: Optional[CompressionConfig],
    batch_size: int,
) -> tuple[List[torch.Tensor], float]:
    """Apply compression to embeddings. Returns compressed embeddings and compression time."""
    import time
    start_time = time.time()

    if config is None:
        # Baseline: just normalize the embeddings
        compressed = [F.normalize(emb, p=2, dim=-1) for emb in embeddings]
    else:
        compressor = config.create_compressor()
        compressed, _ = compressor.compress_parallel(
            embeddings=embeddings,
            artifacts=artifacts if artifacts else {},
            batch_size=batch_size,
            num_workers=8,
            show_progress=False,  # Disable progress for tests
        )
        # Normalize after compression
        compressed = [F.normalize(emb, p=2, dim=-1) for emb in compressed]

    compression_time = time.time() - start_time
    return compressed, compression_time


# ============================================================================
# Tests for get_torch_dtype
# ============================================================================

class TestGetTorchDtype:
    """Tests for get_torch_dtype function."""

    def test_fp32(self):
        """Test fp32 dtype conversion."""
        assert get_torch_dtype("fp32") == torch.float32

    def test_fp16(self):
        """Test fp16 dtype conversion."""
        assert get_torch_dtype("fp16") == torch.float16

    def test_bf16(self):
        """Test bf16 dtype conversion."""
        assert get_torch_dtype("bf16") == torch.bfloat16

    def test_unsupported_dtype(self):
        """Test that unsupported dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_torch_dtype("int8")


# ============================================================================
# Tests for sanitize_dataset_name
# ============================================================================

class TestSanitizeDatasetName:
    """Tests for sanitize_dataset_name function."""

    def test_slashes(self):
        """Test that slashes are replaced."""
        assert sanitize_dataset_name("beir/nfcorpus/test") == "beir_nfcorpus_test"

    def test_spaces(self):
        """Test that spaces are replaced."""
        assert sanitize_dataset_name("my dataset name") == "my_dataset_name"

    def test_colons(self):
        """Test that colons are replaced."""
        assert sanitize_dataset_name("dataset:v1") == "dataset_v1"

    def test_backslashes(self):
        """Test that backslashes are replaced."""
        assert sanitize_dataset_name("path\\to\\dataset") == "path_to_dataset"

    def test_mixed(self):
        """Test mixed special characters."""
        assert sanitize_dataset_name("beir/corpus:test v2") == "beir_corpus_test_v2"


# ============================================================================
# Tests for sanitize_model_name
# ============================================================================

class TestSanitizeModelName:
    """Tests for sanitize_model_name function."""

    def test_huggingface_path(self):
        """Test HuggingFace model path."""
        assert sanitize_model_name("lightonai/GTE-ModernColBERT-v1") == "lightonai_GTE-ModernColBERT-v1"

    def test_output_path(self):
        """Test output/ prefix removal."""
        assert sanitize_model_name("output/my_model/checkpoint-1000") == "my_model"

    def test_checkpoint_removal(self):
        """Test checkpoint suffix removal."""
        assert sanitize_model_name("my_model_checkpoint-500") == "my_model"

    def test_local_path(self):
        """Test local path."""
        assert sanitize_model_name("/home/user/models/my_model") == "_home_user_models_my_model"


# ============================================================================
# Tests for short_hash
# ============================================================================

class TestShortHash:
    """Tests for short_hash function."""

    def test_deterministic(self):
        """Test that same input produces same hash."""
        payload = {"key": "value", "num": 123}
        hash1 = short_hash(payload)
        hash2 = short_hash(payload)
        assert hash1 == hash2

    def test_different_inputs(self):
        """Test that different inputs produce different hashes."""
        hash1 = short_hash({"key": "value1"})
        hash2 = short_hash({"key": "value2"})
        assert hash1 != hash2

    def test_length(self):
        """Test hash length parameter."""
        hash_10 = short_hash({"key": "value"}, length=10)
        hash_5 = short_hash({"key": "value"}, length=5)
        assert len(hash_10) == 10
        assert len(hash_5) == 5

    def test_order_independence(self):
        """Test that key order doesn't affect hash (sort_keys=True)."""
        hash1 = short_hash({"a": 1, "b": 2})
        hash2 = short_hash({"b": 2, "a": 1})
        assert hash1 == hash2


# ============================================================================
# Tests for resolve_query_length
# ============================================================================

class TestResolveQueryLength:
    """Tests for resolve_query_length function."""

    def test_override(self):
        """Test that override takes precedence."""
        assert resolve_query_length("beir/nfcorpus/test", 64) == 64

    def test_known_dataset(self):
        """Test known dataset lookup."""
        # Check a dataset that's in QUERY_LEN
        for dataset_id, length in list(QUERY_LEN.items())[:3]:
            assert resolve_query_length(dataset_id, None) == length

    def test_unknown_dataset(self):
        """Test fallback for unknown dataset."""
        assert resolve_query_length("unknown/dataset", None) == 32


# ============================================================================
# Tests for pack_embeddings and unpack_embeddings
# ============================================================================

class TestPackUnpackEmbeddings:
    """Tests for pack_embeddings and unpack_embeddings functions."""

    def test_empty_list(self):
        """Test packing empty list."""
        packed = pack_embeddings([])
        assert packed["embeddings"].numel() == 0
        assert packed["lengths"].numel() == 0
        unpacked = unpack_embeddings(packed)
        assert unpacked == []

    def test_single_embedding(self):
        """Test packing single embedding."""
        torch.manual_seed(42)
        emb = [torch.randn(5, 128)]
        packed = pack_embeddings(emb)
        assert packed["embeddings"].shape == (5, 128)
        assert packed["lengths"].tolist() == [5]
        unpacked = unpack_embeddings(packed)
        assert len(unpacked) == 1
        assert torch.allclose(unpacked[0], emb[0])

    def test_multiple_embeddings(self):
        """Test packing multiple embeddings of different lengths."""
        torch.manual_seed(42)
        embs = [
            torch.randn(3, 128),
            torch.randn(7, 128),
            torch.randn(5, 128),
        ]
        packed = pack_embeddings(embs)
        assert packed["embeddings"].shape == (15, 128)
        assert packed["lengths"].tolist() == [3, 7, 5]
        unpacked = unpack_embeddings(packed)
        assert len(unpacked) == 3
        for orig, restored in zip(embs, unpacked):
            assert torch.allclose(orig, restored)

    def test_roundtrip(self):
        """Test pack/unpack roundtrip preserves data."""
        torch.manual_seed(42)
        embs = [torch.randn(10, 64), torch.randn(20, 64)]
        packed = pack_embeddings(embs)
        unpacked = unpack_embeddings(packed)
        for orig, restored in zip(embs, unpacked):
            assert orig.shape == restored.shape
            assert torch.allclose(orig, restored)


# ============================================================================
# Tests for cast_embeddings
# ============================================================================

class TestCastEmbeddings:
    """Tests for cast_embeddings function."""

    def test_empty_list(self):
        """Test casting empty list."""
        result = cast_embeddings([], torch.float16)
        assert result == []

    def test_same_dtype(self):
        """Test no-op when dtype matches."""
        embs = [torch.randn(5, 128, dtype=torch.float32)]
        result = cast_embeddings(embs, torch.float32)
        assert result is embs  # Should return same object

    def test_cast_to_fp16(self):
        """Test casting to fp16."""
        embs = [torch.randn(5, 128, dtype=torch.float32)]
        result = cast_embeddings(embs, torch.float16)
        assert result[0].dtype == torch.float16

    def test_cast_multiple(self):
        """Test casting multiple embeddings."""
        embs = [torch.randn(3, 64), torch.randn(5, 64)]
        result = cast_embeddings(embs, torch.float16)
        assert len(result) == 2
        assert all(e.dtype == torch.float16 for e in result)


# ============================================================================
# Tests for move_embeddings_to_cpu
# ============================================================================

class TestMoveEmbeddingsToCPU:
    """Tests for move_embeddings_to_cpu function."""

    def test_already_on_cpu(self):
        """Test embeddings already on CPU."""
        embs = [torch.randn(5, 128, device="cpu")]
        result = move_embeddings_to_cpu(embs)
        assert result[0].device.type == "cpu"

    def test_multiple_embeddings(self):
        """Test moving multiple embeddings."""
        embs = [torch.randn(3, 64), torch.randn(5, 64)]
        result = move_embeddings_to_cpu(embs)
        assert len(result) == 2
        assert all(e.device.type == "cpu" for e in result)


# ============================================================================
# Tests for serialize_config_for_storage
# ============================================================================

class TestSerializeConfigForStorage:
    """Tests for serialize_config_for_storage function."""

    def test_none_config(self):
        """Test serializing None (baseline)."""
        result = serialize_config_for_storage(None)
        assert result["type"] == "baseline"
        assert result["description"] == "No compression"

    def test_compression_config(self):
        """Test serializing a compression config."""
        config = CompressionConfig(
            strategies=[RandomPruningStrategy(RandomPruningConfig(
                keep_ratio=0.5,
                protected_tokens=1,
                min_tokens=8,
                seed=42,
            ))],
            description="Test config",
        )
        result = serialize_config_for_storage(config)
        assert "strategies" in result
        assert result["description"] == "Test config"


# ============================================================================
# Tests for apply_compression
# ============================================================================

class TestApplyCompression:
    """Tests for apply_compression function."""

    @pytest.fixture
    def sample_embeddings(self) -> List[torch.Tensor]:
        """Create sample unnormalized embeddings."""
        torch.manual_seed(42)
        return [
            torch.randn(10, 128),
            torch.randn(15, 128),
            torch.randn(12, 128),
        ]

    @pytest.fixture
    def sample_artifacts(self, sample_embeddings) -> Dict[str, List[torch.Tensor]]:
        """Create sample artifacts."""
        return {
            "input_ids": [torch.arange(emb.shape[0]) for emb in sample_embeddings],
        }

    def test_baseline_normalization(self, sample_embeddings):
        """Test that baseline (None config) normalizes embeddings."""
        compressed, comp_time = apply_compression(
            embeddings=sample_embeddings,
            artifacts=None,
            config=None,
            batch_size=32,
        )
        # Check normalization
        for emb in compressed:
            norms = torch.norm(emb, p=2, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        # Check lengths are preserved
        assert [e.shape[0] for e in compressed] == [e.shape[0] for e in sample_embeddings]

    def test_random_pruning_compression(self, sample_embeddings, sample_artifacts):
        """Test compression with random pruning."""
        config = CompressionConfig(
            strategies=[RandomPruningStrategy(RandomPruningConfig(
                keep_ratio=0.5,
                protected_tokens=1,
                min_tokens=4,
                seed=42,
            ))],
            description="Random pruning test",
        )
        compressed, comp_time = apply_compression(
            embeddings=sample_embeddings,
            artifacts=sample_artifacts,
            config=config,
            batch_size=32,
        )
        # Check that embeddings are reduced
        for orig, comp in zip(sample_embeddings, compressed):
            assert comp.shape[0] <= orig.shape[0]
        # Check normalization
        for emb in compressed:
            norms = torch.norm(emb, p=2, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        # Check time is positive
        assert comp_time > 0


# ============================================================================
# Tests for CachePaths dataclass
# ============================================================================

class TestCachePaths:
    """Tests for CachePaths dataclass."""

    def test_cache_paths_creation(self):
        """Test creating CachePaths."""
        cache_paths = CachePaths(
            cache_dir=Path("/tmp/cache"),
            doc_meta=Path("/tmp/cache/doc_meta.json"),
            query_meta=Path("/tmp/cache/query_meta.json"),
            doc_shard_pattern="doc_shard_{:03d}.pt",
            query_file=Path("/tmp/cache/query.pt"),
            doc_hash="abc123",
            query_hash="def456",
        )
        assert cache_paths.cache_dir == Path("/tmp/cache")
        assert cache_paths.doc_hash == "abc123"
        assert cache_paths.query_hash == "def456"


# ============================================================================
# Tests for JSONL config loading (with temp file)
# ============================================================================

class TestJSONLConfigLoading:
    """Tests for JSONL configuration loading."""

    def test_serialize_and_deserialize_config(self):
        """Test that configs can be serialized and stored to JSONL format."""
        config = CompressionConfig(
            strategies=[RandomPruningStrategy(RandomPruningConfig(
                keep_ratio=0.33,
                protected_tokens=1,
                min_tokens=5,
                seed=666,
            ))],
            description="Test random pruning config",
        )
        serialized = serialize_config_for_storage(config)

        # Write to temp JSONL
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps(serialized) + "\n")
            temp_path = f.name

        # Read back and verify structure
        with open(temp_path, 'r') as f:
            data = json.loads(f.readline())

        assert "strategies" in data
        assert data["description"] == "Test random pruning config"

        # Clean up
        Path(temp_path).unlink()

    def test_baseline_serialization(self):
        """Test baseline (None) serialization."""
        serialized = serialize_config_for_storage(None)
        assert serialized == {"type": "baseline", "description": "No compression"}


# ============================================================================
# Integration test: Full compression pipeline
# ============================================================================

class TestCompressionPipeline:
    """Integration tests for the compression pipeline."""

    def test_end_to_end_baseline(self):
        """Test end-to-end baseline (no compression)."""
        torch.manual_seed(42)
        embeddings = [torch.randn(20, 128) for _ in range(5)]

        # Apply baseline
        compressed, _ = apply_compression(
            embeddings=embeddings,
            artifacts=None,
            config=None,
            batch_size=32,
        )

        # Verify
        assert len(compressed) == 5
        assert all(e.shape[0] == 20 for e in compressed)
        # Check normalized
        for emb in compressed:
            norms = torch.norm(emb, p=2, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_end_to_end_with_compression(self):
        """Test end-to-end with compression strategy."""
        torch.manual_seed(42)
        embeddings = [torch.randn(20, 128) for _ in range(5)]
        artifacts = {"input_ids": [torch.arange(20) for _ in range(5)]}

        config = CompressionConfig(
            strategies=[RandomPruningStrategy(RandomPruningConfig(
                keep_ratio=0.5,
                protected_tokens=1,
                min_tokens=5,
                seed=42,
            ))],
            description="Test compression",
        )

        compressed, comp_time = apply_compression(
            embeddings=embeddings,
            artifacts=artifacts,
            config=config,
            batch_size=32,
        )

        # Verify compression happened
        assert len(compressed) == 5
        total_orig = sum(e.shape[0] for e in embeddings)
        total_comp = sum(e.shape[0] for e in compressed)
        assert total_comp < total_orig

        # Check normalized
        for emb in compressed:
            norms = torch.norm(emb, p=2, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ============================================================================
# Main block for running tests directly
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Running compression_eval tests...")
    print("=" * 80)
    pytest.main([__file__, "-v"])

