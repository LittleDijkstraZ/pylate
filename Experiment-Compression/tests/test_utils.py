"""Tests for compression_eval_iterative.utils module."""

import pytest
import torch

from src.compression_eval_iterative.utils import (
    cast_embeddings,
    get_embedding_size,
    get_torch_dtype,
    move_embeddings_to_cpu,
    pack_embeddings,
    sanitize_dataset_name,
    sanitize_model_name,
    short_hash,
    unpack_embeddings,
    resolve_query_length,
)


class TestGetTorchDtype:
    """Test dtype conversion."""

    def test_fp32_conversion(self):
        """Test float32 conversion."""
        assert get_torch_dtype("fp32") == torch.float32

    def test_fp16_conversion(self):
        """Test float16 conversion."""
        assert get_torch_dtype("fp16") == torch.float16

    def test_bf16_conversion(self):
        """Test bfloat16 conversion."""
        assert get_torch_dtype("bf16") == torch.bfloat16

    def test_invalid_dtype(self):
        """Test invalid dtype raises error."""
        with pytest.raises(ValueError):
            get_torch_dtype("invalid")


class TestSanitization:
    """Test path sanitization."""

    def test_sanitize_dataset_name(self):
        """Test dataset name sanitization."""
        assert sanitize_dataset_name("beir/nfcorpus/test") == "beir_nfcorpus_test"
        assert sanitize_dataset_name("path with spaces") == "path_with_spaces"
        assert sanitize_dataset_name("path:with:colons") == "path_with_colons"

    def test_sanitize_model_name(self):
        """Test model name sanitization."""
        assert sanitize_model_name("lightonai/GTE-ModernColBERT-v1") == "lightonai_GTE-ModernColBERT-v1"
        assert (
            sanitize_model_name("output/checkpoint-model")
            == "checkpoint-model"
        )
        assert (
            sanitize_model_name("output/model_checkpoint-100")
            == "output_model"
        )


class TestShortHash:
    """Test hashing."""

    def test_hash_generation(self):
        """Test hash generation."""
        payload = {"key": "value", "number": 42}
        hash1 = short_hash(payload)
        hash2 = short_hash(payload)
        
        # Same input should produce same hash
        assert hash1 == hash2
        # Hash should be 10 chars by default
        assert len(hash1) == 10

    def test_hash_length(self):
        """Test custom hash length."""
        payload = {"test": True}
        hash_short = short_hash(payload, length=5)
        hash_long = short_hash(payload, length=20)
        
        assert len(hash_short) == 5
        assert len(hash_long) == 20


class TestResolveQueryLength:
    """Test query length resolution."""

    def test_override_takes_precedence(self):
        """Test that override parameter takes precedence."""
        assert resolve_query_length("beir/nfcorpus/test", override=64) == 64

    def test_known_dataset(self):
        """Test known dataset query length."""
        assert resolve_query_length("beir/nfcorpus/test", override=None) == 32
        assert resolve_query_length("beir/scifact/test", override=None) == 48

    def test_unknown_dataset_default(self):
        """Test unknown dataset returns default."""
        assert resolve_query_length("unknown/dataset", override=None) == 32


class TestEmbeddingPacking:
    """Test embedding packing and unpacking."""

    def test_pack_embeddings(self, sample_embeddings):
        """Test packing variable-length embeddings."""
        packed = pack_embeddings(sample_embeddings)
        
        assert "embeddings" in packed
        assert "lengths" in packed
        assert packed["lengths"].shape[0] == 3
        assert (packed["lengths"] == torch.tensor([10, 8, 12])).all()

    def test_pack_empty(self):
        """Test packing empty list."""
        packed = pack_embeddings([])
        
        assert packed["embeddings"].numel() == 0
        assert packed["lengths"].numel() == 0

    def test_unpack_embeddings(self, sample_embeddings):
        """Test unpacking embeddings."""
        packed = pack_embeddings(sample_embeddings)
        unpacked = unpack_embeddings(packed)
        
        assert len(unpacked) == 3
        assert unpacked[0].shape == (10, 128)
        assert unpacked[1].shape == (8, 128)
        assert unpacked[2].shape == (12, 128)

    def test_pack_unpack_roundtrip(self, sample_embeddings):
        """Test pack-unpack roundtrip."""
        packed = pack_embeddings(sample_embeddings)
        unpacked = unpack_embeddings(packed)
        
        for orig, recovered in zip(sample_embeddings, unpacked):
            assert torch.allclose(orig, recovered)


class TestEmbeddingCasting:
    """Test embedding type casting."""

    def test_cast_embeddings_fp32_to_fp16(self):
        """Test casting to fp16."""
        embs = [torch.randn(5, 128, dtype=torch.float32)]
        casted = cast_embeddings(embs, torch.float16)
        
        assert casted[0].dtype == torch.float16
        assert casted[0].shape == (5, 128)

    def test_cast_same_dtype(self):
        """Test casting to same dtype returns same list."""
        embs = [torch.randn(5, 128, dtype=torch.float32)]
        casted = cast_embeddings(embs, torch.float32)
        
        assert casted is embs  # Should return same list

    def test_cast_empty(self):
        """Test casting empty list."""
        casted = cast_embeddings([], torch.float16)
        assert casted == []


class TestMoveToCPU:
    """Test moving embeddings to CPU."""

    def test_move_to_cpu(self):
        """Test moving embeddings to CPU."""
        embs = [torch.randn(5, 128)]
        moved = move_embeddings_to_cpu(embs)
        
        assert all(emb.device.type == "cpu" for emb in moved)

    def test_empty_list(self):
        """Test moving empty list."""
        moved = move_embeddings_to_cpu([])
        assert moved == []
