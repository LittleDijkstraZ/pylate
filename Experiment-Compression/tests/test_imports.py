"""Tests for package structure and imports."""

import pytest


class TestPackageImports:
    """Test that all modules can be imported."""

    def test_import_constants(self):
        """Test importing constants module."""
        from src.compression_eval_iterative import constants
        
        assert hasattr(constants, "QUERY_LEN")
        assert hasattr(constants, "CachePaths")

    def test_import_utils(self):
        """Test importing utils module."""
        from src.compression_eval_iterative import utils
        
        assert hasattr(utils, "get_torch_dtype")
        assert hasattr(utils, "sanitize_dataset_name")
        assert hasattr(utils, "pack_embeddings")

    def test_import_datasets(self):
        """Test importing data_loading module."""
        from src.compression_eval_iterative import data_loading

        assert hasattr(data_loading, "load_dataset")
        assert hasattr(data_loading, "_load_dataset_local")
    def test_import_cache(self):
        """Test importing cache module."""
        from src.compression_eval_iterative import cache
        
        assert hasattr(cache, "build_cache_paths")
        assert hasattr(cache, "encode_documents_with_cache")
        assert hasattr(cache, "iter_document_shards")

    def test_import_configs(self):
        """Test importing configs module."""
        from src.compression_eval_iterative import configs
        
        assert hasattr(configs, "create_default_configs")
        assert hasattr(configs, "serialize_config_for_storage")

    def test_import_provenance(self):
        """Test importing provenance module."""
        from src.compression_eval_iterative import provenance
        
        assert hasattr(provenance, "build_provenance")
        assert hasattr(provenance, "get_git_info")

    def test_import_evaluate(self):
        """Test importing evaluate module."""
        from src.compression_eval_iterative import evaluate
        
        assert hasattr(evaluate, "evaluate_compression_config")
        assert hasattr(evaluate, "apply_compression")

    def test_import_cli(self):
        """Test importing cli module."""
        from src.compression_eval_iterative import cli
        
        assert hasattr(cli, "main")
        assert hasattr(cli, "build_model")


class TestPackageExports:
    """Test that package exports main."""

    def test_main_export(self):
        """Test that main is exported from package."""
        from src.compression_eval_iterative import main
        
        assert callable(main)


class TestModuleAvailability:
    """Test all expected functions are available."""

    def test_all_utils_available(self):
        """Test all expected utils are available."""
        from src.compression_eval_iterative.utils import (
            get_torch_dtype,
            sanitize_dataset_name,
            sanitize_model_name,
            short_hash,
            resolve_query_length,
            get_embedding_size,
            pack_embeddings,
            unpack_embeddings,
            cast_embeddings,
            move_embeddings_to_cpu,
        )
        
        assert all([
            callable(get_torch_dtype),
            callable(sanitize_dataset_name),
            callable(sanitize_model_name),
            callable(short_hash),
            callable(resolve_query_length),
            callable(get_embedding_size),
            callable(pack_embeddings),
            callable(unpack_embeddings),
            callable(cast_embeddings),
            callable(move_embeddings_to_cpu),
        ])

    def test_all_cache_available(self):
        """Test all expected cache functions are available."""
        from src.compression_eval_iterative.cache import (
            build_cache_paths,
            encode_documents_with_cache,
            iter_document_shards,
            compute_and_cache_idf_stats,
            encode_queries_with_cache,
        )
        
        assert all([
            callable(build_cache_paths),
            callable(encode_documents_with_cache),
            callable(iter_document_shards),
            callable(compute_and_cache_idf_stats),
            callable(encode_queries_with_cache),
        ])
