# Test Suite for compression_eval_iterative

Comprehensive test suite for the refactored `compression_eval_iterative` package.

## Test Structure

```
tests/
├── __init__.py              # Package marker
├── conftest.py              # Shared fixtures
├── test_imports.py          # Package import validation
├── test_constants.py        # Constants and data structures
├── test_utils.py            # Utility functions
├── test_datasets.py         # Dataset loading
├── test_cache.py            # Cache and embedding operations
├── test_configs.py          # Compression configurations
└── test_provenance.py       # Provenance and metadata
```

## Test Categories

### 1. **test_imports.py** - Package Structure
- Validates all modules can be imported
- Checks that all expected functions are available
- Tests package exports

### 2. **test_constants.py** - Constants & Data Structures
- Query length mappings for datasets
- CachePaths dataclass validation
- Immutability checks

### 3. **test_utils.py** - Utilities
- PyTorch dtype conversion
- Path sanitization (dataset/model names)
- Hash generation
- Query length resolution
- **Embedding operations:**
  - Packing/unpacking variable-length embeddings
  - Casting between dtypes
  - Moving to CPU

### 4. **test_datasets.py** - Dataset Loading
- Local BEIR-format dataset loading
- Case conversion (lowercase)
- Dataset structure validation
- Document/query/qrels field validation

### 5. **test_cache.py** - Cache Management
- Cache path building with proper sanitization
- Metadata file paths
- Shard filename patterns
- Embedding packing/unpacking integration

### 6. **test_configs.py** - Compression Configs
- Config serialization (baseline & actual)
- Default config creation
- Config descriptions and strategies
- Presence of specific compression methods:
  - Random pruning/pooling
  - Attention-based pruning/pooling
  - Clustering-based pooling
  - IDF-based methods

### 7. **test_provenance.py** - Metadata
- Git information retrieval
- Provenance metadata structure
- Environment information tracking

## Running Tests

### Run All Tests
```bash
cd pylate/Experiment-Compression
pytest
```

### Run Specific Test File
```bash
pytest tests/test_utils.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_utils.py::TestGetTorchDtype -v
```

### Run Specific Test
```bash
pytest tests/test_utils.py::TestGetTorchDtype::test_fp32_conversion -v
```

### Run with Coverage
```bash
pytest --cov=src/compression_eval_iterative tests/
```

### Run Only Fast Tests (skip slow)
```bash
pytest -m "not slow"
```

## Fixtures (conftest.py)

- **`tmp_cache_dir`** - Temporary directory for cache operations
- **`sample_embeddings`** - Pre-generated sample embeddings (3 tensors of varying sizes)
- **`sample_documents`** - Sample document dicts with id/text
- **`sample_queries`** - Sample query dictionary
- **`sample_qrels`** - Sample relevance judgments
- **`sample_beir_dataset`** - Complete BEIR-format dataset in temp directory

## Test Coverage

Tests validate:
- ✅ Module imports and exports
- ✅ Constants and data structures
- ✅ Utility function correctness
- ✅ Dataset loading (local BEIR format)
- ✅ Embedding packing/unpacking
- ✅ Cache path generation
- ✅ Configuration management
- ✅ Metadata/provenance tracking

## Example Test Results

```
tests/test_imports.py::TestPackageImports::test_import_constants PASSED
tests/test_imports.py::TestPackageImports::test_import_utils PASSED
tests/test_utils.py::TestGetTorchDtype::test_fp32_conversion PASSED
tests/test_utils.py::TestEmbeddingPacking::test_pack_unpack_roundtrip PASSED
tests/test_datasets.py::TestLoadDatasetLocal::test_load_local_beir_dataset PASSED
tests/test_cache.py::TestBuildCachePaths::test_build_cache_paths PASSED
tests/test_configs.py::TestCreateDefaultConfigs::test_creates_multiple_configs PASSED

========================== 50+ tests passed in 2.3s ==========================
```

## Adding New Tests

1. Create test file in `tests/test_module_name.py`
2. Use `Test*` class names for grouping
3. Use `test_*` function names
4. Leverage fixtures from `conftest.py`
5. Run locally before committing:
   ```bash
   pytest tests/test_module_name.py -v
   ```

## Dependencies

Tests require:
- `pytest >= 6.0`
- `torch`
- `pylate` (local package)
- `omegaconf`
- `ranx` (for evaluation tests)

Install test dependencies:
```bash
pip install pytest pytest-cov
```
