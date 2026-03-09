# `compression_eval_iterative` — Package Reference

A modular Python package for benchmarking ColBERT document-embedding compression
strategies against retrieval quality.  It replaces the monolithic
`compression_eval_iterative.py` script and is the recommended entry point for
all new experiments.

---

## How to invoke the CLI

The package is designed to be run as a Python module from inside the
`Experiment-Compression/` directory:

```bash
cd pylate/Experiment-Compression

# Default config (compression_eval.yaml)
python -m src.compression_eval_iterative.cli

# Named config (e.g. the Amazon dataset experiment)
python -m src.compression_eval_iterative.cli --config-name compression_eval_amazon

# Override any Hydra key at runtime
python -m src.compression_eval_iterative.cli \
    --config-name compression_eval_amazon \
    dataset.name=/path/to/beir_format_full \
    compression.configs_file=src/conf/my_configs.jsonl
```

The convenience shell script `scripts/paper_exp/compression_eval_iterative.sh`
wraps the above invocation, sets `CUDA_VISIBLE_DEVICES`, redirects logs, and
waits for the job to finish.

---

## Module map

| File | Purpose |
|---|---|
| `cli.py` | **Entry point.** Hydra `@main`; orchestrates the full pipeline. |
| `configs.py` | Build or load `CompressionConfig` objects (JSONL or built-in defaults). |
| `cache.py` | Shard-based document / query embedding cache (`.pt` files). |
| `evaluate.py` | Apply compression + PLAID indexing + ColBERT retrieval for one config. |
| `bm25_eval.py` | BM25 sparse-retrieval baseline (optional, enabled via config). |
| `data_loading.py` | Load a BEIR-format dataset (corpus, queries, qrels). |
| `provenance.py` | Write `provenance.json` with full config + environment metadata. |
| `utils.py` | dtype helpers, name sanitisation, tensor pack/unpack utilities. |
| `constants.py` | `CachePaths` dataclass and other shared constants. |

Hydra configs live one level up in `src/conf/`:

| File | Purpose |
|---|---|
| `compression_eval.yaml` | Base defaults (inherited by all named configs). |
| `compression_eval_amazon.yaml` | Amazon-dataset overrides (model, dataset path, BM25 on). |
| `amazon_compression_configs.jsonl` | 16 compression configs used in the paper experiment. |

---

## Pipeline overview

```
CLI (cli.py)
 ├─ build_model()            — load ColBERT, optionally compile + cast dtype
 ├─ load_dataset()           — BEIR corpus / queries / qrels
 ├─ encode_documents_with_cache()   — shard embeddings to disk (.pt)
 ├─ compute_and_cache_idf_stats()   — global IDF needed for IDF-pruning configs
 ├─ encode_queries_with_cache()     — query embeddings (cached)
 │
 ├─ [optional] evaluate_bm25()     — BM25 baseline (bm25.enabled: true)
 │
 └─ for each CompressionConfig:
      evaluate_compression_config()
        ├─ iter_document_shards()  — stream shard from disk
        ├─ apply_compression()     — compress + L2-normalise
        ├─ indexes.PLAID.add_documents()
        ├─ retrieve.ColBERT.retrieve()
        └─ ranx.evaluate()  →  evaluation.json + provenance.json
```

Results are written incrementally to `results.jsonl` so a crash can be resumed
with `compression.resume: true`.

---

## BM25 baseline

BM25 is controlled by the `bm25` config block:

```yaml
bm25:
  enabled: true   # set false to skip
  variant: okapi  # okapi | plus | l
```

When enabled, `bm25_eval.py` runs **before** the ColBERT configs and appends a
`config_bm25/evaluation.json` entry.  Because BM25 operates on word tokens (not
ColBERT sub-word tokens), treat it as a **horizontal reference line** on any
compression-vs-quality plot, not a point on the compression curve.

---

## Compression strategies (built-in defaults)

| Strategy | Description |
|---|---|
| `Baseline` | No compression — full ColBERT embeddings. |
| `Random pruning` | Drop tokens at random, `keep_ratio` ∈ {0.1, 0.2, 0.33, 0.5, 0.75}. |
| `Random pooling` | Merge tokens at random, `keep_ratio` ∈ {0.1, 0.2, 0.33, 0.5}. |
| `Attention pruning` | Drop low-attention tokens. |
| `Attention pooling` | Merge low-attention tokens. |
| `IDF pruning` | Drop high-IDF (common) tokens using corpus-level IDF stats. |
| `Spherical / Hierarchical Pooling` | K-Means or Ward clustering, pool factor ∈ {2, 3, 5, 10}. |

Custom configs can be supplied as a JSONL file (`compression.configs_file`).

---

## Key output files

```
results/<run_name>/<timestamp>/
├─ results.jsonl           # one JSON line per config (appended live)
├─ metadata.json           # run-level summary (model, dataset, timing)
├─ config_bm25/
│   └─ evaluation.json
├─ config_0/               # Baseline
│   ├─ evaluation.json
│   ├─ provenance.json
│   └─ runfile.json        # (if output.save_runfile: true)
├─ config_1/ …
```

