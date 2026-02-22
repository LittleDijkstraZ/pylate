"""CLI entry point for iterative compression evaluation."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm

from pylate import models

from .cache import (
    build_cache_paths,
    compute_and_cache_idf_stats,
    encode_documents_with_cache,
    encode_queries_with_cache,
)
from .configs import create_default_configs, load_configs_from_jsonl
from .datasets import load_dataset
from .evaluate import evaluate_compression_config
from .utils import get_torch_dtype, resolve_query_length

logger = logging.getLogger(__name__)


def build_model(cfg: DictConfig, model_name: str, query_length: int, doc_length: int) -> models.ColBERT:
    """Build and configure the ColBERT model."""
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=doc_length,
        query_length=query_length,
        trust_remote_code=True,
    )
    if cfg.model.compile:
        model.compile()
    model_dtype = get_torch_dtype(cfg.model.dtype)
    current_dtype = next(model.parameters()).dtype
    if current_dtype != model_dtype:
        model = model.to(model_dtype)
    return model


@hydra.main(version_base=None, config_path="../conf", config_name="compression_eval")
def main(cfg: DictConfig) -> None:
    """Main entry point for compression evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    overall_start = time.time()

    model_name = cfg.model.name_or_path
    dataset_id = cfg.dataset.name
    embedding_dtype = get_torch_dtype(cfg.cache.embedding_dtype)

    logger.info("Model: %s", model_name)
    logger.info("Dataset: %s", dataset_id)

    query_length = resolve_query_length(dataset_id, cfg.model.get("query_len"))
    doc_length = cfg.model.doc_len

    logger.info("")
    logger.info("=" * 80)
    logger.info("Loading model...")
    logger.info("=" * 80)
    model = build_model(cfg, model_name, query_length, doc_length)
    logger.info("Model loaded: query_length=%d, doc_length=%d", query_length, doc_length)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Loading dataset...")
    logger.info("=" * 80)
    documents, queries, qrels = load_dataset(dataset_id, lowercase=cfg.dataset.lowercase)

    resume_dir = cfg.compression.get("resume_dir", None)
    resume_mode = cfg.compression.get("resume", False)
    if resume_mode and resume_dir:
        results_dir = Path(resume_dir)
        logger.info("Resuming into existing results directory: %s", results_dir)
    else:
        hydra_cfg = HydraConfig.get()
        results_dir = Path(hydra_cfg.runtime.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = results_dir.name
    logger.info("Output directory: %s", results_dir)
    logger.info("Run ID: %s", run_id)

    cache_paths = build_cache_paths(
        cfg=cfg,
        dataset_id=dataset_id,
        model_name=model_name,
        doc_length=doc_length,
        query_length=query_length,
        lowercase=cfg.dataset.lowercase,
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("Loading compression configurations...")
    logger.info("=" * 80)
    if cfg.compression.configs_file:
        configs = load_configs_from_jsonl(Path(cfg.compression.configs_file), model)
        logger.info("Loaded %d configs from %s", len(configs), cfg.compression.configs_file)
    else:
        configs = create_default_configs(model)
        logger.info("Created %d default configs", len(configs))

    for i, config in enumerate(configs):
        if config is None:
            logger.info("  [%d] Baseline (no compression)", i)
        else:
            logger.info("  [%d] %s", i, config.description)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Encoding documents...")
    logger.info("=" * 80)
    encode_start = time.time()
    num_shards = encode_documents_with_cache(
        model=model,
        documents=documents,
        batch_size=cfg.encode.batch_size,
        shard_size=cfg.encode.shard_size,
        embedding_dtype=embedding_dtype,
        move_to_cpu=cfg.encode.move_embeddings_to_cpu,
        cache_paths=cache_paths,
    )
    encode_time = time.time() - encode_start
    logger.info("Encoded %d documents (%d shards) in %.2fs", len(documents), num_shards, encode_time)

    idf_stats = compute_and_cache_idf_stats(cache_paths, num_shards)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Encoding queries...")
    logger.info("=" * 80)
    query_encode_start = time.time()
    queries_embeddings = encode_queries_with_cache(
        model=model,
        queries=queries,
        batch_size=cfg.encode.batch_size,
        embedding_dtype=embedding_dtype,
        move_to_cpu=cfg.encode.move_embeddings_to_cpu,
        cache_paths=cache_paths,
        cache_enabled=cfg.cache.enable,
    )
    query_encode_time = time.time() - query_encode_start
    logger.info("Encoded %d queries in %.2fs", len(queries_embeddings), query_encode_time)

    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATING COMPRESSION CONFIGS")
    logger.info("=" * 80)

    all_results: List[Dict[str, Any]] = []
    skip_count = cfg.compression.get("skip", 0)
    run_indices = cfg.compression.get("indices", None)

    existing_by_name: Dict[str, Dict[str, Any]] = {}
    next_config_idx: Optional[int] = None
    if resume_mode:
        from .configs import serialize_config_for_storage
        from .evaluate import evaluate_compression_config
        from .provenance import build_provenance
        from .utils import sanitize_dataset_name

        existing_by_name = scan_existing_results(results_dir)
        logger.info("Found %d existing results in %s", len(existing_by_name), results_dir)
        existing_indices = [
            int(d.name.split("_", 1)[1])
            for d in results_dir.glob("config_*")
            if d.is_dir() and d.name.split("_", 1)[1].isdigit()
        ]
        next_config_idx = (max(existing_indices) + 1) if existing_indices else 0
        logger.info("New configs will be numbered starting at config_%d", next_config_idx)

    if run_indices is not None:
        config_sequence = [(i, configs[i]) for i in run_indices]
    else:
        config_sequence = [(i, c) for i, c in enumerate(configs) if i >= skip_count]

    with logging_redirect_tqdm():
        for config_idx, config in config_sequence:
            config_name = "Baseline" if config is None else config.description

            if resume_mode and config_name in existing_by_name:
                logger.info("[%d] Skipping (already completed): %s", config_idx, config_name)
                all_results.append(existing_by_name[config_name])
                continue

            if resume_mode and next_config_idx is not None:
                run_idx = next_config_idx
                next_config_idx += 1
            else:
                run_idx = config_idx

            result = evaluate_compression_config(
                cfg=cfg,
                config_idx=run_idx,
                config=config,
                num_documents=len(documents),
                cache_paths=cache_paths,
                num_shards=num_shards,
                queries=queries,
                queries_embeddings=queries_embeddings,
                qrels=qrels,
                dataset_id=dataset_id,
                model_name=model_name,
                results_dir=results_dir,
                run_id=run_id,
                embedding_dtype=embedding_dtype,
                idf_stats=idf_stats,
            )
            all_results.append(result)

            jsonl_path = results_dir / "results.jsonl"
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

    total_time = time.time() - overall_start

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("Total time: %.2fs", total_time)
    logger.info("Results saved to: %s", results_dir)

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "dataset_id": dataset_id,
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_configs": len(configs),
        "encode_time": encode_time,
        "query_encode_time": query_encode_time,
        "total_time": total_time,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    logger.info("")
    logger.info("%-50s %12s %12s %12s", "Config", "Tokens", "Avg/Doc", "NDCG@10")
    logger.info("-" * 90)
    for result in all_results:
        ndcg10 = result["evaluation"].get("ndcg@10", 0)
        logger.info(
            "%-50s %12d %12.1f %12.4f",
            result["config_name"][:50],
            result["token_count"],
            result["avg_tokens_per_doc"],
            ndcg10,
        )


def scan_existing_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Scan an existing results directory and build a map of config_name -> result data."""
    existing: Dict[str, Dict[str, Any]] = {}
    for config_dir in sorted(results_dir.glob("config_*")):
        eval_file = config_dir / "evaluation.json"
        if not eval_file.exists():
            continue
        try:
            with open(eval_file, "r") as f:
                data = json.load(f)
            name = data.get("config_name")
            if name:
                existing[name] = data
                logger.debug("Found existing result: %s (config_%s)", name, data.get("config_idx"))
        except Exception as exc:
            logger.warning("Could not load %s: %s", eval_file, exc)
    return existing
