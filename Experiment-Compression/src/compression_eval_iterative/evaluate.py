"""Compression evaluation routines."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate
from tqdm.auto import tqdm

from pylate import indexes, retrieve
from pylate.models.compression import CompressionConfig
from pylate.models.utils import TokenTFIDFStats

from .cache import iter_document_shards
from .configs import serialize_config_for_storage
from .provenance import build_provenance
from .utils import sanitize_dataset_name, sanitize_model_name

logger = logging.getLogger(__name__)


def apply_compression(
    embeddings: List[torch.Tensor],
    artifacts: Optional[Dict[str, List[torch.Tensor]]],
    compressor,
    batch_size: int,
    num_workers: int,
    show_progress: bool = True,
) -> Tuple[List[torch.Tensor], float]:
    """Apply compression to embeddings. Returns compressed embeddings and compression time."""
    compression_start_time = time.time()
    compressed, _ = compressor.compress_parallel(
        embeddings=embeddings,
        artifacts=artifacts if artifacts else {},
        batch_size=batch_size,
        num_workers=num_workers,
        show_progress=show_progress,
    )
    compression_time = time.time() - compression_start_time
    logger.info("Compression time: %.3fs", compression_time)
    post_normalize_start_time = time.time()
    compressed = [
        F.normalize(emb, p=2, dim=-1) for emb in tqdm(compressed, desc="post-compression normalization")
    ]
    post_normalize_time = time.time() - post_normalize_start_time
    logger.info("Post-compression normalization time: %.3fs", post_normalize_time)

    return compressed, compression_time + post_normalize_time


def evaluate_compression_config(
    cfg,
    config_idx: int,
    config: Optional[CompressionConfig],
    num_documents: int,
    cache_paths,
    num_shards: int,
    queries: Dict[str, str],
    queries_embeddings: List[torch.Tensor],
    qrels: Dict[str, Dict[str, int]],
    dataset_id: str,
    model_name: str,
    results_dir: Path,
    run_id: str,
    embedding_dtype: torch.dtype,
    idf_stats: Optional[TokenTFIDFStats] = None,
) -> Dict[str, Any]:
    """Evaluate a single compression configuration using shard-by-shard compression and indexing."""
    config_name = config.description if config else "Baseline"
    logger.info("")
    logger.info("=" * 80)
    logger.info("[%d] Evaluating: %s", config_idx, config_name)
    logger.info("=" * 80)

    dataset_slug = sanitize_dataset_name(dataset_id)
    model_slug = sanitize_model_name(model_name)
    index_name = f"{dataset_slug}_{model_slug}_config_{config_idx}"

    index_kwargs = dict(
        override=True,
        index_name=index_name,
        use_fast=cfg.index.use_fast,
        nbits=cfg.index.nbits,
        kmeans_niters=cfg.index.kmeans_niters,
        max_points_per_centroid=cfg.index.max_points_per_centroid,
        n_ivf_probe=cfg.index.n_ivf_probe,
        n_full_scores=cfg.index.n_full_scores,
        n_samples_kmeans=cfg.index.n_samples_kmeans,
        batch_size=cfg.index.batch_size,
        show_progress=True,
        device=cfg.index.device,
        use_triton=cfg.index.use_triton,
    )

    compressor = config.create_compressor() if config is not None else None
    num_workers = max(1, min(os.cpu_count() or 1, 32))
    logger.info("Using %d workers for compression", num_workers)

    index = indexes.PLAID(**index_kwargs)

    logger.info("Beginning compression...")
    num_tokens = 0
    compression_time = 0.0
    index_time = 0.0

    for shard_idx, (shard_doc_ids, shard_embeddings, shard_artifacts) in enumerate(
        iter_document_shards(cache_paths, num_shards, embedding_dtype, device=cfg.index.device)
    ):
        if idf_stats is not None:
            shard_artifacts = {**shard_artifacts, "tfidf_stats": idf_stats}

        if compressor is not None:
            compressed, shard_compression_time = apply_compression(
                embeddings=shard_embeddings,
                artifacts=shard_artifacts,
                compressor=compressor,
                batch_size=cfg.encode.batch_size,
                num_workers=num_workers,
                show_progress=True,
            )
        else:
            compressed = [F.normalize(emb, p=2, dim=-1) for emb in shard_embeddings]
            shard_compression_time = 0.0

        compression_time += shard_compression_time
        num_tokens += sum(len(emb) for emb in compressed)
        logger.info(
            "  shard %d/%d: %d docs, compression %.3fs",
            shard_idx + 1,
            num_shards,
            len(shard_doc_ids),
            shard_compression_time,
        )

        t0 = time.time()
        index.add_documents(
            documents_ids=shard_doc_ids,
            documents_embeddings=compressed,
            batch_size=cfg.index.batch_size,
        )
        shard_index_time = time.time() - t0
        index_time += shard_index_time
        logger.info(
            "  shard %d/%d: index add %.3fs",
            shard_idx + 1,
            num_shards,
            shard_index_time,
        )

    avg_tokens_per_doc = num_tokens / num_documents if num_documents else 0
    logger.info("Token count: %d, Avg tokens/doc: %.1f", num_tokens, avg_tokens_per_doc)

    logger.info("Retrieving...")
    retriever = retrieve.ColBERT(index=index)
    retrieve_start = time.time()
    scores = retriever.retrieve(
        queries_embeddings=queries_embeddings,
        k=cfg.retrieve.k,
    )
    retrieve_time = time.time() - retrieve_start

    for query_id, query_scores in zip(queries.keys(), scores):
        query_scores[:] = [score for score in query_scores if score["id"] != query_id]

    query_list = list(queries.keys())
    qrels_obj = Qrels(qrels=qrels)
    run_dict = {
        query: {match["id"]: match["score"] for match in query_matches}
        for query, query_matches in zip(query_list, scores)
    }
    run = Run(run=run_dict)

    evaluation_scores = ranx_evaluate(
        qrels=qrels_obj,
        run=run,
        metrics=list(cfg.metrics),
        make_comparable=True,
    )

    logger.info("Evaluation scores:")
    for metric, value in evaluation_scores.items():
        logger.info("  %s: %.4f", metric, value)

    result = {
        "config_idx": config_idx,
        "config_name": config_name,
        "config": serialize_config_for_storage(config),
        "token_count": num_tokens,
        "avg_tokens_per_doc": avg_tokens_per_doc,
        "compression_time": compression_time,
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "evaluation": evaluation_scores,
    }

    config_run_dir = results_dir / f"config_{config_idx}"
    config_run_dir.mkdir(parents=True, exist_ok=True)

    provenance = build_provenance(
        cfg=cfg,
        dataset_id=dataset_id,
        model_name=model_name,
        compression_config=config,
        run_id=f"{run_id}_config_{config_idx}",
        stats={
            "num_documents": num_documents,
            "num_queries": len(queries),
            "token_count": num_tokens,
            "avg_tokens_per_doc": avg_tokens_per_doc,
            "compression_time": compression_time,
            "index_time": index_time,
            "retrieve_time": retrieve_time,
        },
    )
    (config_run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (config_run_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")

    if cfg.output.save_runfile:
        run.save((config_run_dir / "runfile.json").as_posix())

    return result
