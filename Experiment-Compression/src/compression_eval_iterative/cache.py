"""Embedding cache and shard helpers."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from pylate import models
from pylate.models.utils import TokenTFIDFStats

from .constants import CachePaths
from .utils import (
    cast_embeddings,
    move_embeddings_to_cpu,
    pack_embeddings,
    sanitize_dataset_name,
    sanitize_model_name,
    unpack_embeddings,
)

logger = logging.getLogger(__name__)


def build_cache_paths(
    cfg,
    dataset_id: str,
    model_name: str,
    doc_length: int,
    query_length: int,
    lowercase: bool,
) -> CachePaths:
    """Build cache paths for embeddings."""
    dataset_slug = sanitize_dataset_name(dataset_id)
    model_slug = sanitize_model_name(model_name)
    base_dir = (
        Path(cfg.cache.dir)
        / dataset_slug
        / model_slug
        / f"schema_v{cfg.cache.schema_version}"
        / f"doclen{doc_length}"
        / f"qlen{query_length}"
        / f"lower{int(lowercase)}"
        / f"edtype_{cfg.cache.embedding_dtype}"
        / f"mdtype_{cfg.model.dtype}"
    )
    return CachePaths(
        cache_dir=base_dir,
        doc_meta=base_dir / "doc_meta.json",
        query_meta=base_dir / "query_meta.json",
        doc_emb_shard_pattern="shard_{:03d}_emb.pt",
        doc_art_shard_pattern="shard_{:03d}_art.pt",
        doc_idf_stats_file=base_dir / "idf_stats.pt",
        query_file=base_dir / "query.pt",
    )


def encode_documents_with_cache(
    model: models.ColBERT,
    documents: List[Dict[str, str]],
    batch_size: int,
    shard_size: int,
    embedding_dtype: torch.dtype,
    move_to_cpu: bool,
    cache_paths: CachePaths,
) -> int:
    """Encode documents to split shard cache files. Returns num_shards."""
    num_documents = len(documents)
    num_shards = (num_documents + shard_size - 1) // shard_size
    cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)

    for shard_idx in range(num_shards):
        emb_path = cache_paths.cache_dir / cache_paths.doc_emb_shard_pattern.format(shard_idx)
        art_path = cache_paths.cache_dir / cache_paths.doc_art_shard_pattern.format(shard_idx)

        if emb_path.exists() and art_path.exists():
            logger.info("Shard %d/%d already cached, skipping.", shard_idx + 1, num_shards)
            continue

        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, num_documents)
        shard_documents = documents[start_idx:end_idx]
        shard_doc_ids = [doc["id"] for doc in shard_documents]
        logger.info(
            "Encoding shard %d/%d (documents %d to %d)",
            shard_idx + 1,
            num_shards,
            start_idx,
            end_idx - 1,
        )

        shard_embeddings, shard_artifacts = model.encode(
            sentences=[doc["text"] for doc in shard_documents],
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
            return_extra_artifacts={"input_ids": True, "attention_scores": True},
        )

        shard_embeddings = cast_embeddings(shard_embeddings, embedding_dtype)
        if move_to_cpu:
            shard_embeddings = move_embeddings_to_cpu(shard_embeddings)
            for key in shard_artifacts:
                shard_artifacts[key] = [
                    t.cpu() if hasattr(t, "cpu") else t for t in shard_artifacts[key]
                ]

        packed_emb = pack_embeddings(shard_embeddings)
        torch.save(
            {
                "embeddings": packed_emb["embeddings"],
                "lengths": packed_emb["lengths"],
                "doc_ids": shard_doc_ids,
            },
            emb_path,
        )

        packed_art: Dict[str, Any] = {"doc_ids": shard_doc_ids}
        for key, tensors in shard_artifacts.items():
            packed = pack_embeddings(tensors)
            packed_art[key] = packed["embeddings"]
            packed_art[f"{key}_lengths"] = packed["lengths"]
        torch.save(packed_art, art_path)

    cache_paths.doc_meta.write_text(
        json.dumps(
            {
                "num_documents": num_documents,
                "num_shards": num_shards,
                "shard_size": shard_size,
                "created_at": datetime.now().isoformat(),
                "format": "v3_split_emb_art",
            },
            indent=2,
        )
        + "\n"
    )

    return num_shards


def iter_document_shards(
    cache_paths: CachePaths,
    num_shards: int,
    embedding_dtype: torch.dtype,
    device: str = "cpu",
) -> Iterable[Tuple[List[str], List[torch.Tensor], Dict[str, List[torch.Tensor]]]]:
    """Yield (doc_ids, embeddings, artifacts) for each cached document shard."""
    for shard_idx in range(num_shards):
        emb_path = cache_paths.cache_dir / cache_paths.doc_emb_shard_pattern.format(shard_idx)
        art_path = cache_paths.cache_dir / cache_paths.doc_art_shard_pattern.format(shard_idx)

        logger.info("Loading shard %d embeddings from %s", shard_idx, emb_path)
        logger.info("Loading shard %d artifacts from %s", shard_idx, art_path)
        logger.info("Loading shard onto device %s", device)

        start_time = time.time()
        emb_data = torch.load(emb_path, map_location=device)
        logger.info("Loading shard %d embeddings took %.3fs", shard_idx, time.time() - start_time)
        start_time = time.time()
        art_data = torch.load(art_path, map_location=device)
        logger.info("Loading shard %d artifacts took %.3fs", shard_idx, time.time() - start_time)

        doc_ids = emb_data["doc_ids"]
        embeddings = unpack_embeddings(
            {"embeddings": emb_data["embeddings"], "lengths": emb_data["lengths"]}
        )
        embeddings = cast_embeddings(embeddings, embedding_dtype)

        artifacts: Dict[str, List[torch.Tensor]] = {}
        for key in ("input_ids", "attention_scores"):
            if key in art_data:
                artifacts[key] = unpack_embeddings(
                    {"embeddings": art_data[key], "lengths": art_data[f"{key}_lengths"]}
                )

        yield doc_ids, embeddings, artifacts


def compute_and_cache_idf_stats(
    cache_paths: CachePaths,
    num_shards: int,
) -> TokenTFIDFStats:
    """Build global IDF stats from cached artifact shards. Returns cached stats if available."""
    if cache_paths.doc_idf_stats_file.exists():
        logger.info("Loading cached IDF stats from %s", cache_paths.doc_idf_stats_file)
        return torch.load(cache_paths.doc_idf_stats_file, map_location="cpu", weights_only=False)

    logger.info("Computing global IDF stats from %d artifact shards...", num_shards)
    all_input_ids: List[List[int]] = []
    for shard_idx in range(num_shards):
        art_path = cache_paths.cache_dir / cache_paths.doc_art_shard_pattern.format(shard_idx)
        art_data = torch.load(art_path, map_location="cpu")
        shard_ids = unpack_embeddings(
            {"embeddings": art_data["input_ids"], "lengths": art_data["input_ids_lengths"]}
        )
        all_input_ids.extend(ids.cpu().tolist() for ids in shard_ids)

    stats = TokenTFIDFStats(num_docs=len(all_input_ids))
    stats.fit(all_input_ids, show_progress=True)

    torch.save(stats, cache_paths.doc_idf_stats_file)
    logger.info("Saved IDF stats (%d docs) to %s", len(all_input_ids), cache_paths.doc_idf_stats_file)
    return stats


def encode_queries_with_cache(
    model: models.ColBERT,
    queries: Dict[str, str],
    batch_size: int,
    embedding_dtype: torch.dtype,
    move_to_cpu: bool,
    cache_paths: CachePaths,
    cache_enabled: bool,
) -> List[torch.Tensor]:
    """Encode queries with caching support."""
    if cache_enabled and cache_paths.query_file.exists():
        packed = torch.load(cache_paths.query_file, map_location="cpu")
        query_embeddings = unpack_embeddings(packed)
        query_embeddings = cast_embeddings(query_embeddings, embedding_dtype)
        if move_to_cpu:
            query_embeddings = move_embeddings_to_cpu(query_embeddings)
        return query_embeddings

    logger.info("Encoding queries...")
    query_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    query_embeddings = cast_embeddings(query_embeddings, embedding_dtype)
    if move_to_cpu:
        query_embeddings = move_embeddings_to_cpu(query_embeddings)

    if cache_enabled:
        cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
        packed = pack_embeddings(query_embeddings)
        torch.save(packed, cache_paths.query_file)
        cache_paths.query_meta.write_text(
            json.dumps(
                {
                    "num_queries": len(queries),
                    "created_at": datetime.now().isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
    return query_embeddings
