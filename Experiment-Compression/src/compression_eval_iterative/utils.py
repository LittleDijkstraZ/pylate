"""Shared utility helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pylate import models

from .constants import QUERY_LEN


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


def versioned_dir(base: Path) -> Path:
    """Return *base* if it does not exist, otherwise *base_YYYYMMDD_HHMMSS*.

    This lets callers write results to a stable name on the first run and
    automatically version subsequent runs without overwriting previous output.

    Example::

        # first run  → results/amazon/baselines/config_dense_mpnet
        # second run → results/amazon/baselines/config_dense_mpnet_20260309_142301
    """
    if not base.exists():
        return base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base.parent / f"{base.name}_{timestamp}"


def short_hash(payload: Dict[str, Any], length: int = 10) -> str:
    """Generate a short hash from a dictionary payload."""
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:length]


def resolve_query_length(dataset_id: str, override: Optional[int]) -> int:
    """Resolve query length from dataset ID or override."""
    if override is not None:
        return override
    return QUERY_LEN.get(dataset_id, 32)


def get_embedding_size(model: models.ColBERT) -> int:
    """Get the embedding dimension from a model."""
    if hasattr(model, "get_sentence_embedding_dimension"):
        return int(model.get_sentence_embedding_dimension())
    try:
        last = model[-1]
        if hasattr(last, "out_features"):
            return int(last.out_features)
    except Exception:
        pass
    return 128


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
