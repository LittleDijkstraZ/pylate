"""Compression config helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pylate import models
from pylate.models.compression import (
    CompressionConfig,
    IDFPruningConfig,
    IDFPruningStrategy,
    PoolingConfig,
    PoolingStrategy,
    AttentionPruningConfig,
    AttentionPruningStrategy,
    AttentionPoolingConfig,
    AttentionPoolingStrategy,
    RandomPruningConfig,
    RandomPruningStrategy,
    RandomPoolingConfig,
    RandomPoolingStrategy,
)


def serialize_config_for_storage(config: Optional[CompressionConfig]) -> Dict[str, Any]:
    """Serialize a compression config (or baseline) for storage."""
    if config is None:
        return {"type": "baseline", "description": "No compression"}
    return config.serialize()


def load_configs_from_jsonl(jsonl_path: Path, model: models.ColBERT) -> List[Optional[CompressionConfig]]:
    """Load compression configurations from a JSONL file."""
    configs: List[Optional[CompressionConfig]] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if (
                data.get("type") == "baseline"
                or not data.get("strategies")
                or len(data.get("strategies", [])) == 0
            ):
                configs.append(None)
            else:
                config = CompressionConfig.from_dict(data)
                for strategy in config.strategies:
                    if isinstance(strategy, IDFPruningStrategy) and strategy.config.ignore_token_ids is None:
                        strategy.config.ignore_token_ids = model.tokenizer.all_special_ids
                configs.append(config)
    return configs


def create_default_configs(model: models.ColBERT) -> List[Optional[CompressionConfig]]:
    """Create default compression configurations."""
    configs: List[Optional[CompressionConfig]] = [None]

    pruning_keep_ratios = [0.1, 0.2, 0.33, 0.5, 0.75]
    pooling_keep_ratios = [0.1, 0.2, 0.33, 0.5]

    for keep_ratio in pruning_keep_ratios:
        rand_prune_cfg = RandomPruningConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            seed=666,
        )
        configs.append(
            CompressionConfig(
                strategies=[RandomPruningStrategy(rand_prune_cfg)],
                description=f"Random pruning keep_ratio={keep_ratio}",
            )
        )

    for keep_ratio in pooling_keep_ratios:
        rand_pool_cfg = RandomPoolingConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            seed=666,
        )
        configs.append(
            CompressionConfig(
                strategies=[RandomPoolingStrategy(rand_pool_cfg)],
                description=f"Random pooling keep_ratio={keep_ratio}",
            )
        )

    for keep_ratio in pruning_keep_ratios:
        attention_config = AttentionPruningConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            track_pruned_tokens=False,
        )
        configs.append(
            CompressionConfig(
                strategies=[AttentionPruningStrategy(attention_config)],
                description=f"Attention score pruning keep_ratio={keep_ratio}",
            )
        )

    for keep_ratio in pooling_keep_ratios:
        attention_pool_config = AttentionPoolingConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            show_progress_bar=True,
        )
        configs.append(
            CompressionConfig(
                strategies=[AttentionPoolingStrategy(attention_pool_config)],
                description=f"Attention score pooling keep_ratio={keep_ratio}",
            )
        )

    for keep_ratio in pruning_keep_ratios:
        idf_pruning_config = IDFPruningConfig(
            mode="document",
            keep_ratio=keep_ratio,
            protected_tokens=1,
            ignore_token_ids=model.tokenizer.added_tokens_decoder.keys(),
            use_tfidf=False,
            track_pruned_tokens=False,
        )
        configs.append(
            CompressionConfig(
                strategies=[IDFPruningStrategy(idf_pruning_config)],
                description=f"Doc-wise IDF pruning keep_ratio={keep_ratio}",
            )
        )

    for method in ["spherical", "hierarchical"]:
        for k in [2, 3, 5, 10]:
            pooling_config = PoolingConfig(
                pool_factor=k,
                protected_tokens=1,
                clustering_method=method,
                show_progress_bar=True,
                kmeans_gpu=True,
                hierarchical_variant="ward_embeddings",
            )
            configs.append(
                CompressionConfig(
                    strategies=[PoolingStrategy(pooling_config)],
                    description=f"{method.capitalize()} Pooling f={k} protected tokens=1",
                )
            )

    return configs
