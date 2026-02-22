"""Provenance helpers."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf

from pylate.models.compression import CompressionConfig


def get_git_info() -> Dict[str, Any]:
    """Get git commit information."""
    info = {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        info["commit"] = commit.decode("utf-8").strip()
        info["dirty"] = subprocess.call(["git", "diff", "--quiet"]) != 0
    except Exception:
        pass
    return info


def build_provenance(
    cfg,
    dataset_id: str,
    model_name: str,
    compression_config: Optional[CompressionConfig],
    run_id: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Build provenance metadata for a run."""
    from .configs import serialize_config_for_storage

    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_id,
        "model": model_name,
        "compression_config": serialize_config_for_storage(compression_config),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "hostname": platform.node(),
        },
        "git": get_git_info(),
        "stats": stats,
    }
