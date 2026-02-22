"""Constants and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Query length mapping for different datasets.
QUERY_LEN = {
    # BEIR datasets
    "beir/nfcorpus/test": 32,
    "beir/fiqa/test": 32,
    "beir/scidocs": 48,
    "beir/scifact/test": 48,
    "beir/trec-covid": 48,
    "beir/webis-touche2020/v2": 32,
    "beir/quora/test": 32,
    "beir/nq": 32,
    # TREC
    "disks45/nocr/trec-robust-2004": 32,
    # LoTTE datasets
    "lotte/lifestyle/dev/forum": 32,
    "lotte/lifestyle/dev/search": 32,
    "lotte/lifestyle/test/forum": 32,
    "lotte/lifestyle/test/search": 32,
    "lotte/pooled/dev/forum": 32,
    "lotte/pooled/dev/search": 32,
    "lotte/pooled/test/forum": 32,
    "lotte/pooled/test/search": 32,
    "lotte/recreation/dev/forum": 32,
    "lotte/recreation/dev/search": 32,
    "lotte/recreation/test/forum": 32,
    "lotte/recreation/test/search": 32,
    "lotte/science/dev/forum": 32,
    "lotte/science/dev/search": 32,
    "lotte/science/test/forum": 32,
    "lotte/science/test/search": 32,
    "lotte/technology/dev/forum": 32,
    "lotte/technology/dev/search": 32,
    "lotte/technology/test/forum": 32,
    "lotte/technology/test/search": 32,
    "lotte/writing/dev/forum": 32,
    "lotte/writing/dev/search": 32,
    "lotte/writing/test/forum": 32,
    "lotte/writing/test/search": 32,
    # Short names (for backward compatibility with compression_experiment.py)
    "quora": 32,
    "climate-fever": 64,
    "nq": 32,
    "msmarco": 32,
    "hotpotqa": 32,
    "nfcorpus": 32,
    "scifact": 48,
    "trec-covid": 48,
    "fiqa": 32,
    "arguana": 64,
    "scidocs": 48,
    "dbpedia-entity": 32,
    "webis-touche2020": 32,
    "fever": 32,
}


@dataclass(frozen=True)
class CachePaths:
    """Paths for embedding cache files."""

    cache_dir: Path
    doc_meta: Path
    query_meta: Path
    doc_emb_shard_pattern: str
    doc_art_shard_pattern: str
    doc_idf_stats_file: Path
    query_file: Path
