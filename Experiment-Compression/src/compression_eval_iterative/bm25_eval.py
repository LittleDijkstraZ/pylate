"""BM25 baseline evaluation for the iterative compression pipeline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25L, BM25Okapi, BM25Plus
from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate

logger = logging.getLogger(__name__)

_VARIANTS = {"okapi": BM25Okapi, "plus": BM25Plus, "l": BM25L}


def evaluate_bm25(
    cfg,
    documents: List[Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    results_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """Run BM25 retrieval and evaluate with the same metrics as the ColBERT pipeline.

    BM25 operates on word-level terms (not subword tokens), so ``token_count``
    and ``avg_tokens_per_doc`` in the returned result reflect word counts.  When
    plotting, treat BM25 as a horizontal reference line rather than a point on
    the compression curve (whose x-axis is ColBERT subword token count).
    """
    variant = cfg.bm25.get("variant", "okapi")
    top_k = cfg.retrieve.k
    lowercase = cfg.dataset.lowercase

    def tokenize(text: str) -> List[str]:
        return (text.lower() if lowercase else text).split()

    logger.info("BM25: tokenizing %d documents...", len(documents))
    tokenized_corpus = [tokenize(doc["text"]) for doc in documents]
    doc_ids = [doc["id"] for doc in documents]

    # Build index
    t0 = time.time()
    BM25Class = _VARIANTS.get(variant, BM25Okapi)
    bm25 = BM25Class(tokenized_corpus)
    index_time = time.time() - t0
    logger.info("BM25: built %s index in %.2fs", variant, index_time)

    # Retrieve top-k per query
    t0 = time.time()
    run_dict: Dict[str, Dict[str, float]] = {}
    for query_id, query_text in queries.items():
        tok_query = tokenize(query_text)
        scores = bm25.get_scores(tok_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        run_dict[query_id] = {
            doc_ids[i]: float(scores[i])
            for i in top_indices
            if doc_ids[i] != query_id  # exclude self-matches, same as ColBERT pipeline
        }
    retrieve_time = time.time() - t0
    logger.info("BM25: retrieved top-%d for %d queries in %.2fs", top_k, len(queries), retrieve_time)

    # Evaluate
    qrels_obj = Qrels(qrels=qrels)
    run_obj = Run(run=run_dict)
    evaluation_scores = ranx_evaluate(
        qrels=qrels_obj,
        run=run_obj,
        metrics=list(cfg.metrics),
        make_comparable=True,
    )
    logger.info("BM25 evaluation scores:")
    for metric, value in evaluation_scores.items():
        logger.info("  %s: %.4f", metric, value)

    # Word-count stats (BM25 term count, not ColBERT subword tokens)
    word_counts = [len(t) for t in tokenized_corpus]
    total_words = sum(word_counts)
    avg_words_per_doc = total_words / len(documents) if documents else 0.0

    result: Dict[str, Any] = {
        "config_idx": "bm25",
        "config_name": f"BM25 ({variant})",
        "config": {"type": "bm25", "variant": variant},
        "token_count": total_words,
        "avg_tokens_per_doc": avg_words_per_doc,
        "compression_time": 0.0,
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "evaluation": evaluation_scores,
    }

    # Save per-config artifacts (config_bm25/ doesn't conflict with config_0/, config_1/, ...)
    config_dir = results_dir / "config_bm25"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    if cfg.output.save_runfile:
        run_obj.save((config_dir / "runfile.json").as_posix())

    return result
