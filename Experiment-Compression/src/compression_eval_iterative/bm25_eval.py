"""BM25 baseline evaluation for the iterative compression pipeline.

Multiple BM25 variants (okapi, plus, l) can be evaluated in one run by listing
them under ``bm25.variants`` in the Hydra config (see compression_eval.yaml).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25L, BM25Okapi, BM25Plus
from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate

from .utils import versioned_dir

logger = logging.getLogger(__name__)

_VARIANTS = {"okapi": BM25Okapi, "plus": BM25Plus, "l": BM25L}


def _resolve_bm25_variants(bm25_cfg) -> List[str]:
    """Return the list of BM25 variant names to evaluate.

    New format (preferred)::

        bm25:
          variants: [okapi, plus]

    Legacy format (still supported)::

        bm25:
          variant: okapi
    """
    variants_list = bm25_cfg.get("variants", None)
    if variants_list is not None:
        return list(variants_list)
    return [bm25_cfg.get("variant", "okapi")]


def _evaluate_one_bm25(
    variant: str,
    tokenized_corpus: List[List[str]],
    doc_ids: List[str],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    metrics: List[str],
    top_k: int,
    lowercase: bool,
    results_dir: Path,
    save_runfile: bool,
) -> Dict[str, Any]:
    """Build a BM25 index for one variant, retrieve, and evaluate."""

    def tokenize(text: str) -> List[str]:
        return (text.lower() if lowercase else text).split()

    BM25Class = _VARIANTS.get(variant, BM25Okapi)

    t0 = time.time()
    bm25 = BM25Class(tokenized_corpus)
    index_time = time.time() - t0
    logger.info("BM25 [%s]: built index in %.2fs", variant, index_time)

    t0 = time.time()
    run_dict: Dict[str, Dict[str, float]] = {}
    for query_id, query_text in queries.items():
        tok_query = tokenize(query_text)
        scores = bm25.get_scores(tok_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        run_dict[query_id] = {
            doc_ids[i]: float(scores[i])
            for i in top_indices
            if doc_ids[i] != query_id
        }
    retrieve_time = time.time() - t0
    logger.info("BM25 [%s]: retrieved top-%d for %d queries in %.2fs", variant, top_k, len(queries), retrieve_time)

    qrels_obj = Qrels(qrels=qrels)
    run_obj = Run(run=run_dict)
    evaluation_scores = ranx_evaluate(qrels=qrels_obj, run=run_obj, metrics=metrics, make_comparable=True)
    for metric, value in evaluation_scores.items():
        logger.info("  %s: %.4f", metric, value)

    word_counts = [len(t) for t in tokenized_corpus]
    total_words = sum(word_counts)
    avg_words_per_doc = total_words / len(tokenized_corpus) if tokenized_corpus else 0.0

    result: Dict[str, Any] = {
        "config_idx": f"bm25_{variant}",
        "config_name": f"BM25 ({variant})",
        "config": {"type": "bm25", "variant": variant},
        "token_count": total_words,
        "avg_tokens_per_doc": avg_words_per_doc,
        "compression_time": 0.0,
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "evaluation": evaluation_scores,
    }

    config_dir = versioned_dir(results_dir / f"config_bm25_{variant}")
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    if save_runfile:
        run_obj.save((config_dir / "runfile.json").as_posix())

    return result


def evaluate_bm25(
    cfg,
    documents: List[Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    results_dir: Path,
    run_id: str,
) -> List[Dict[str, Any]]:
    """Run BM25 retrieval for every configured variant and evaluate with the same metrics as ColBERT.

    BM25 operates on word-level terms (not subword tokens), so ``token_count``
    and ``avg_tokens_per_doc`` in the returned results reflect word counts.
    """
    variants = _resolve_bm25_variants(cfg.bm25)
    lowercase = cfg.dataset.lowercase

    def tokenize(text: str) -> List[str]:
        return (text.lower() if lowercase else text).split()

    logger.info("BM25: tokenizing %d documents...", len(documents))
    tokenized_corpus = [tokenize(doc["text"]) for doc in documents]
    doc_ids = [doc["id"] for doc in documents]

    results: List[Dict[str, Any]] = []
    for variant in variants:
        logger.info("")
        logger.info("-" * 60)
        logger.info("BM25 variant: %s", variant)
        logger.info("-" * 60)
        results.append(_evaluate_one_bm25(
            variant=variant,
            tokenized_corpus=tokenized_corpus,
            doc_ids=doc_ids,
            queries=queries,
            qrels=qrels,
            metrics=list(cfg.metrics),
            top_k=cfg.retrieve.k,
            lowercase=lowercase,
            results_dir=results_dir,
            save_runfile=cfg.output.save_runfile,
        ))
    return results
