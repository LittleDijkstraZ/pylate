"""Dataset loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import ir_datasets
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _load_dataset_local(
    dataset_id: str, lowercase: bool = False
) -> Tuple[List[Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load a local BEIR-format dataset directory (corpus/queries/qrels sub-structure)."""
    from pylate import evaluation as pylate_evaluation

    logger.info("Detected local path — loading BEIR-format dataset from: %s", dataset_id)
    documents, queries, qrels = pylate_evaluation.load_custom_dataset(dataset_id, split="test")

    if lowercase:
        documents = [{"id": doc["id"], "text": doc["text"].lower()} for doc in documents]
        queries = {qid: q.lower() for qid, q in queries.items()}

    logger.info(
        "Loaded %d documents, %d queries, %d queries with qrels",
        len(documents),
        len(queries),
        len(qrels),
    )
    return documents, queries, qrels


def load_dataset_irds(
    dataset_id: str, lowercase: bool = False
) -> Tuple[List[Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load dataset using ir_datasets."""
    logger.info("Loading dataset: %s", dataset_id)
    try:
        dataset = ir_datasets.load(dataset_id)
    except Exception as exc:
        raise ValueError(
            f"Failed to load dataset '{dataset_id}': {exc}. "
            "Make sure the dataset ID is correct and ir_datasets is installed."
        ) from exc

    if not dataset.has_docs():
        raise ValueError(f"Dataset '{dataset_id}' does not have documents")
    if not dataset.has_queries():
        raise ValueError(f"Dataset '{dataset_id}' does not have queries")

    documents: List[Dict[str, str]] = []
    logger.info("Loading documents...")
    for doc in tqdm(dataset.docs_iter(), desc="Loading documents", unit="docs"):
        if hasattr(doc, "title") and doc.title:
            text = f"{doc.title}\n\n{doc.text}".strip()
        else:
            text = doc.text.strip()
        if lowercase:
            text = text.lower()
        documents.append({"id": doc.doc_id, "text": text})

    queries: Dict[str, str] = {}
    logger.info("Loading queries...")
    for query in tqdm(dataset.queries_iter(), desc="Loading queries", unit="queries"):
        query_text = query.text.strip()
        if lowercase:
            query_text = query_text.lower()
        queries[query.query_id] = query_text

    qrels: Dict[str, Dict[str, int]] = {}
    if dataset.has_qrels():
        logger.info("Loading qrels...")
        for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels", unit="qrels"):
            relevance = int(qrel.relevance)
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = relevance
    else:
        logger.warning("Dataset '%s' does not have qrels", dataset_id)

    logger.info(
        "Loaded %d documents, %d queries, %d queries with qrels",
        len(documents),
        len(queries),
        len(qrels),
    )
    return documents, queries, qrels


def _looks_like_local_path(dataset_id: str) -> bool:
    """Return True if the string looks like a filesystem path rather than an ir_datasets ID.

    Checks both string heuristics (starts with /, ./, ../, ~) and whether the
    path actually resolves to an existing directory.  Using heuristics avoids
    false-negatives when Hydra has changed the working directory so that a
    relative path no longer resolves from the new CWD.
    """
    p = dataset_id
    if p.startswith("/") or p.startswith("./") or p.startswith("../") or p.startswith("~"):
        return True
    # Also accept bare names that resolve to an existing directory (absolute or
    # relative to the current CWD — handles any case Hydra has NOT changed it).
    return Path(p).is_dir()


def load_dataset(
    dataset_id: str, lowercase: bool = False
) -> Tuple[List[Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load a dataset, dispatching to a local BEIR loader or ir_datasets."""
    if _looks_like_local_path(dataset_id):
        return _load_dataset_local(dataset_id, lowercase)
    return load_dataset_irds(dataset_id, lowercase)
