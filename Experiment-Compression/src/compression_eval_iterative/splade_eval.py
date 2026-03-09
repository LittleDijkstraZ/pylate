"""SPLADE baseline evaluation for the iterative compression pipeline.

Supports SPLADE v2 and v3. Encodes each text as a sparse vector over the
full vocabulary using:
    vec = log(1 + relu(MLM_logits)).max(dim=tokens)
Retrieval is the inner product of query vs document sparse vectors.

Multiple SPLADE models can be evaluated in one run by listing them under
``splade.models`` in the Hydra config (see compression_eval.yaml).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.sparse import csr_matrix, vstack
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .utils import versioned_dir

from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "naver/splade-v3"


def _short_name(model_name: str) -> str:
    """Filesystem-safe short label: last path component, dots → dashes."""
    return model_name.split("/")[-1].replace(".", "-")


def _resolve_splade_models(splade_cfg) -> List[Dict[str, Any]]:
    """Return a list of per-model config dicts, merging shared defaults with per-model overrides.

    New format (preferred)::

        splade:
          batch_size: 64
          models:
            - model_name_or_path: naver/splade-v3
              max_length: 256
            - model_name_or_path: naver/splade-cocondenser-ensembledistil

    Legacy format (still supported)::

        splade:
          model_name_or_path: naver/splade-v3
          max_length: 256
    """
    shared = {
        "batch_size": int(splade_cfg.get("batch_size", 64)),
        "max_length": int(splade_cfg.get("max_length", 256)),
        "retrieve_query_batch_size": int(splade_cfg.get("retrieve_query_batch_size", 512)),
        "device": splade_cfg.get("device", None),
    }
    models_list = splade_cfg.get("models", None)
    if models_list is not None:
        result = []
        for m in models_list:
            merged = dict(shared)
            merged.update(OmegaConf.to_container(m, resolve=True))
            result.append(merged)
        return result
    # Legacy: single model_name_or_path at top level
    return [{**shared, "model_name_or_path": splade_cfg.get("model_name_or_path", _DEFAULT_MODEL)}]


def _encode_to_sparse(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    device: torch.device,
    max_length: int,
    desc: str = "Encoding",
) -> csr_matrix:
    """Encode *texts* to a (N, vocab_size) scipy CSR sparse matrix.

    Aggregation: ``log(1 + relu(logits)).max(dim=seq_len)`` — the SPLADE
    max-aggregation formula.  Each resulting vector is sparse over vocabulary.
    """
    vocab_size = model.config.vocab_size
    batches: List[csr_matrix] = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits  # (B, seq_len, vocab_size)

        # SPLADE v2: log(1 + relu(x)), max-pool over token positions → (B, vocab_size)
        vecs = torch.log1p(torch.relu(logits)).max(dim=1).values.cpu().float()

        # Convert to sparse row-by-row
        rows, cols, data = [], [], []
        for local_i, vec in enumerate(vecs):
            nz = (vec > 0).nonzero(as_tuple=True)[0]
            if len(nz):
                rows.extend([local_i] * len(nz))
                cols.extend(nz.tolist())
                data.extend(vec[nz].tolist())

        batches.append(
            csr_matrix(
                (data, (rows, cols)),
                shape=(len(batch_texts), vocab_size),
                dtype=np.float32,
            )
        )

    return vstack(batches, format="csr") if batches else csr_matrix((0, vocab_size), dtype=np.float32)


def _evaluate_one_splade(
    model_cfg: Dict[str, Any],
    doc_ids: List[str],
    doc_texts: List[str],
    query_ids: List[str],
    query_texts: List[str],
    qrels: Dict[str, Dict[str, int]],
    metrics: List[str],
    top_k: int,
    results_dir: Path,
    save_runfile: bool,
) -> Dict[str, Any]:
    """Encode, retrieve, and evaluate for a single SPLADE model config."""
    model_name = model_cfg["model_name_or_path"]
    batch_size = int(model_cfg.get("batch_size", 64))
    max_length = int(model_cfg.get("max_length", 256))
    retrieve_q_batch = int(model_cfg.get("retrieve_query_batch_size", 512))
    device_cfg = model_cfg.get("device", None)
    device = torch.device(device_cfg) if device_cfg else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    short = _short_name(model_name)

    logger.info("SPLADE [%s]: loading on %s", short, device)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    logger.info("SPLADE [%s]: loaded in %.2fs", short, time.time() - t0)

    # --- Encode documents ---
    t0 = time.time()
    doc_matrix = _encode_to_sparse(
        model, tokenizer, doc_texts, batch_size, device, max_length,
        desc=f"SPLADE [{short}] docs",
    )
    index_time = time.time() - t0
    nnz_per_doc = np.diff(doc_matrix.indptr)
    total_activations = int(nnz_per_doc.sum())
    avg_activations = float(nnz_per_doc.mean()) if doc_texts else 0.0
    logger.info("SPLADE [%s]: encoded %d docs in %.2fs", short, len(doc_texts), index_time)

    # --- Encode queries ---
    t0 = time.time()
    query_matrix = _encode_to_sparse(
        model, tokenizer, query_texts, batch_size, device, max_length,
        desc=f"SPLADE [{short}] queries",
    )
    logger.info("SPLADE [%s]: encoded %d queries in %.2fs", short, len(query_ids), time.time() - t0)

    # Free GPU memory before retrieval
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Retrieve top-k via batched sparse dot product ---
    t0 = time.time()
    run_dict: Dict[str, Dict[str, float]] = {}
    doc_matrix_T = doc_matrix.T.tocsc()
    for q_start in range(0, len(query_ids), retrieve_q_batch):
        q_slice = query_matrix[q_start : q_start + retrieve_q_batch]
        scores_batch = (q_slice @ doc_matrix_T).toarray()
        for local_i, scores in enumerate(scores_batch):
            q_id = query_ids[q_start + local_i]
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            run_dict[q_id] = {doc_ids[i]: float(scores[i]) for i in top_idx if doc_ids[i] != q_id}
    retrieve_time = time.time() - t0
    logger.info("SPLADE [%s]: retrieved top-%d in %.2fs", short, top_k, retrieve_time)

    # --- Evaluate ---
    qrels_obj = Qrels(qrels=qrels)
    run_obj = Run(run=run_dict)
    evaluation_scores = ranx_evaluate(qrels=qrels_obj, run=run_obj, metrics=metrics, make_comparable=True)
    for metric, value in evaluation_scores.items():
        logger.info("  %s: %.4f", metric, value)

    result: Dict[str, Any] = {
        "config_idx": f"splade_{short}",
        "config_name": f"SPLADE ({model_name})",
        "config": {"type": "splade", "model_name_or_path": model_name, "max_length": max_length},
        "token_count": total_activations,
        "avg_tokens_per_doc": avg_activations,
        "compression_time": 0.0,
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "evaluation": evaluation_scores,
    }

    config_dir = versioned_dir(results_dir / f"config_splade_{short}")
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    if save_runfile:
        run_obj.save((config_dir / "runfile.json").as_posix())

    return result


def evaluate_splade(
    cfg,
    documents: List[Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    results_dir: Path,
    run_id: str,
) -> List[Dict[str, Any]]:
    """Run SPLADE retrieval for every configured model and evaluate with the same metrics as ColBERT."""
    model_cfgs = _resolve_splade_models(cfg.splade)
    doc_ids = [doc["id"] for doc in documents]
    doc_texts = [doc["text"] for doc in documents]
    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    results: List[Dict[str, Any]] = []
    for model_cfg in model_cfgs:
        model_name = model_cfg["model_name_or_path"]
        logger.info("")
        logger.info("-" * 60)
        logger.info("SPLADE model: %s", model_name)
        logger.info("-" * 60)
        try:
            results.append(_evaluate_one_splade(
                model_cfg=model_cfg,
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                query_ids=query_ids,
                query_texts=query_texts,
                qrels=qrels,
                metrics=list(cfg.metrics),
                top_k=cfg.retrieve.k,
                results_dir=results_dir,
                save_runfile=cfg.output.save_runfile,
            ))
        except Exception as exc:
            logger.warning("SPLADE [%s]: SKIPPED — %s: %s", model_name, type(exc).__name__, exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results

