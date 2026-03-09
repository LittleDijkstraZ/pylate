"""Single-vector dense (bi-encoder) baseline evaluation for the iterative compression pipeline.

Encodes documents and queries using a sentence-transformers-compatible model (e.g. GTE,
BGE, E5, all-MiniLM, NV-Embed, …) producing one dense vector per text.  Retrieval is
exact nearest-neighbour search via batched matrix multiply on GPU (or CPU).

Two encoding paths are supported automatically:

* **Standard path** (mean-pool over token embeddings): used when the model does NOT
  expose a custom ``.encode()`` method.  A ``query_prefix`` / ``passage_prefix`` string
  is prepended directly to each text before tokenisation (E5-instruct style).

* **model.encode() path**: used when the loaded model exposes a callable ``.encode()``
  (e.g. ``nvidia/NV-Embed-v2``).  The prefix is forwarded as ``instruction=`` kwarg so
  the model can handle it internally (no double-prepending).

Multiple models can be evaluated in one run by listing them under ``dense.models``
in the Hydra config; every entry can override any shared parameter including the
instruction prefixes (see compression_eval.yaml for examples).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate

from .utils import versioned_dir

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Alibaba-NLP/gte-base-en-v1.5"

# ---------------------------------------------------------------------------
# Compatibility patch: nvidia/NV-Embed-v2 calls DynamicCache.get_usable_length()
# which was removed in transformers >= 4.45.  Restore it so the model works
# with newer transformers without downgrading.
# ---------------------------------------------------------------------------
try:
    from transformers.cache_utils import DynamicCache as _DynamicCache  # noqa: F401
    if not hasattr(_DynamicCache, "get_usable_length"):
        _DynamicCache.get_usable_length = (
            lambda self, new_seq_len, layer_idx=0: self.get_seq_length(layer_idx)
        )
except Exception:  # pragma: no cover – best-effort patch
    pass


def _short_name(model_name: str) -> str:
    """Filesystem-safe short label: last path component, dots → dashes."""
    return model_name.split("/")[-1].replace(".", "-")


def _resolve_dense_models(dense_cfg) -> List[Dict[str, Any]]:
    """Return a list of per-model config dicts, merging shared defaults with per-model overrides.

    New format (preferred)::

        dense:
          batch_size: 128
          models:
            - model_name_or_path: Alibaba-NLP/gte-base-en-v1.5
            - model_name_or_path: intfloat/e5-base-v2
              max_length: 256

    Legacy format (still supported)::

        dense:
          model_name_or_path: Alibaba-NLP/gte-base-en-v1.5
          max_length: 512
    """
    shared = {
        "batch_size": int(dense_cfg.get("batch_size", 128)),
        "max_length": int(dense_cfg.get("max_length", 512)),
        "normalize_embeddings": bool(dense_cfg.get("normalize_embeddings", True)),
        "retrieve_query_batch_size": int(dense_cfg.get("retrieve_query_batch_size", 256)),
        "device": dense_cfg.get("device", None),
        # Instruction / prefix strings (empty string = no prefix)
        "query_prefix": str(dense_cfg.get("query_prefix", "")),
        "passage_prefix": str(dense_cfg.get("passage_prefix", "")),
        # Pooling strategy: "mean" (encoder models), "last_token" (decoder/causal-LM models
        # such as e5-mistral), or "cls" (first token / pooler_output).
        "pooling_strategy": str(dense_cfg.get("pooling_strategy", "mean")),
    }
    models_list = dense_cfg.get("models", None)
    if models_list is not None:
        result = []
        for m in models_list:
            merged = dict(shared)
            merged.update(OmegaConf.to_container(m, resolve=True))
            result.append(merged)
        return result
    # Legacy: single model_name_or_path at top level
    return [{**shared, "model_name_or_path": dense_cfg.get("model_name_or_path", _DEFAULT_MODEL)}]


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding."""
    mask_expanded = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the last non-padding token embedding (for decoder/causal-LM models, e.g. e5-mistral).

    Works correctly regardless of whether the tokenizer uses left- or right-padding.
    """
    # Left-padded: all positions in the last column are non-padding → take index -1
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]
    # Right-padded: last non-padding position differs per sample
    seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), seq_lens]


def _encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    device: torch.device,
    max_length: int,
    normalize: bool,
    prefix: str = "",
    pooling_strategy: str = "mean",
    desc: str = "Encoding",
) -> torch.Tensor:
    """Encode *texts* to a (N, dim) float32 CPU tensor with optional L2 norm.

    If *prefix* is non-empty it is prepended to every text before tokenisation
    (E5-instruct / BGE-instruct style).

    *pooling_strategy* controls how token embeddings are aggregated:
      - ``"mean"``       — mean-pool over non-padding tokens (default; most encoder models)
      - ``"cls"``        — pooler_output if available, else first-token hidden state
      - ``"last_token"`` — last non-padding token (decoder/causal-LM models, e.g. e5-mistral)
    """
    all_vecs: List[torch.Tensor] = []
    prepared = [prefix + t for t in texts] if prefix else texts

    for start in tqdm(range(0, len(prepared), batch_size), desc=desc, unit="batch"):
        batch = prepared[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**inputs)

        hs = out.last_hidden_state.float()
        mask = inputs["attention_mask"]

        if pooling_strategy == "last_token":
            vecs = _last_token_pool(hs, mask)
        elif pooling_strategy == "cls":
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vecs = out.pooler_output.float()
            else:
                vecs = hs[:, 0]
        else:  # "mean" (default)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vecs = out.pooler_output.float()
            else:
                vecs = _mean_pool(hs, mask)

        if normalize:
            vecs = F.normalize(vecs, p=2, dim=-1)

        all_vecs.append(vecs.cpu())

    return torch.cat(all_vecs, dim=0) if all_vecs else torch.empty(0)


def _encode_with_model_encode(
    model,
    texts: List[str],
    instruction: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
    desc: str = "Encoding",
) -> torch.Tensor:
    """Encode *texts* using the model's own ``.encode()`` method (e.g. NV-Embed-v2).

    The *instruction* is forwarded as the ``instruction=`` keyword so the model can
    handle prefix formatting internally.  The result is returned as a (N, dim) float32
    CPU tensor.
    """
    embeddings = model.encode(
        texts,
        instruction=instruction,
        max_length=max_length,
    )
    # model.encode may return numpy or torch; normalise to a CPU float32 tensor
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    else:
        embeddings = embeddings.float().cpu()

    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    return embeddings


def _evaluate_one_dense(
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
    """Encode, retrieve, and evaluate for a single dense bi-encoder model config."""
    model_name = model_cfg["model_name_or_path"]
    batch_size = int(model_cfg.get("batch_size", 128))
    max_length = int(model_cfg.get("max_length", 512))
    normalize = bool(model_cfg.get("normalize_embeddings", True))
    retrieve_q_batch = int(model_cfg.get("retrieve_query_batch_size", 256))
    device_cfg = model_cfg.get("device", None)
    device = torch.device(device_cfg) if device_cfg else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_prefix = str(model_cfg.get("query_prefix", ""))
    passage_prefix = str(model_cfg.get("passage_prefix", ""))
    pooling_strategy = str(model_cfg.get("pooling_strategy", "mean"))
    short = _short_name(model_name)

    logger.info("Dense [%s]: loading on %s", short, device)
    logger.info("Dense [%s]: pooling_strategy = %s", short, pooling_strategy)
    if query_prefix:
        logger.info("Dense [%s]: query_prefix = %r", short, query_prefix[:80])
    if passage_prefix:
        logger.info("Dense [%s]: passage_prefix = %r", short, passage_prefix[:80])

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    logger.info("Dense [%s]: loaded in %.2fs", short, time.time() - t0)

    # Auto-detect encoding path: use model.encode() if the model exposes it (e.g. NV-Embed-v2),
    # otherwise fall back to our tokeniser + pooling path.
    use_model_encode = callable(getattr(model, "encode", None))
    if use_model_encode:
        logger.info("Dense [%s]: using model.encode() path (instruction-aware)", short)
    else:
        logger.info("Dense [%s]: using standard %s-pooling path", short, pooling_strategy)

    # --- Encode documents ---
    t0 = time.time()
    if use_model_encode:
        doc_embeddings = _encode_with_model_encode(
            model, doc_texts, instruction=passage_prefix,
            batch_size=batch_size, max_length=max_length, normalize=normalize,
            desc=f"Dense [{short}] docs",
        )
    else:
        doc_embeddings = _encode_texts(
            model, tokenizer, doc_texts, batch_size, device, max_length, normalize,
            prefix=passage_prefix, pooling_strategy=pooling_strategy,
            desc=f"Dense [{short}] docs",
        )
    index_time = time.time() - t0
    embed_dim = doc_embeddings.shape[1] if doc_embeddings.ndim == 2 else 0
    logger.info("Dense [%s]: encoded %d docs in %.2fs  |  dim=%d", short, len(doc_texts), index_time, embed_dim)

    # --- Encode queries ---
    t0 = time.time()
    if use_model_encode:
        query_embeddings = _encode_with_model_encode(
            model, query_texts, instruction=query_prefix,
            batch_size=batch_size, max_length=max_length, normalize=normalize,
            desc=f"Dense [{short}] queries",
        )
    else:
        query_embeddings = _encode_texts(
            model, tokenizer, query_texts, batch_size, device, max_length, normalize,
            prefix=query_prefix, pooling_strategy=pooling_strategy,
            desc=f"Dense [{short}] queries",
        )
    logger.info("Dense [%s]: encoded %d queries in %.2fs", short, len(query_ids), time.time() - t0)

    # Free GPU memory before retrieval
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Retrieve top-k via batched matmul ---
    t0 = time.time()
    run_dict: Dict[str, Dict[str, float]] = {}
    doc_embeddings_dev = doc_embeddings.to(device)

    for q_start in range(0, len(query_ids), retrieve_q_batch):
        q_slice = query_embeddings[q_start : q_start + retrieve_q_batch].to(device)
        scores_batch = torch.mm(q_slice, doc_embeddings_dev.T)  # (batch, N)
        for local_i, scores in enumerate(scores_batch):
            q_id = query_ids[q_start + local_i]
            scores_np = scores.cpu().float().numpy()
            top_idx = np.argpartition(scores_np, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores_np[top_idx])[::-1]]
            run_dict[q_id] = {doc_ids[i]: float(scores_np[i]) for i in top_idx if doc_ids[i] != q_id}

    retrieve_time = time.time() - t0
    logger.info("Dense [%s]: retrieved top-%d in %.2fs", short, top_k, retrieve_time)

    # --- Evaluate ---
    qrels_obj = Qrels(qrels=qrels)
    run_obj = Run(run=run_dict)
    evaluation_scores = ranx_evaluate(qrels=qrels_obj, run=run_obj, metrics=metrics, make_comparable=True)
    for metric, value in evaluation_scores.items():
        logger.info("  %s: %.4f", metric, value)

    total_scalars = int(doc_embeddings.numel())
    avg_scalars_per_doc = float(embed_dim)

    result: Dict[str, Any] = {
        "config_idx": f"dense_{short}",
        "config_name": f"Dense ({model_name})",
        "config": {
            "type": "dense",
            "model_name_or_path": model_name,
            "max_length": max_length,
            "normalize_embeddings": normalize,
            "embed_dim": embed_dim,
            "query_prefix": query_prefix,
            "passage_prefix": passage_prefix,
            "encoding_path": "model.encode" if use_model_encode else "mean-pool",
        },
        "token_count": total_scalars,
        "avg_tokens_per_doc": avg_scalars_per_doc,
        "compression_time": 0.0,
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "evaluation": evaluation_scores,
    }

    config_dir = versioned_dir(results_dir / f"config_dense_{short}")
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    if save_runfile:
        run_obj.save((config_dir / "runfile.json").as_posix())

    return result


def evaluate_dense(
    cfg,
    documents: List[Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    results_dir: Path,
    run_id: str,
) -> List[Dict[str, Any]]:
    """Run dense retrieval for every configured model and evaluate with the same metrics as ColBERT."""
    model_cfgs = _resolve_dense_models(cfg.dense)
    doc_ids = [doc["id"] for doc in documents]
    doc_texts = [doc["text"] for doc in documents]
    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    results: List[Dict[str, Any]] = []
    for model_cfg in model_cfgs:
        model_name = model_cfg["model_name_or_path"]
        logger.info("")
        logger.info("-" * 60)
        logger.info("Dense model: %s", model_name)
        logger.info("-" * 60)
        try:
            results.append(_evaluate_one_dense(
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
            logger.warning("Dense [%s]: SKIPPED — %s: %s", model_name, type(exc).__name__, exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results

