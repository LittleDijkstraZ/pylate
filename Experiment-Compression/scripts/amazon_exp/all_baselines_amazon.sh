#!/bin/bash
set -euo pipefail

# Run ALL baselines (BM25 + SPLADE + Dense) for the Amazon dataset in one shot.
# Each baseline is enabled/disabled and configured entirely through
# compression_eval_amazon.yaml — no model names are hardcoded here.
#
# Results land under:
#   results/compression_eval_amazon_iterative/<timestamp>/
#     config_bm25_okapi/
#     config_splade_splade-v3/
#     config_dense_all-mpnet-base-v2/
#     config_dense_NV-Embed-v2/   (if enabled in YAML)
#     ...
#
# Each baseline gets its own log file under work_dirs/amazon_logs/.
#
# Override the dataset path at runtime:
#   AMAZON_DATASET=/path/to/beir_format ./all_baselines_amazon.sh
#
# Override GPU at runtime:
#   CUDA_VISIBLE_DEVICES=1 ./all_baselines_amazon.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # Experiment-Compression/
cd "${EXP_DIR}"

mkdir -p work_dirs/amazon_logs

AMAZON_DATASET="${AMAZON_DATASET:-../../amazon_dataset/beir_format_full}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "========================================================"
echo "  Amazon baseline evaluation — all retrievers"
echo "  dataset : ${AMAZON_DATASET}"
echo "  GPU     : ${GPU}"
echo "  config  : compression_eval_amazon.yaml"
echo "========================================================"
echo ""

# ── 1. BM25 (CPU-only) ───────────────────────────────────────
LOG_BM25="work_dirs/amazon_logs/amazon_bm25.log"
echo "[1/3] BM25"
echo "      variants : (from yaml → bm25.variants)"
echo "      log      : ${LOG_BM25}"

python -m src.compression_eval_iterative.cli \
  --config-name compression_eval_amazon \
  dataset.name="${AMAZON_DATASET}" \
  bm25.enabled=true \
  splade.enabled=false \
  dense.enabled=false \
  "compression.indices=[]" \
  > "${LOG_BM25}" 2>&1

echo "      ✓ done"
echo ""

# ── 2. SPLADE (GPU) ──────────────────────────────────────────
LOG_SPLADE="work_dirs/amazon_logs/amazon_splade.log"
echo "[2/3] SPLADE"
echo "      models : (from yaml → splade.models)"
echo "      log    : ${LOG_SPLADE}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.compression_eval_iterative.cli \
  --config-name compression_eval_amazon \
  dataset.name="${AMAZON_DATASET}" \
  bm25.enabled=false \
  splade.enabled=true \
  dense.enabled=false \
  "compression.indices=[]" \
  > "${LOG_SPLADE}" 2>&1

echo "      ✓ done"
echo ""

# ── 3. Dense bi-encoders (GPU) ───────────────────────────────
LOG_DENSE="work_dirs/amazon_logs/amazon_dense.log"
echo "[3/3] Dense"
echo "      models : (from yaml → dense.models)"
echo "      log    : ${LOG_DENSE}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m src.compression_eval_iterative.cli \
  --config-name compression_eval_amazon \
  dataset.name="${AMAZON_DATASET}" \
  bm25.enabled=false \
  splade.enabled=false \
  dense.enabled=true \
  "compression.indices=[]" \
  > "${LOG_DENSE}" 2>&1

echo "      ✓ done"
echo ""

echo "========================================================"
echo "  All baselines complete."
echo "  Logs:"
echo "    ${LOG_BM25}"
echo "    ${LOG_SPLADE}"
echo "    ${LOG_DENSE}"
echo "========================================================"

