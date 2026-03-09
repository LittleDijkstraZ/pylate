#!/bin/bash
set -euo pipefail

# BM25 baseline evaluation for the Amazon dataset.
# Runs only the BM25 baseline (no ColBERT encoding or PLAID indexing).
# Variants are taken from bm25.variants in compression_eval_amazon.yaml.
# Results are saved under results/compression_eval_amazon_iterative/<timestamp>/config_bm25_<variant>/.
#
# BM25 is CPU-only — no GPU required.
#
# Override the dataset path at runtime via env var, e.g.:
#   AMAZON_DATASET=../amazon_dataset/beir_format ./bm25_eval_amazon.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # Experiment-Compression/
cd "${EXP_DIR}"

mkdir -p work_dirs/amazon_logs

AMAZON_DATASET="${AMAZON_DATASET:-../../amazon_dataset/beir_format_full}"

LOG_FILE="work_dirs/amazon_logs/amazon_bm25.log"

echo "[BM25] Amazon full content"
echo "       dataset  : ${AMAZON_DATASET}"
echo "       variants : (from compression_eval_amazon.yaml → bm25.variants)"
echo "       results  : results/compression_eval_amazon_iterative/<timestamp>/config_bm25_<variant>/"
echo "       log      : ${LOG_FILE}"
echo ""

# compression.indices=[] skips all ColBERT configs so only BM25 runs.
# bm25.enabled=true is already the default in compression_eval_amazon.yaml.
python -m src.compression_eval_iterative.cli \
  --config-name compression_eval_amazon \
  dataset.name="${AMAZON_DATASET}" \
  bm25.enabled=true \
  "compression.indices=[]" \
  > "${LOG_FILE}" 2>&1

echo "✓ BM25 eval completed. Log: ${LOG_FILE}"

