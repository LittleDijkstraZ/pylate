#!/bin/bash
set -euo pipefail

# SPLADE baseline evaluation for the Amazon dataset.
# Runs only SPLADE (no ColBERT encoding or PLAID indexing).
# Models are taken from splade.models in compression_eval_amazon.yaml.
# Results are saved under results/compression_eval_amazon_iterative/<timestamp>/config_splade_<model>/.
#
# Override the dataset path at runtime:
#   AMAZON_DATASET=../amazon_dataset/beir_format ./splade_eval_amazon.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # Experiment-Compression/
cd "${EXP_DIR}"

mkdir -p work_dirs/amazon_logs

AMAZON_DATASET="${AMAZON_DATASET:-../../amazon_dataset/beir_format_full}"

LOG_FILE="work_dirs/amazon_logs/amazon_splade.log"

echo "[SPLADE] Amazon full content"
echo "         dataset : ${AMAZON_DATASET}"
echo "         models  : (from compression_eval_amazon.yaml → splade.models)"
echo "         results : results/compression_eval_amazon_iterative/<timestamp>/config_splade_<model>/"
echo "         log     : ${LOG_FILE}"
echo ""

# compression.indices=[] skips all ColBERT configs so only SPLADE runs.
# bm25.enabled=false disables the BM25 block that is on by default in this config.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m src.compression_eval_iterative.cli \
  --config-name compression_eval_amazon \
  dataset.name="${AMAZON_DATASET}" \
  bm25.enabled=false \
  splade.enabled=true \
  "compression.indices=[]" \
  > "${LOG_FILE}" 2>&1

echo "✓ SPLADE eval completed. Log: ${LOG_FILE}"

