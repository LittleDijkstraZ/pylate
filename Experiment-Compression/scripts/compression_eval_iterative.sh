#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/Experiment-Compression"

cd "${EXP_DIR}"

LOG_DIR="${ROOT_DIR}/work_dirs/amazon_logs"
mkdir -p "${LOG_DIR}"

# Path to the Amazon dataset (local BEIR format, full content).
# Override via env var to switch to the summary variant, e.g.:
#   AMAZON_DATASET=/path/to/amazon_dataset/beir_format ./compression_eval_iterative.sh
AMAZON_DATASET="${AMAZON_DATASET:-${ROOT_DIR}/amazon_dataset/beir_format_full}"

# Optional compression config list (JSONL). If missing, defaults are used.
CONFIGS_FILE="${CONFIGS_FILE:-${EXP_DIR}/compression/conf/amazon_compression_configs.jsonl}"
CONFIG_ARGS=()
if [[ -f "${CONFIGS_FILE}" ]]; then
  CONFIG_ARGS+=("compression.configs_file=${CONFIGS_FILE}")
fi

# GPU 0 — Amazon (full content)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python compression/compression_eval_iterative.py \
  dataset.name="${AMAZON_DATASET}" \
  model.dtype=bf16 \
  cache.embedding_dtype=fp32 \
  encode.batch_size=512 \
  encode.shard_size=50000 \
  index.nbits=4 \
  index.fallback_on_oom=true \
  output.results_dir="results/compression_eval_amazon_iterative" \
  "${CONFIG_ARGS[@]}" \
  > "${LOG_DIR}/amazon_iterative.log" 2>&1 &

pid0=$!
echo "[GPU 0] Amazon full content (iterative)  (PID ${pid0})"
echo "        dataset : ${AMAZON_DATASET}"
echo "        results : results/compression_eval_amazon_iterative/<timestamp>"
echo "        log     : ${LOG_DIR}/amazon_iterative.log"

echo ""
echo "Amazon iterative eval job launched. Log in ${LOG_DIR}/"
echo "Waiting for job to finish..."

if ! wait "${pid0}"; then
  echo "✗ Amazon iterative eval job failed. Check ${LOG_DIR}/amazon_iterative.log"
  exit 1
fi

echo "✓ Amazon iterative eval job completed successfully!"
