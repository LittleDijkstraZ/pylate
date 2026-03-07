#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # pylate/
AI2AI_DIR="$(cd "${ROOT_DIR}/.." && pwd)"       # AI2AI/ (parent of pylate)
EXP_DIR="${ROOT_DIR}/Experiment-Compression"

cd "${EXP_DIR}"

LOG_DIR="${ROOT_DIR}/work_dirs/amazon_logs"
mkdir -p "${LOG_DIR}"

# Path to the Amazon dataset (local BEIR format, full content).
# Override via env var to switch to the summary variant, e.g.:
#   AMAZON_DATASET=/path/to/amazon_dataset/beir_format ./compression_eval_iterative.sh
AMAZON_DATASET="${AMAZON_DATASET:-${AI2AI_DIR}/amazon_dataset/beir_format_full}"

# Optional: override the compression configs JSONL at runtime.
# Defaults to the one embedded in compression_eval_amazon.yaml.
CONFIG_ARGS=()
if [[ -n "${CONFIGS_FILE:-}" && -f "${CONFIGS_FILE}" ]]; then
  CONFIG_ARGS+=("compression.configs_file=${CONFIGS_FILE}")
fi

# GPU 0 — Amazon (full content)
# Uses refactored modular compression_eval_iterative package.
# Config: src/conf/compression_eval_amazon.yaml
# Reproduces results from results/compression_eval_amazon/20260220_235637/
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m src.compression_eval_iterative.cli \
  --config-name compression_eval_amazon \
  dataset.name="${AMAZON_DATASET}" \
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
