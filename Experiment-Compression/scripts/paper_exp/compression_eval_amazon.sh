#!/bin/bash
set -euo pipefail

# Compression evaluation for the Amazon custom dataset.
# Runs a single job on GPU 0 using the full-content BEIR-format variant.
# Results are saved to results/compression_eval_amazon/<timestamp>/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYLATE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PYLATE_DIR"

mkdir -p work_dirs/amazon_logs

# Path to the Amazon dataset (local BEIR format, full content).
# Override via env var to switch to the summary variant, e.g.:
#   AMAZON_DATASET=../amazon_dataset/beir_format ./compression_eval_amazon.sh
AMAZON_DATASET="${AMAZON_DATASET:-../amazon_dataset/beir_format_full}"

# GPU 0 — Amazon (full content)
# --config-name selects conf/compression_eval_amazon.yaml which already sets the
# dataset path, configs file (attention/spherical/random pooling only), and output dir.
# The dataset.name override below lets the AMAZON_DATASET env var take precedence.
CUDA_VISIBLE_DEVICES=0 python experiments/compression/compression_eval.py \
  --config-name=compression_eval_amazon \
  dataset.name="${AMAZON_DATASET}" \
  > work_dirs/amazon_logs/amazon_full.log 2>&1 &
pid0=$!
echo "[GPU 0] Amazon full content  (PID $pid0)"
echo "        dataset : ${AMAZON_DATASET}"
echo "        results : results/compression_eval_amazon/<timestamp>"
echo "        log     : work_dirs/amazon_logs/amazon_full.log"

echo ""
echo "Amazon eval job launched. Log in work_dirs/amazon_logs/"
echo "Waiting for job to finish..."

if ! wait "$pid0"; then
  echo "✗ Amazon eval job failed. Check work_dirs/amazon_logs/amazon_full.log"
  exit 1
fi

echo "✓ Amazon eval job completed successfully!"

