#!/bin/bash
set -euo pipefail

# Resume 4 old compression eval runs in parallel, one per GPU.
# Each run picks up the 7 missing keep_ratio=0.75 configs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYLATE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PYLATE_DIR"

mkdir -p work_dirs/resume_logs

# GPU 0 — TREC-COVID
CUDA_VISIBLE_DEVICES=0 python experiments/compression/compression_eval.py \
  dataset.name="beir/trec-covid" \
  compression.resume=true \
  compression.resume_dir="results/compression_eval/beir_trec-covid_20260205" \
  > work_dirs/resume_logs/resume_trec-covid.log 2>&1 &
pid0=$!
echo "[GPU 0] TREC-COVID  (PID $pid0)"

# GPU 1 — SciFact
CUDA_VISIBLE_DEVICES=1 python experiments/compression/compression_eval.py \
  dataset.name="beir/scifact/test" \
  compression.resume=true \
  compression.resume_dir="results/compression_eval/beir_scifact_test_20260205" \
  > work_dirs/resume_logs/resume_scifact.log 2>&1 &
pid1=$!
echo "[GPU 1] SciFact     (PID $pid1)"

# GPU 2 — NFCorpus
CUDA_VISIBLE_DEVICES=2 python experiments/compression/compression_eval.py \
  dataset.name="beir/nfcorpus/test" \
  compression.resume=true \
  compression.resume_dir="results/compression_eval/beir_nfcorpus_test_20260205" \
  > work_dirs/resume_logs/resume_nfcorpus.log 2>&1 &
pid2=$!
echo "[GPU 2] NFCorpus    (PID $pid2)"

# GPU 3 — FiQA
CUDA_VISIBLE_DEVICES=3 python experiments/compression/compression_eval.py \
  dataset.name="beir/fiqa/test" \
  compression.resume=true \
  compression.resume_dir="results/compression_eval/beir_fiqa_test_20260205" \
  > work_dirs/resume_logs/resume_fiqa.log 2>&1 &
pid3=$!
echo "[GPU 3] FiQA        (PID $pid3)"

echo ""
echo "All 4 resume jobs launched. Logs in work_dirs/resume_logs/"
echo "Waiting for all to finish..."

# Wait and track failures
failed=0
for pid in $pid0 $pid1 $pid2 $pid3; do
  if ! wait "$pid"; then
    echo "✗ PID $pid failed"
    failed=$((failed + 1))
  fi
done

if [[ $failed -gt 0 ]]; then
  echo "✗ $failed job(s) failed. Check logs in work_dirs/resume_logs/"
  exit 1
else
  echo "✓ All 4 resume jobs completed successfully!"
fi