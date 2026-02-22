#!/bin/bash

# These SBATCH lines are for running the whole loop as a single job (local mode)
# In --sbatch mode, we submit one Slurm job per dataset with its own SBATCH args.
#SBATCH --job-name=compression-eval
#SBATCH --partition=h100,nvl
#SBATCH --gpus=2
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/compression_eval_%j.out
#SBATCH --error=work_dirs/slurm/compression_eval_%j.err
#SBATCH --time=48:00:00
#SBATCH --exclude=h04

set -euo pipefail

# Resume 2 old compression eval runs in parallel, one per GPU.
# Each run picks up the missing configs.

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="./experiments/compression/"
PYLATE_DIR="./"

mkdir -p "${PYLATE_DIR}/work_dirs/resume_logs"

p1=results/compression_eval_old/20260217_000507
p2=results/compression_eval_old/20260217_023708

# GPU 0 — SciDocs (20260217_000507 was originally beir/scidocs)
CUDA_VISIBLE_DEVICES=0 python experiments/compression/compression_eval.py \
  dataset.name="beir/scidocs" \
  compression.resume=true \
  compression.resume_dir="$p1" \
  output.results_dir="results/compression_eval_old" \
  > work_dirs/resume_logs/p1.log 2>&1 &
pid0=$!
echo "[GPU 0] $p1  (PID $pid0)"

# GPU 1 — ArguAna (20260217_023708 was originally beir/arguana)
CUDA_VISIBLE_DEVICES=1 python experiments/compression/compression_eval.py \
  dataset.name="beir/arguana" \
  compression.resume=true \
  compression.resume_dir="$p2" \
  output.results_dir="results/compression_eval_old" \
  > work_dirs/resume_logs/p2.log 2>&1 &
pid1=$!
echo "[GPU 1] $p2     (PID $pid1)"


echo ""
echo "All 2 resume jobs launched. Logs in work_dirs/resume_logs/"
echo "Waiting for all to finish..."

# Wait and track failures
failed=0
for pid in $pid0 $pid1; do
  if ! wait "$pid"; then
    echo "✗ PID $pid failed"
    failed=$((failed + 1))
  fi
done

if [[ $failed -gt 0 ]]; then
  echo "✗ $failed job(s) failed. Check logs in work_dirs/resume_logs/"
  exit 1
else
  echo "✓ All 2 resume jobs completed successfully!"
fi