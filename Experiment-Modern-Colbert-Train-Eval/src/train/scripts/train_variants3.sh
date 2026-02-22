#!/bin/bash

# Orchestrator: submit and monitor per-variant training jobs until all eval folders exist.
# This script is a plain bash script (no SBATCH header). It submits worker jobs via sbatch
# and monitors them, only (re)submitting when needed.

set -euo pipefail

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYLATE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # .../AI2AI/pylate
cd "${PYLATE_ROOT}"

# Resources (overridable via CLI flags)
PARTITION="h200"
GPUS=1
CPUS=16
MEM="80G"
TIME_LIMIT="48:00:00"
EXCLUDE=""
SLEEP_SEC=600
MAX_SUBMIT_ATTEMPTS=0   # 0 = unlimited

usage() {
  cat <<EOF
Usage: $0 [options] [-- EXTRA_ARGS]

Submit and monitor ColBERT variant training jobs until all have eval folders.

Options:
  --partition STR       Slurm partition (default: ${PARTITION})
  --gpus N              GPUs per job (default: ${GPUS})
  --cpus N              CPUs per job (default: ${CPUS})
  --mem STR             Memory per job (default: ${MEM})
  --time STR            Time limit (default: ${TIME_LIMIT})
  --exclude NODES       Comma-separated nodes to exclude (default: ${EXCLUDE})
  --sleep-sec N         Polling interval seconds (default: ${SLEEP_SEC})
  --max-submit N        Max submits per config (0 = unlimited, default: 0)
  -h, --help            Show this help

Any args after '--' are passed to the worker as EXTRA_ARGS.
EOF
}

# Parse CLI
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --time) TIME_LIMIT="$2"; shift 2;;
    --exclude) EXCLUDE="$2"; shift 2;;
    --sleep-sec) SLEEP_SEC="$2"; shift 2;;
    --max-submit) MAX_SUBMIT_ATTEMPTS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    --) shift; EXTRA_ARGS="$*"; break;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

mkdir -p logs

# Config variants to run
configs=(
  "gte_modern_colbert"
  "gte_modern_colbert model=proxy_attention model.variant_args.num_proxy_tokens=32 model.variant_args.num_select_tokens=32 compile=false"
  "gte_modern_colbert model=constbert model.variant_args.constbert_seq_length=32"
  "gte_modern_colbert model=memory_token model.variant_args.num_memory_tokens=32"
  "gte_modern_colbert model=proxy_attention model.variant_args.num_proxy_tokens=16 model.variant_args.num_select_tokens=16 compile=false"
  "gte_modern_colbert model=constbert model.variant_args.constbert_seq_length=16"
  "gte_modern_colbert model=memory_token model.variant_args.num_memory_tokens=16"
)

# Helper: resolve output_dir for a config
resolve_output_dir() {
  REPO_ROOT="${PYLATE_ROOT}" python - "$@" <<'PYCODE'
import os, sys

repo_root = os.environ["REPO_ROOT"]

from hydra import initialize_config_dir, compose

config_dir = os.path.join(repo_root, "conf2")

# Ensure repo root on sys.path for 'examples' imports
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

args = sys.argv[1:]
cfg_name = None
overrides = []
for tok in args:
    if cfg_name is None and "=" not in tok:
        cfg_name = tok
    else:
        overrides.append(tok)
if cfg_name is None:
    cfg_name = "gte_modern_colbert"

with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name=cfg_name, overrides=overrides)
# Compute run_name locally to avoid importing training module
variant_args = cfg.model.get("variant_args") or {}
base = f"{cfg.model.type}-lr-{cfg.lr}-bs{cfg.batch_size}nway{cfg.n_ways}"
wd = cfg.get("weight_decay")
if wd is not None:
    base = f"{base}-wd{wd}"
if cfg.model.type == "constbert":
    constbert_variant = variant_args["constbert_variant"]
    constbert_seq_length = variant_args["constbert_seq_length"]
    base = f"{base}-{constbert_variant}-C{constbert_seq_length}"
if cfg.model.type == "memory_token":
    num_memory_tokens = variant_args["num_memory_tokens"]
    base = f"{base}-M{num_memory_tokens}"
if cfg.model.type == "proxy_attention":
    num_proxy_tokens = variant_args["num_proxy_tokens"]
    num_select_tokens = variant_args["num_select_tokens"]
    base = f"{base}-P{num_proxy_tokens}-S{num_select_tokens}"
run_name = cfg.get("run_name") or base
print(os.path.join(repo_root, "output", run_name))
PYCODE
}

# State tracking
declare -A JOB_IDS=()           # idx -> jobid
declare -A SUBMIT_COUNTS=()     # idx -> count
declare -A DONE=()              # idx -> 1 if eval exists

# Submit a worker job for config index $1
submit_job() {
  local idx="$1"
  local cfg_str="${configs[$idx]}"
  local job_name="colbert-variant-${idx}"
  local sb_cmd=(sbatch
    --job-name="${job_name}"
    --chdir="${PYLATE_ROOT}"
    --partition="${PARTITION}"
    --gpus="${GPUS}"
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --time="${TIME_LIMIT}"
    --output="${PYLATE_ROOT}/logs/${job_name}_%j.out"
    --error="${PYLATE_ROOT}/logs/${job_name}_%j.err"
    --qos="h200_4"
  )
  if [[ -n "${EXCLUDE}" ]]; then
    sb_cmd+=(--exclude="${EXCLUDE}")
  fi
  # Always export PYLATE_ROOT so the worker can find the repo even under Slurm spool
  if [[ -n "${EXTRA_ARGS}" ]]; then
    sb_cmd+=(--export="ALL,EXTRA_ARGS=${EXTRA_ARGS},PYLATE_ROOT=${PYLATE_ROOT}")
  else
    sb_cmd+=(--export="ALL,PYLATE_ROOT=${PYLATE_ROOT}")
  fi
  sb_cmd+=("${SCRIPT_DIR}/train_variant_worker.sh")

  # Split cfg_str into tokens safely
  # shellcheck disable=SC2206
  local tokens=( ${cfg_str} )
  # Append tokens as args to worker
  for t in "${tokens[@]}"; do sb_cmd+=("${t}"); done

  local submit_out
  if ! submit_out="$("${sb_cmd[@]}" 2>&1)"; then
    echo "✗ Failed to submit job for index ${idx}: ${submit_out}" >&2
    return 1
  fi
  local job_id
  job_id="$(sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p' <<< "${submit_out}")"
  if [[ -z "${job_id}" ]]; then
    # Fallback: echo full output
    echo "⚠ Could not parse job id from sbatch output: ${submit_out}" >&2
    return 1
  fi
  JOB_IDS["${idx}"]="${job_id}"
  SUBMIT_COUNTS["${idx}"]=$(( ${SUBMIT_COUNTS["${idx}"]:-0} + 1 ))
  echo "✓ Submitted job ${job_id} for config index ${idx}"
}

is_job_active() {
  local job_id="$1"
  [[ -n "${job_id}" ]] || return 1
  local out
  out="$(squeue -j "${job_id}" -h 2>/dev/null || true)"
  [[ -n "${out}" ]]
}

all_done=false
while ! ${all_done}; do
  all_done=true
  for i in "${!configs[@]}"; do
    # Skip if already marked done
    if [[ "${DONE["$i"]:-}" == "1" ]]; then
      continue
    fi

    # Compute EVAL_DIR
    OUTPUT_DIR="$(resolve_output_dir ${configs[$i]})"
    EVAL_DIR="${OUTPUT_DIR}/eval"

    if [[ -d "${EVAL_DIR}" ]]; then
      echo "[index ${i}] ✓ Eval exists: ${EVAL_DIR}"
      DONE["$i"]=1
      continue
    fi

    all_done=false

    # If a job is running/pending, keep waiting
    job_id="${JOB_IDS["$i"]:-}"
    if is_job_active "${job_id}"; then
      echo "[index ${i}] Waiting (job ${job_id} active) ..."
      continue
    fi

    # No active job and eval missing: (re)submit if under max attempts (or unlimited)
    if [[ "${MAX_SUBMIT_ATTEMPTS}" -eq 0 || ${SUBMIT_COUNTS["$i"]:-0} -lt ${MAX_SUBMIT_ATTEMPTS} ]]; then
      echo "[index ${i}] Submitting worker (attempt $(( ${SUBMIT_COUNTS["$i"]:-0} + 1 ))) ..."
      submit_job "$i" || true
    else
      echo "[index ${i}] Reached max submit attempts (${MAX_SUBMIT_ATTEMPTS}); not resubmitting."
    fi
  done

  ${all_done} && break
  echo "Sleeping ${SLEEP_SEC}s before next poll..."
  sleep "${SLEEP_SEC}"
done

echo "All configs have eval folders. Done."
