#!/bin/bash
# ProxyAttention Evaluation Script
# Uses GPUs 1,2,3 (skips GPU 0 which is in use by another program)
# Uses smaller batch size for stability

# Kill any existing compression experiments first
pkill -9 -f compression_experiment 2>/dev/null || true
sleep 2

# Set visible GPUs to 1,2,3 (skips GPU 0)
export CUDA_VISIBLE_DEVICES=1,2,3

# Activate conda environment
source ~/.bashrc
conda activate pl

# Create output directory
mkdir -p results/amazon_compression_comparison_20260209/ProxyAttention-P32-ckpt15000
mkdir -p logs

echo "======================================"
echo "Starting ProxyAttention Evaluation"
echo "Model: proxy_attention-lr-3e-05-bs24nway16-wd1e-06-P32-S32/checkpoint-15000"
echo "Using GPUs: 1, 2, 3 (via CUDA_VISIBLE_DEVICES)"
echo "Batch size: 1024"
echo "num_select_tokens: 27, 54, 89, 135 (matching training-free keep_ratio=0.1, 0.2, 0.33, 0.5)"
echo "======================================"

# Run ProxyAttention evaluation
# Note: with CUDA_VISIBLE_DEVICES=1,2,3, the GPUs will appear as 0,1,2 to the script
# So we use --num_gpus 3
# num_select_tokens values match training-free compression ratios:
#   27 tokens  ≈ keep_ratio=0.1  (0.1 × 270.5 baseline)
#   54 tokens  ≈ keep_ratio=0.2  (0.2 × 270.5 baseline)
#   89 tokens  ≈ keep_ratio=0.33 (0.33 × 270.5 baseline)
#   135 tokens ≈ keep_ratio=0.5  (0.5 × 270.5 baseline)
python -u experiments/compression/compression_experiment.py \
    --model_name "output/proxy_attention-lr-3e-05-bs24nway16-wd1e-06-P32-S32/checkpoint-15000" \
    --dataset_name "../amazon_dataset/beir_format_full" \
    --batch_size 1024 \
    --multi_gpu \
    --num_gpus 3 \
    --num_select_tokens "27,54,89,135" \
    --save_retrieval_results \
    --experiment_output_dir "results/amazon_compression_comparison_20260209/ProxyAttention-P32-ckpt15000" \
    2>&1 | tee logs/proxy_attention_eval.log

echo "======================================"
echo "ProxyAttention Evaluation Complete!"
echo "Results saved to: results/amazon_compression_comparison_20260209/ProxyAttention-P32-ckpt15000"
echo "======================================"

