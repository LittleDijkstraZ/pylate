#!/bin/bash
# ConstBERT Evaluation Script
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
mkdir -p results/amazon_compression_comparison_20260209/ConstBERT-transpose-C32-ckpt15000
mkdir -p logs

echo "======================================"
echo "Starting ConstBERT Evaluation"
echo "Model: constbert-lr-3e-05-bs24nway16-wd1e-06-transpose-C32/checkpoint-15000"
echo "Using GPUs: 1, 2, 3 (via CUDA_VISIBLE_DEVICES)"
echo "Batch size: 1024"
echo "======================================"

# Run ConstBERT evaluation
# Note: with CUDA_VISIBLE_DEVICES=1,2,3, the GPUs will appear as 0,1,2 to the script
# So we use --num_gpus 3
python -u experiments/compression/compression_experiment.py \
    --model_name "output/constbert-lr-3e-05-bs24nway16-wd1e-06-transpose-C32/checkpoint-15000" \
    --dataset_name "../amazon_dataset/beir_format_full" \
    --batch_size 1024 \
    --multi_gpu \
    --num_gpus 3 \
    --save_retrieval_results \
    --experiment_output_dir "results/amazon_compression_comparison_20260209/ConstBERT-transpose-C32-ckpt15000" \
    2>&1 | tee logs/constbert_eval.log

echo "======================================"
echo "ConstBERT Evaluation Complete!"
echo "Results saved to: results/amazon_compression_comparison_20260209/ConstBERT-transpose-C32-ckpt15000"
echo "======================================"

