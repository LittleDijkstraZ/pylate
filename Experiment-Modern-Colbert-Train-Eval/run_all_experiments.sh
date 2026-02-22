#!/bin/bash
# Master script to run all compression experiments
# Uses GPUs 1,2,3 (skips GPU 0 which is in use by another program)

echo "======================================"
echo "Compression Experiments Master Script"
echo "======================================"
echo ""
echo "This will run the following experiments:"
echo "  1. Training-free compression (GTE-ModernColBERT-v1)"
echo "  2. ConstBERT evaluation (checkpoint-15000)"
echo "  3. ProxyAttention evaluation (checkpoint-15000)"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Kill any existing experiments
echo ""
echo "Killing any existing compression experiments..."
pkill -9 -f compression_experiment 2>/dev/null || true
sleep 2

# Run experiments sequentially
echo ""
echo "======================================" 
echo "Step 1/3: Training-free Compression"
echo "======================================"
# bash run_training_free_compression.sh

echo ""
echo "======================================"
echo "Step 2/3: ConstBERT Evaluation"
echo "======================================"
bash run_constbert_eval.sh

echo ""
echo "======================================"
echo "Step 3/3: ProxyAttention Evaluation"  
echo "======================================"
bash run_proxy_attention_eval.sh

echo ""
echo "======================================"
echo "All experiments completed!"
echo "======================================"
echo ""
echo "Results are saved in:"
echo "  - results/amazon_compression_comparison_20260209/training_free/"
echo "  - results/amazon_compression_comparison_20260209/ConstBERT-transpose-C32-ckpt15000/"
echo "  - results/amazon_compression_comparison_20260209/ProxyAttention-P32-ckpt15000/"

