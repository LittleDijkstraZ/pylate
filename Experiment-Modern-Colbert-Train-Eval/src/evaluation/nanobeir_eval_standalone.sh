# python ./examples/evaluation/nanobeir_eval_standalone.py \
#     --only "ProxyAttention-Base-32tok-5000-bs24" \
#         "ColBERT-Base-300tok-5000-HPool32" \
#         "ColBERT-Base-300tok-5000-HPool32-h2pool" \
#         "ColBERT-Base-300tok-5000" \
#     --datasets-scifact-nfcorpus \
#     --output "model_comparison_scifact_nfcorpus_short.csv" \
#     --batch-size 64 \
#     --show-progress

python ./examples/evaluation/nanobeir_eval_standalone.py \
    --only "ProxyAttention-Base-32tok-10000-bs24" \
        "ColBERT-Base-300tok-10000-HPool32-h2pool" \
        "ColBERT-Base-300tok-10000" \
    --datasets-scifact-nfcorpus \
    --output "model_comparison_scifact_nfcorpus_clean2_10000.csv" \
    --batch-size 64 \
    --show-progress

# python ./examples/evaluation/nanobeir_eval_standalone.py \
#     --only "ColBERT-Base-300tok-5000-HPool32" \
#     --output "nanobeir_model_comparison_new.csv" \
#     --merge \
#     --show-progress