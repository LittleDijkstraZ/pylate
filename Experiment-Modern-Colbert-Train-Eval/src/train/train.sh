#!/bin/sh

#SBATCH --job-name=colbert-finetune
#SBATCH --partition=h200
#SBATCH --gpus=2
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/finetune_%j.out
#SBATCH --error=work_dirs/slurm/finetune_%j.err
#SBATCH --time=48:00:00
#SBATCH --qos=h200_4


nvidia-smi

torchrun --nproc_per_node=2 examples/train/train_colbert_config.py \
    --config examples/train/configs/proxy_attention_config.json

