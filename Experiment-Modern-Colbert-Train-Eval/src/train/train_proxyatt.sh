#!/bin/sh

#SBATCH --job-name=colbert-finetune
#SBATCH --partition=nvl,h100
#SBATCH --gpus=2
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/finetune_%j.out
#SBATCH --error=work_dirs/slurm/finetune_%j.err
#SBATCH --time=48:00:00
#SBATCH --exclude=n07

nvidia-smi

torchrun --nproc_per_node=2 ./examples/train/proxy_attention_colbert.py

