#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --output=log.out
# export PATH=/vol/bitbucket/trm25/myvenv/bin/:$PATH

source /vol/gpudata/trm25

echo "Running on host: $(hostname)"
echo "GPU in use:"
/usr/bin/nvidia-smi

python ~/Documents/dgl/cw2/stp_gsr/main.py 

echo "Training finished. Uptime:"
uptime