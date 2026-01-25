#!/bin/bash
#SBATCH --job-name=cross_dom
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/home/ttdo/cross_domain_out.txt
#SBATCH --error=/home/ttdo/cross_domain_err.txt

# Cross-Domain Subspace Alignment for All Models
# Tests GSM8K ↔ LogiQA error direction alignment

echo "=== Cross-Domain Analysis Job Started ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniver_env
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

cd ~/maniver/ManiVer

DATA_DIR=~/maniver/ManiVer/data/trajectories_0shot
RESULTS_DIR=~/maniver/ManiVer/results/cross_domain

mkdir -p $RESULTS_DIR

python scripts/analysis/cross_domain_all_models.py \
    --data-dir $DATA_DIR \
    --output-dir $RESULTS_DIR \
    --models olmo3_rl_zero,olmo3_think \
    --max-samples 300

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"
