#!/bin/bash
#SBATCH --job-name=h3_cv
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/home/ttdo/h3_analyses_cv_out.txt
#SBATCH --error=/home/ttdo/h3_analyses_cv_err.txt

# H3 Remaining Analyses (WITH CROSS-VALIDATION)
# CPU-only (no GPU needed for this analysis)
# Uses tesla2 which is fully idle
#
# FIXES: H_jac2 now uses 5-fold CV to avoid circular analysis
# Also includes random baseline and permutation test

echo "=== H3 Analyses Job Started ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Environment setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniver_env
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

cd ~/maniver/ManiVer

# Data and output directories
DATA_DIR=~/maniver/ManiVer/data/trajectories_0shot
RESULTS_DIR=~/maniver/ManiVer/results/h3_remaining

mkdir -p $RESULTS_DIR

echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS_DIR"
echo ""

# Check data exists
echo "=== Checking data files ==="
ls -lh $DATA_DIR/*/gsm8k*.h5 2>/dev/null || echo "No gsm8k files"
ls -lh $DATA_DIR/*/humaneval*.h5 2>/dev/null || echo "No humaneval files"
ls -lh $DATA_DIR/*/logiqa*.h5 2>/dev/null || echo "No logiqa files"
echo ""

# Run analysis (with cross-validation for H_jac2)
echo "=== Running H3 Analyses (CV for H_jac2) ==="
python scripts/analysis/h3_remaining_analyses.py \
    --data-dir $DATA_DIR \
    --output-dir $RESULTS_DIR \
    --models olmo3_base,olmo3_sft,olmo3_rl_zero,olmo3_think \
    --tasks gsm8k,humaneval,logiqa \
    --max-samples 200

echo ""
echo "=== Job Complete ==="
echo "Results in: $RESULTS_DIR"
ls -la $RESULTS_DIR/
echo ""
echo "Date: $(date)"
