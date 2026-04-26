#!/bin/bash
#SBATCH --job-name=belief_track
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/ttdo/belief_tracking_out.txt
#SBATCH --error=/home/ttdo/belief_tracking_err.txt

# Belief State Tracking Analysis - Phase 1
# CPU-only (no GPU needed - just probe training and analysis)
# Uses tesla2 (available for CPU jobs)
#
# Tests H_belief: RLVR = smooth belief evolution, SFT = discrete jumps
# Tests cross-model transfer: does belief probe generalize across models?

echo "=== Belief Tracking Analysis Started ==="
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
RESULTS_DIR=~/maniver/ManiVer/experiments/belief_tracking/results

mkdir -p $RESULTS_DIR

echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS_DIR"
echo ""

# Check data exists
echo "=== Checking data files ==="
for MODEL in olmo3_rl_zero olmo3_think olmo3_sft; do
    echo "Checking $MODEL..."
    ls -lh $DATA_DIR/$MODEL/gsm8k*.h5 2>/dev/null || echo "  No gsm8k files"
done
echo ""

# Run belief dynamics analysis
echo "=== Running Belief Dynamics Analysis ==="
python experiments/belief_tracking/analyze_belief_dynamics.py \
    --data-dir $DATA_DIR \
    --output-dir $RESULTS_DIR \
    --models olmo3_rl_zero,olmo3_think,olmo3_sft \
    --task gsm8k \
    --max-samples 200

echo ""
echo "=== Analysis Complete ==="
echo "Results in: $RESULTS_DIR"
ls -la $RESULTS_DIR/
echo ""
echo "Date: $(date)"
