#!/bin/bash
#SBATCH --job-name=path_jac
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/home/ttdo/maniver/ManiVer/results/path_jac_%j.out

# ============================================================================
# Path Signatures + True Empirical Jacobian Analysis
# ============================================================================
# Runs two analyses:
# 1. Path signatures (reparameterization-invariant trajectory features)
# 2. True empirical Jacobian (regression-based, not delta-based)
#
# Models: olmo3_base, olmo3_sft, olmo3_rl_zero
# Tasks: gsm8k, humaneval (logiqa only for base)
#
# Expected runtime: ~2-3 hours
# ============================================================================

echo "=============================================="
echo "Path Signatures + True Jacobian Analysis"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniver_env

# Change to project directory
cd ~/maniver/ManiVer

# Verify data exists
echo ""
echo "Checking data files..."
for model in olmo3_base olmo3_sft olmo3_rl_zero; do
    for task in gsm8k humaneval; do
        f="data/trajectories_0shot/${model}/${task}_trajectories.h5"
        if [ -f "$f" ]; then
            echo "  OK: $f"
        else
            echo "  MISSING: $f"
        fi
    done
done

# Also check logiqa for base
f="data/trajectories_0shot/olmo3_base/logiqa_trajectories.h5"
if [ -f "$f" ]; then
    echo "  OK: $f"
else
    echo "  MISSING: $f"
fi

echo ""
echo "=============================================="
echo "Analysis 1: Path Signatures"
echo "=============================================="

python -u scripts/analysis/path_signature_analysis.py \
    --data-dir data/trajectories_0shot \
    --models olmo3_base,olmo3_sft,olmo3_rl_zero \
    --output-dir results \
    --max-samples 100

echo ""
echo "=============================================="
echo "Analysis 2: True Empirical Jacobian"
echo "=============================================="

python -u scripts/analysis/empirical_jacobian_lyapunov.py \
    --data-dir data/trajectories_0shot \
    --models olmo3_base,olmo3_sft,olmo3_rl_zero \
    --output-dir results \
    --max-samples 100

echo ""
echo "=============================================="
echo "Analysis Complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Output files:"
ls -lh results/h2_path_signatures*.csv results/h2_true_jacobian*.csv 2>/dev/null || echo "No output files found"
