#!/bin/bash
#SBATCH --job-name=pre_ln_cmp
#SBATCH --output=results/pre_ln_%j.out
#SBATCH --error=results/pre_ln_%j.err
#SBATCH --partition=defq
#SBATCH --nodelist=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Pre-LayerNorm vs Post-LayerNorm Comparison
# Tests if cos(X_l, X_{l+1}) increases when using pre-LN activations
# Expected runtime: 20-30 minutes per model

set -e

echo "=========================================="
echo "Pre-LayerNorm vs Post-LayerNorm Comparison"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Configuration
WORK_DIR="${HOME}/maniver/ManiVer"
MODELS="olmo3_base olmo3_sft"  # Test base and SFT
N_SAMPLES=50  # Quick test, increase if results promising
SEED=42

cd "$WORK_DIR"

# Check if results directory exists
mkdir -p results

# Load conda
source /home/ttdo/miniconda3/etc/profile.d/conda.sh
conda activate base

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Install dependencies if needed
pip install transformer_lens -q 2>/dev/null || true

# Run comparison for each model
for MODEL in $MODELS; do
    echo ""
    echo "=========================================="
    echo "Testing model: $MODEL"
    echo "=========================================="
    echo "Time: $(date)"

    python scripts/analysis/pre_ln_comparison.py \
        --model "$MODEL" \
        --n-samples "$N_SAMPLES" \
        --seed "$SEED" \
        --output-dir results

    echo ""
    echo "Completed $MODEL at $(date)"
done

echo ""
echo "=========================================="
echo "All models completed"
echo "End time: $(date)"
echo "=========================================="

# Show results summary
echo ""
echo "Results files:"
ls -la results/pre_ln_comparison_*.json 2>/dev/null || echo "No result files found"
