#!/bin/bash
#SBATCH --job-name=add_analyses
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/ttdo/additional_analyses_out.txt
#SBATCH --error=/home/ttdo/additional_analyses_err.txt

# Additional Analyses for Surface Structure vs Reasoning
# - Path Signatures (requires signatory library)
# - Token-Position Specificity
# - Cross-Domain Subspace Alignment
#
# CPU-only (no GPU needed)
# Uses tesla2 which should be available

echo "=== Additional Analyses Job Started ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Environment setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniver_env
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

cd ~/maniver/ManiVer

# Install signatory if not present (for path signatures)
echo "=== Checking/Installing signatory ==="
pip show signatory > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing signatory..."
    pip install signatory==1.2.6.1.9.0 --quiet
else
    echo "signatory already installed"
fi

# Data and output directories
DATA_DIR=~/maniver/ManiVer/data/trajectories_0shot
RESULTS_DIR=~/maniver/ManiVer/results/additional_analyses

mkdir -p $RESULTS_DIR

echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS_DIR"
echo ""

# Check data exists
echo "=== Checking data files ==="
ls -lh $DATA_DIR/*/gsm8k*.h5 2>/dev/null || echo "No gsm8k files"
ls -lh $DATA_DIR/*/logiqa*.h5 2>/dev/null || echo "No logiqa files"
echo ""

# Run additional analyses
echo "=== Running Additional Analyses ==="
python scripts/analysis/additional_analyses.py \
    --data-dir $DATA_DIR \
    --output-dir $RESULTS_DIR \
    --models olmo3_base \
    --tasks gsm8k,logiqa \
    --max-samples 200

echo ""
echo "=== Job Complete ==="
echo "Results in: $RESULTS_DIR"
ls -la $RESULTS_DIR/
echo ""
echo "Date: $(date)"
