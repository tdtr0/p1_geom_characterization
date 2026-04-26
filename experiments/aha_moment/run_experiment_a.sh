#!/bin/bash
# Run Experiment A: Wynroe-Style Error Detection (GPU required)
#
# This script:
# 1. Generates CoT traces from RL-Zero model
# 2. Creates clean/corrupted pairs
# 3. Collects trajectories at error positions
# 4. Analyzes error-detection direction
#
# Requirements:
# - GPU with 16GB+ VRAM (24GB recommended for 7B models)
# - Python with torch, transformers, datasets, h5py, scipy

set -e

# Configuration
N_PROBLEMS=100  # Reduce for faster testing; 200 for full experiment
MODELS="rl_zero,think"  # Start with these; add "base,sft" for full comparison
OUTPUT_DIR="experiments/aha_moment/data/wynroe_replication"
RESULTS_DIR="experiments/aha_moment/results/wynroe"

# Detect if we're on vast.ai or local
if [ -d "/workspace" ]; then
    echo "Running on vast.ai instance"
    WORKDIR="/workspace/maniver"
else
    echo "Running locally"
    WORKDIR="$(dirname "$0")/../.."
fi

cd "$WORKDIR"

echo "============================================"
echo "Experiment A: Wynroe-Style Error Detection"
echo "============================================"
echo ""
echo "Configuration:"
echo "  N_PROBLEMS: $N_PROBLEMS"
echo "  MODELS: $MODELS"
echo "  OUTPUT: $OUTPUT_DIR"
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Install dependencies if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install torch transformers datasets h5py scipy matplotlib tqdm
fi

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

# Step 1: Collect clean/corrupted trajectory pairs
echo ""
echo "Step 1: Collecting clean/corrupted trajectory pairs..."
echo "This will take 3-6 hours for 200 problems × 4 models"
echo "======================================================"

python experiments/aha_moment/collect_clean_corrupted_pairs.py \
    --n_problems $N_PROBLEMS \
    --models $MODELS \
    --output "$OUTPUT_DIR"

# Step 2: Analyze error-detection direction
echo ""
echo "Step 2: Analyzing error-detection direction..."
echo "======================================================"

python experiments/aha_moment/analyze_wynroe_direction.py \
    --input "$OUTPUT_DIR/wynroe_trajectories.h5" \
    --output "$RESULTS_DIR"

# Summary
echo ""
echo "============================================"
echo "Experiment A Complete!"
echo "============================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Key files:"
echo "  - $RESULTS_DIR/wynroe_analysis.json"
echo "  - $RESULTS_DIR/layer_profile.png"
echo "  - $RESULTS_DIR/model_comparison.png"
echo "  - $RESULTS_DIR/projection_distributions.png"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/wynroe_analysis.json | python -m json.tool"
