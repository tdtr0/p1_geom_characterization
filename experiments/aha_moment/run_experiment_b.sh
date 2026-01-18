#!/bin/bash
# Run Experiment B: Natural Pivot Detection (GPU required)
#
# This script:
# 1. Generates text from olmo3_think and collects generation trajectories
# 2. Detects pivot tokens in generated text
# 3. Analyzes trajectory dynamics at pivot points
#
# Requirements:
# - GPU with 16GB+ VRAM (24GB recommended for 7B model)
# - Python with transformers, h5py, scipy, matplotlib, datasets
#
# IMPORTANT: Phase 2 data CANNOT be used for this experiment because
# it only contains prompt trajectories, not generation trajectories.
# Pivots occur during generation, so we must collect new data.

set -e
cd "$(dirname "$0")/../.."  # Go to project root

echo "============================================"
echo "Experiment B: Natural Pivot Detection"
echo "============================================"
echo ""

# Configuration
N_SAMPLES=${N_SAMPLES:-200}
MODEL=${MODEL:-"olmo3_think"}
MAX_TOKENS=${MAX_TOKENS:-512}
OUTPUT_DIR="experiments/aha_moment/data/pivot_collection"
RESULTS_DIR="experiments/aha_moment/results/pivot_analysis"

echo "Configuration:"
echo "  N_SAMPLES: $N_SAMPLES"
echo "  MODEL: $MODEL"
echo "  MAX_TOKENS: $MAX_TOKENS"
echo "  OUTPUT: $OUTPUT_DIR"
echo ""

# Check for GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "WARNING: nvidia-smi not found. This experiment requires a GPU."
fi
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

# Check if data already exists
H5_FILE="$OUTPUT_DIR/pivot_trajectories.h5"

if [ -f "$H5_FILE" ]; then
    echo "Found existing data: $H5_FILE"
    echo "Skipping collection, going directly to analysis."
    echo ""
else
    # Step 1: Collect generation trajectories
    echo ""
    echo "Step 1: Collecting generation trajectories..."
    echo "======================================================"
    echo "This will take 2-3 hours for $N_SAMPLES samples."
    echo ""

    python experiments/aha_moment/collect_pivot_trajectories.py \
        --n_samples "$N_SAMPLES" \
        --model "$MODEL" \
        --max_tokens "$MAX_TOKENS" \
        --output "$OUTPUT_DIR"
fi

# Step 2: Analyze pivot dynamics
echo ""
echo "Step 2: Analyzing pivot dynamics..."
echo "======================================================"

python experiments/aha_moment/analyze_pivot_trajectories.py \
    --input "$H5_FILE" \
    --output "$RESULTS_DIR"

# Summary
echo ""
echo "============================================"
echo "Experiment B Complete!"
echo "============================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Key files:"
echo "  - $OUTPUT_DIR/pivot_trajectories.h5"
echo "  - $OUTPUT_DIR/collection_summary.json"
echo "  - $RESULTS_DIR/pivot_analysis.json"
echo "  - $RESULTS_DIR/pivot_distributions.png"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/pivot_analysis.json | python -m json.tool"
