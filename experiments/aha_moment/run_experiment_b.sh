#!/bin/bash
# Run Experiment B: Natural Pivot Detection (CPU-only)
#
# This script:
# 1. Downloads Phase 2 trajectory data from B2 (olmo3_think)
# 2. Detects pivot tokens in model outputs
# 3. Analyzes trajectory dynamics at pivot points
#
# Requirements:
# - B2 CLI configured (configs/b2-configs.txt)
# - Python with transformers, h5py, scipy, matplotlib

set -e
cd "$(dirname "$0")/../.."  # Go to project root

echo "============================================"
echo "Experiment B: Natural Pivot Detection"
echo "============================================"
echo ""

# Configuration
DATA_DIR="experiments/aha_moment/data/phase2"
RESULTS_DIR="experiments/aha_moment/results"
PIVOT_LABELS="experiments/aha_moment/data/pivot_labels.json"

# Models to analyze (Think model has most pivots)
# Add more models by uncommenting
MODELS=(
    "olmo3_think"
    # "olmo3_rl_zero"
    # "olmo3_sft"
    # "olmo3_base"
)

# Tasks to analyze
TASKS=(
    "gsm8k"
    "logiqa"
    # "humaneval"  # Less interesting for pivot analysis
)

mkdir -p "$DATA_DIR" "$RESULTS_DIR"

# Step 1: Download Phase 2 data from B2
echo ""
echo "Step 1: Downloading Phase 2 trajectories from B2..."
echo "======================================================"

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        file="${model}_${task}.h5"
        local_path="$DATA_DIR/$file"

        if [ -f "$local_path" ]; then
            echo "  [SKIP] $file (already exists)"
        else
            echo "  [DOWNLOAD] $file"
            python scripts/storage/b2_download.py \
                --file "trajectories/${model}/${file}" \
                --local-dir "$DATA_DIR" \
                2>&1 | grep -v "^Authorizing"
        fi
    done
done

# Check what we downloaded
echo ""
echo "Downloaded files:"
ls -lh "$DATA_DIR"/*.h5 2>/dev/null || echo "  No HDF5 files found!"

# Step 2: Detect pivots in model outputs
echo ""
echo "Step 2: Detecting pivot tokens..."
echo "======================================================"

# Build list of input files
INPUT_FILES=""
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        file="$DATA_DIR/${model}_${task}.h5"
        if [ -f "$file" ]; then
            INPUT_FILES="$INPUT_FILES $file"
        fi
    done
done

if [ -z "$INPUT_FILES" ]; then
    echo "ERROR: No trajectory files found in $DATA_DIR"
    exit 1
fi

python experiments/aha_moment/detect_pivots.py \
    --input $INPUT_FILES \
    --output "$PIVOT_LABELS" \
    --method regex

# Step 3: Analyze pivot dynamics
echo ""
echo "Step 3: Analyzing pivot dynamics..."
echo "======================================================"

python experiments/aha_moment/analyze_phase2_pivots.py \
    --trajectories $INPUT_FILES \
    --pivots "$PIVOT_LABELS" \
    --output "$RESULTS_DIR" \
    --plot-examples 5

# Summary
echo ""
echo "============================================"
echo "Experiment B Complete!"
echo "============================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Key files:"
echo "  - $RESULTS_DIR/pivot_analysis_results.json"
echo "  - $RESULTS_DIR/pivot_comparison_all_samples.png"
echo "  - $RESULTS_DIR/example_trajectory_*.png"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/pivot_analysis_results.json | python -m json.tool"
