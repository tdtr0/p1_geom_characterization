#!/usr/bin/env bash
#
# Quick test of optimized collection script
# Tests on 10 samples to verify all optimizations work correctly
#

set -e

echo "================================"
echo "Testing Optimized Collection"
echo "================================"
echo ""

MODEL_KEY="olmo3_sft"
BATCH_SIZE=4
NUM_SAMPLES=10

echo "Model: $MODEL_KEY"
echo "Batch size: $BATCH_SIZE"
echo "Samples: $NUM_SAMPLES (test run)"
echo ""

# Run optimized collection
cd /Users/thanhdo/CascadeProjects/ManiVer/main

PYTHONPATH=./src python scripts/collection/collect_logiqa_optimized.py \
    $MODEL_KEY \
    --batch-size $BATCH_SIZE \
    --num-samples $NUM_SAMPLES

# Check output
OUTPUT_FILE="data/trajectories/${MODEL_KEY}/logiqa_trajectories_optimized.h5"

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ Output file created successfully"
    ls -lh "$OUTPUT_FILE"

    # Show HDF5 structure
    echo ""
    echo "HDF5 file structure:"
    python -c "
import h5py
import sys

with h5py.File('$OUTPUT_FILE', 'r') as f:
    print(f'  Trajectories: {f[\"trajectories\"].shape}')
    print(f'  Correct: {f[\"is_correct\"][:].sum()}/{len(f[\"is_correct\"])}')
    print(f'  Sample output length: {len(f[\"model_outputs\"][0])}')
"
else
    echo "✗ Output file not found!"
    exit 1
fi

echo ""
echo "Test completed successfully!"
