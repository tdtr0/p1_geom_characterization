#!/bin/bash
# =============================================================================
# Phase 2b: 8-Shot GSM8K Collection Pipeline
# Run this on vast.ai after instance is set up
# =============================================================================

set -e  # Exit on error

echo "=============================================================="
echo "PHASE 2b: 8-Shot GSM8K Trajectory Collection"
echo "=============================================================="
echo ""
echo "Format: lm-evaluation-harness standard 8-shot CoT"
echo "Models: olmo3_base, olmo3_sft, olmo3_rl_zero, olmo3_think"
echo "Samples: 500 per model"
echo ""

# Configuration
WORKDIR="${WORKDIR:-/workspace/maniver}"
N_SAMPLES="${N_SAMPLES:-500}"
LOG_DIR="$WORKDIR/data/logs"
OUTPUT_DIR="$WORKDIR/data/trajectories_8shot"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$WORKDIR"

# Setup B2 if credentials exist
if [ -f "configs/b2-configs.txt" ]; then
    echo "Setting up B2 credentials..."
    source configs/b2-configs.txt
    export B2_APPLICATION_KEY_ID
    export B2_APPLICATION_KEY
fi

# Fix HDF5 file locking (needed for some systems)
export HDF5_USE_FILE_LOCKING=FALSE

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/8shot_collection_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo ""

# Run collection for all models
{
    echo "Starting 8-shot collection at $(date)"
    echo ""

    # Check GPU
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""

    # Run collection
    python scripts/collection/collect_8shot_trajectories.py \
        --all \
        --n-samples "$N_SAMPLES"

    echo ""
    echo "Collection completed at $(date)"

    # Show results
    echo ""
    echo "Results:"
    ls -la "$OUTPUT_DIR"/*/

    # Calculate total size
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
    echo ""
    echo "Total size: $TOTAL_SIZE"

} 2>&1 | tee "$LOG_FILE"

# Upload to B2 if configured
if [ -n "$B2_APPLICATION_KEY_ID" ]; then
    echo ""
    echo "=============================================================="
    echo "Uploading to Backblaze B2..."
    echo "=============================================================="

    python scripts/storage/b2_upload.py \
        --local-dir "$OUTPUT_DIR" \
        --remote-prefix "trajectories_8shot" \
        --bucket "ml-activations-store"

    echo "Upload complete!"
fi

echo ""
echo "=============================================================="
echo "8-Shot Collection Pipeline Complete"
echo "=============================================================="
echo ""
echo "Next steps:"
echo "1. Download data: python scripts/storage/b2_download.py --remote-prefix trajectories_8shot"
echo "2. Run analysis comparing 0-shot vs 8-shot geometry"
echo ""
