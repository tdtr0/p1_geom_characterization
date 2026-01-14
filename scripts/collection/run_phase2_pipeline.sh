#!/bin/bash
#
# Phase 2 Complete Pipeline: Collect → Upload → Cleanup
#
# This script:
# 1. Sets up B2 CLI
# 2. Runs trajectory collection with labels
# 3. Automatically uploads to B2
# 4. Cleans up local storage (optional)
#
# Usage:
#   bash scripts/run_phase2_pipeline.sh [--models MODEL1,MODEL2] [--tasks TASK1,TASK2] [--keep-local]
#
# Examples:
#   # Full Phase 2 collection (all models, all tasks)
#   bash scripts/run_phase2_pipeline.sh
#
#   # Specific models and tasks
#   bash scripts/run_phase2_pipeline.sh --models olmo3_base,olmo3_rl_zero --tasks gsm8k,logiqa
#
#   # Keep local copies after upload
#   bash scripts/run_phase2_pipeline.sh --keep-local

set -e  # Exit on error

# Default settings
MODELS="olmo3_base,olmo3_sft,olmo3_rl_zero,olmo3_think"
TASKS="gsm8k,logiqa,humaneval"
KEEP_LOCAL=false
TIMESTAMP=$(date +%Y%m%d_%H%M)
RUN_NAME="phase2_${TIMESTAMP}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --keep-local)
            KEEP_LOCAL=true
            shift
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "PHASE 2 PIPELINE: Collect → Upload → Cleanup"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Models: $MODELS"
echo "  Tasks: $TASKS"
echo "  Run name: $RUN_NAME"
echo "  Keep local: $KEEP_LOCAL"
echo "  Timestamp: $TIMESTAMP"
echo ""
echo "========================================================================"
echo ""

# Step 1: Setup B2 (if not already done)
echo "[1/4] Setting up Backblaze B2..."
echo ""
if ! command -v b2 &> /dev/null; then
    echo "Installing B2 CLI..."
    pip install -q b2
fi

# Authorize B2
if [ -f "configs/b2-configs.txt" ]; then
    source <(grep -v '^#' configs/b2-configs.txt | sed 's/^/export /')
    b2 authorize-account "$B2_KEY_ID" "$B2_APP_KEY" > /dev/null
    echo "✓ B2 authorized (Bucket: $B2_BUCKET_NAME)"
else
    echo "⚠ Warning: configs/b2-configs.txt not found. Upload step will fail."
fi
echo ""

# Step 2: Run collection
echo "[2/4] Running Phase 2 trajectory collection..."
echo ""
echo "This will take 60-85 GPU hours for full collection."
echo "Progress will be checkpointed every 25 samples."
echo ""

# Create log file
LOG_FILE="logs/phase2_collection_${TIMESTAMP}.log"
mkdir -p logs

# Run collection with tee to show output and save to log
python scripts/collect_trajectories_with_labels.py 2>&1 | tee "$LOG_FILE"

COLLECTION_EXIT_CODE=${PIPESTATUS[0]}

if [ $COLLECTION_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "✗ Collection failed with exit code $COLLECTION_EXIT_CODE"
    echo "  Log file: $LOG_FILE"
    exit $COLLECTION_EXIT_CODE
fi

echo ""
echo "✓ Collection completed successfully"
echo ""

# Step 3: Upload to B2
echo "[3/4] Uploading trajectories to Backblaze B2..."
echo ""

if [ -z "$B2_BUCKET_NAME" ]; then
    echo "⚠ Skipping upload (B2 not configured)"
else
    # Check if data directory exists and has files
    if [ -d "data/trajectories" ] && [ "$(ls -A data/trajectories)" ]; then
        # Calculate total size
        TOTAL_SIZE=$(du -sh data/trajectories | cut -f1)
        echo "Total size to upload: $TOTAL_SIZE"
        echo ""

        # Upload using b2 sync
        echo "Syncing to b2://$B2_BUCKET_NAME/$RUN_NAME/trajectories/"
        b2 sync \
            --replace-newer \
            --keep-days 30 \
            --threads 4 \
            data/trajectories/ \
            "b2://$B2_BUCKET_NAME/$RUN_NAME/trajectories/"

        echo ""
        echo "✓ Upload completed successfully"
        echo ""
        echo "Files available at:"
        echo "  Direct B2: https://f005.backblazeb2.com/file/$B2_BUCKET_NAME/$RUN_NAME/trajectories/"
        echo "  Cloudflare CDN: https://$CLOUDFLARE_DOMAIN/$RUN_NAME/trajectories/"
        echo ""

        # Also upload checkpoints and logs
        if [ -d "data/checkpoints" ]; then
            echo "Uploading checkpoints..."
            b2 sync \
                --threads 2 \
                data/checkpoints/ \
                "b2://$B2_BUCKET_NAME/$RUN_NAME/checkpoints/"
        fi

        if [ -d "logs" ]; then
            echo "Uploading logs..."
            b2 sync \
                --threads 2 \
                logs/ \
                "b2://$B2_BUCKET_NAME/$RUN_NAME/logs/"
        fi

        echo ""
        echo "✓ All files uploaded to B2"
    else
        echo "⚠ No trajectories found in data/trajectories/"
    fi
fi
echo ""

# Step 4: Cleanup (optional)
echo "[4/4] Cleanup..."
echo ""

if [ "$KEEP_LOCAL" = true ]; then
    echo "Keeping local copies (--keep-local specified)"
else
    echo "Cleaning up local trajectories to save disk space..."
    echo ""
    read -p "Delete local trajectories? Files are safely stored in B2. [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf data/trajectories/*
        echo "✓ Local trajectories deleted"
    else
        echo "Keeping local copies"
    fi
fi

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  Run name: $RUN_NAME"
echo "  Log file: $LOG_FILE"
if [ -n "$B2_BUCKET_NAME" ]; then
    echo "  B2 location: b2://$B2_BUCKET_NAME/$RUN_NAME/"
    echo "  Download URL: https://f005.backblazeb2.com/file/$B2_BUCKET_NAME/$RUN_NAME/trajectories/"
fi
echo ""
echo "To download on another machine:"
echo "  python scripts/b2_download.py --remote-prefix $RUN_NAME/trajectories"
echo ""
echo "Next steps:"
echo "  1. Verify data quality: python scripts/verify_trajectories.py"
echo "  2. Compute signatures: python scripts/compute_signatures.py"
echo "  3. Test H1: python scripts/test_h1.py"
echo ""
