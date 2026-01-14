#!/bin/bash
# Phase 2 Collection - PARALLEL Execution (4x faster!)
# Runs all 4 models in parallel, one per GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure PYTHONPATH includes src/
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# CRITICAL: Disable HDF5 file locking to prevent conflicts between processes
export HDF5_USE_FILE_LOCKING=FALSE

echo "========================================================================"
echo "Phase 2 Collection - PARALLEL Mode (4 GPUs)"
echo "========================================================================"
echo ""
echo "Running 4 models in parallel (one per GPU):"
echo "  GPU 0: olmo3_base"
echo "  GPU 1: olmo3_sft"
echo "  GPU 2: olmo3_rl_zero"
echo "  GPU 3: olmo3_think"
echo ""
echo "Expected duration: ~1.5-2 hours (4x faster than sequential)"
echo "Expected output: ~100 GB (12 HDF5 files)"
echo ""
echo "========================================================================"
echo ""

# Create directories
mkdir -p data/trajectories data/checkpoints data/logs

# Define models and their assigned GPUs
declare -A MODEL_TO_GPU=(
    ["olmo3_base"]="0"
    ["olmo3_sft"]="1"
    ["olmo3_rl_zero"]="2"
    ["olmo3_think"]="3"
)

# Start collection for each model in parallel (staggered to avoid HDF5 conflicts)
PIDS=()
delay=0
for model in olmo3_base olmo3_sft olmo3_rl_zero olmo3_think; do
    gpu=${MODEL_TO_GPU[$model]}
    log_file="data/logs/${model}_collection_$(date +%Y%m%d_%H%M%S).log"

    # Stagger starts by 10 seconds to avoid HDF5 file creation conflicts
    if [ $delay -gt 0 ]; then
        echo "[$(date +%T)] Waiting ${delay}s before starting $model..."
        sleep $delay
    fi

    echo "[$(date +%T)] Starting $model on GPU $gpu (log: $log_file)"

    # Run collection for this model on its assigned GPU
    CUDA_VISIBLE_DEVICES=$gpu HDF5_USE_FILE_LOCKING=FALSE \
        python3 scripts/collection/collect_trajectories_with_labels.py \
        --models $model \
        --tasks gsm8k logiqa humaneval \
        --n_samples 500 \
        --max_new_tokens 512 \
        > "$log_file" 2>&1 &

    PIDS+=($!)
    delay=10  # 10 second delay between starts
done

echo ""
echo "All 4 models started! PIDs: ${PIDS[@]}"
echo ""
echo "Monitor progress:"
echo "  tail -f data/logs/*_collection_*.log"
echo "  watch -n 30 'find data/trajectories -name \"*.h5\" | wc -l'"
echo "  watch nvidia-smi"
echo ""
echo "Waiting for all models to complete..."
echo ""

# Monitor progress
start_time=$(date +%s)
while true; do
    # Check if all processes are still running
    all_done=true
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        break
    fi

    # Show progress every 5 minutes
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $((elapsed % 300)) -eq 0 ] && [ $elapsed -gt 0 ]; then
        echo "[$(date +%T)] Progress update ($(($elapsed / 60)) minutes elapsed):"
        echo "  Files created: $(find data/trajectories -name '*.h5' 2>/dev/null | wc -l) / 12"
        echo "  Total size: $(du -sh data/trajectories 2>/dev/null | cut -f1)"

        # Show per-model status
        for model in olmo3_base olmo3_sft olmo3_rl_zero olmo3_think; do
            count=$(find data/trajectories/$model -name '*.h5' 2>/dev/null | wc -l)
            echo "    $model: $count/3 tasks complete"
        done
        echo ""
    fi

    sleep 60
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "========================================================================"
echo "ALL MODELS COMPLETE!"
echo "========================================================================"
echo "Total time: $(($total_time / 60)) minutes"
echo ""

# Check for any errors
echo "Checking for errors..."
error_count=0
for model in olmo3_base olmo3_sft olmo3_rl_zero olmo3_think; do
    log_file=$(ls -t data/logs/${model}_collection_*.log 2>/dev/null | head -1)
    if [ -n "$log_file" ]; then
        errors=$(grep -i "error\|failed\|traceback" "$log_file" | head -5)
        if [ -n "$errors" ]; then
            echo "✗ $model had errors:"
            echo "$errors"
            error_count=$((error_count + 1))
        else
            echo "✓ $model completed successfully"
        fi
    fi
done

echo ""
echo "Summary:"
echo "  Files created: $(find data/trajectories -name '*.h5' | wc -l) / 12"
echo "  Total size: $(du -sh data/trajectories | cut -f1)"
echo "  Models with errors: $error_count / 4"
echo ""

if [ $error_count -eq 0 ]; then
    echo "========================================================================"
    echo "Uploading to Backblaze B2..."
    echo "========================================================================"
    echo ""

    python3 scripts/storage/b2_upload.py

    echo ""
    echo "========================================================================"
    echo "PHASE 2 COLLECTION COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Verify upload: python3 scripts/storage/b2_download.py --list-only"
    echo "  2. Destroy instance to stop charges"
    echo "  3. Download for analysis when needed"
    echo ""
else
    echo "⚠️  Some models had errors. Check logs before uploading."
    echo "Skipping B2 upload."
fi
