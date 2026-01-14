#!/bin/bash
# Monitor collection and auto-restart crashed processes
# Usage: bash scripts/collection/monitor_and_restart.sh

set -e

cd /workspace/maniver
export PYTHONPATH=/workspace/maniver/src:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE

LOG_DIR="data/logs"
TRAJ_DIR="data/trajectories"

# Model to GPU mapping
declare -A MODEL_GPU=(
    ["olmo3_base"]="0"
    ["olmo3_sft"]="1"
    ["olmo3_rl_zero"]="2"
    ["olmo3_think"]="3"
)

MODELS=(olmo3_base olmo3_sft olmo3_rl_zero olmo3_think)

check_model_running() {
    local model=$1
    pgrep -f "collect_trajectories.*--models $model" > /dev/null 2>&1
}

get_model_progress() {
    local model=$1
    local total_size=0
    for task in gsm8k logiqa humaneval; do
        local file="$TRAJ_DIR/$model/${task}_trajectories.h5"
        if [[ -f "$file" ]]; then
            local size=$(stat -c%s "$file" 2>/dev/null || echo 0)
            total_size=$((total_size + size))
        fi
    done
    echo $((total_size / 1024 / 1024))  # MB
}

start_model() {
    local model=$1
    local gpu=${MODEL_GPU[$model]}
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/${model}_monitor_${timestamp}.log"

    echo "[$(date)] Starting $model on GPU $gpu..."

    # Clean any corrupted files for this model
    rm -rf "$TRAJ_DIR/$model"
    mkdir -p "$TRAJ_DIR/$model"

    # Remove old checkpoints to start fresh
    rm -f "data/checkpoints/labeled_${model}_"*.json

    CUDA_VISIBLE_DEVICES=$gpu HDF5_USE_FILE_LOCKING=FALSE \
        nohup python3 scripts/collection/collect_trajectories_with_labels.py \
        --models $model \
        --tasks gsm8k logiqa humaneval \
        --n_samples 500 \
        --max_new_tokens 512 \
        > "$log_file" 2>&1 &

    echo "[$(date)] Started $model with PID $! -> $log_file"
}

print_status() {
    echo ""
    echo "========================================"
    echo "STATUS: $(date)"
    echo "========================================"

    # GPU status
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

    echo ""
    echo "Model Status:"
    for model in "${MODELS[@]}"; do
        local gpu=${MODEL_GPU[$model]}
        local running="NO"
        local progress=$(get_model_progress $model)

        if check_model_running $model; then
            running="YES"
        fi

        printf "  %-15s GPU:%s  Running:%-3s  Size:%6dMB\n" "$model" "$gpu" "$running" "$progress"
    done

    echo ""
    echo "Total trajectory size: $(du -sh $TRAJ_DIR 2>/dev/null | cut -f1)"
    echo "========================================"
}

# Main monitoring loop
echo "Starting collection monitor..."
echo "Checking every 60 seconds for crashed processes"
echo ""

iteration=0
while true; do
    iteration=$((iteration + 1))

    # Print status every iteration
    print_status

    # Check for crashed processes and restart
    all_done=true
    for model in "${MODELS[@]}"; do
        if ! check_model_running $model; then
            # Check if this model has completed all tasks (each task ~4-8GB when done)
            progress=$(get_model_progress $model)

            if [[ $progress -lt 10000 ]]; then  # Less than 10GB = not complete
                echo ""
                echo "[$(date)] WARNING: $model not running (${progress}MB collected)"
                echo "[$(date)] Restarting $model..."
                start_model $model
                sleep 5  # Give it time to start
            else
                echo "[$(date)] $model appears complete (${progress}MB)"
            fi
        fi

        # Check if still running (for all_done check)
        if check_model_running $model; then
            all_done=false
        fi
    done

    if $all_done; then
        echo ""
        echo "========================================"
        echo "ALL MODELS COMPLETE!"
        echo "========================================"
        print_status

        # Run B2 upload
        echo ""
        echo "Starting B2 upload..."
        bash scripts/collection/run_phase2_pipeline.sh --upload-only

        break
    fi

    echo ""
    echo "Next check in 60 seconds... (Ctrl+C to stop monitoring)"
    sleep 60
done
