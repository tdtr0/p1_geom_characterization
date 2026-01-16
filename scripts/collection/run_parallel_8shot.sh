#!/bin/bash
# Run 8-shot collection on 2 GPUs in parallel
# Usage: ./run_parallel_8shot.sh [task] [n_samples]

TASK=${1:-logiqa}
N_SAMPLES=${2:-500}

cd ~/p1_geom_characterization

echo "=============================================="
echo "Parallel 8-shot Collection: $TASK"
echo "=============================================="
echo "Samples: $N_SAMPLES per model"
echo "GPUs: 2x RTX 4090"
echo ""

# Function to run collection on specific GPU
run_on_gpu() {
    local gpu_id=$1
    local model=$2
    local task=$3
    local n_samples=$4

    echo "[GPU $gpu_id] Starting $model..."
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/collection/collect_8shot_trajectories.py \
        --model $model \
        --task $task \
        --n-samples $n_samples \
        2>&1 | tee data/logs/${model}_${task}_8shot.log
    echo "[GPU $gpu_id] Completed $model"
}

# Create logs directory
mkdir -p data/logs

# Batch 1: olmo3_base + olmo3_sft (parallel)
echo ""
echo "=== BATCH 1: olmo3_base + olmo3_sft ==="
run_on_gpu 0 olmo3_base $TASK $N_SAMPLES &
PID1=$!
run_on_gpu 1 olmo3_sft $TASK $N_SAMPLES &
PID2=$!

# Wait for batch 1
wait $PID1
wait $PID2

echo ""
echo "=== BATCH 1 COMPLETE ==="
echo ""

# Batch 2: olmo3_rl_zero + olmo3_think (parallel)
echo ""
echo "=== BATCH 2: olmo3_rl_zero + olmo3_think ==="
run_on_gpu 0 olmo3_rl_zero $TASK $N_SAMPLES &
PID3=$!
run_on_gpu 1 olmo3_think $TASK $N_SAMPLES &
PID4=$!

# Wait for batch 2
wait $PID3
wait $PID4

echo ""
echo "=== BATCH 2 COMPLETE ==="
echo ""

echo "=============================================="
echo "ALL MODELS COMPLETE"
echo "=============================================="

# Show results
ls -lh data/trajectories_8shot/*/
