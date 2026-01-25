#!/bin/bash
# Download 0-shot trajectory data from B2 to ai_inst
# Run this via SLURM job, NOT on login node

set -e

echo "=== B2 Download Script ==="
echo "Date: $(date)"

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniver_env

# Set up B2
cd ~/maniver/ManiVer
source configs/b2-configs.txt
b2 authorize-account $B2_KEY_ID $B2_APP_KEY

# Create directory structure
DATA_DIR=~/maniver/ManiVer/data/trajectories_0shot
mkdir -p $DATA_DIR/{olmo3_base,olmo3_sft,olmo3_rl_zero,olmo3_think}

echo ""
echo "=== Downloading GSM8K files ==="
for model in olmo3_base olmo3_sft olmo3_rl_zero olmo3_think; do
    echo "Downloading $model/gsm8k..."
    b2 download-file ml-activations-store \
        trajectories/$model/gsm8k_trajectories.h5 \
        $DATA_DIR/$model/gsm8k_trajectories.h5 || echo "WARNING: $model/gsm8k failed"
done

echo ""
echo "=== Downloading HumanEval files ==="
for model in olmo3_base olmo3_sft olmo3_rl_zero olmo3_think; do
    echo "Downloading $model/humaneval..."
    b2 download-file ml-activations-store \
        trajectories/$model/humaneval_trajectories.h5 \
        $DATA_DIR/$model/humaneval_trajectories.h5 || echo "WARNING: $model/humaneval failed"
done

echo ""
echo "=== Downloading LogiQA for base ==="
b2 download-file ml-activations-store \
    trajectories/olmo3_base/logiqa_trajectories.h5 \
    $DATA_DIR/olmo3_base/logiqa_trajectories.h5 || echo "WARNING: base/logiqa failed"

echo ""
echo "=== Creating symlinks for new LogiQA files ==="
TRAJ_DIR=~/maniver/ManiVer/data/trajectories
ln -sf $TRAJ_DIR/olmo3_rl_zero/logiqa_trajectories_vllm_optimized.h5 $DATA_DIR/olmo3_rl_zero/logiqa_trajectories.h5
ln -sf $TRAJ_DIR/olmo3_sft/logiqa_trajectories_vllm_optimized.h5 $DATA_DIR/olmo3_sft/logiqa_trajectories.h5
ln -sf $TRAJ_DIR/olmo3_think/logiqa_trajectories_vllm_optimized.h5 $DATA_DIR/olmo3_think/logiqa_trajectories.h5

echo ""
echo "=== Verifying files ==="
echo "File sizes:"
ls -lh $DATA_DIR/*/gsm8k*.h5 2>/dev/null || echo "No gsm8k files"
ls -lh $DATA_DIR/*/humaneval*.h5 2>/dev/null || echo "No humaneval files"
ls -lh $DATA_DIR/*/logiqa*.h5 2>/dev/null || echo "No logiqa files"

echo ""
echo "=== Download complete ==="
echo "Date: $(date)"
