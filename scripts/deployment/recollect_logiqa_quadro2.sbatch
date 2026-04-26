#!/bin/bash
#SBATCH --job-name=logiqa_q2
#SBATCH --output=/home/ttdo/logiqa_q2_out.txt
#SBATCH --error=/home/ttdo/logiqa_q2_err.txt
#SBATCH --time=0-12:00
#SBATCH --mem=80000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=quadro2

# Parallel job on quadro2 for olmo3_rl_zero and olmo3_think
# While quadro1 runs olmo3_sft

set -e

echo "========================================="
echo "LogiQA 0-shot (quadro2) - rl_zero + think"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

WORK_DIR="/home/ttdo/maniver"
REPO_DIR="$WORK_DIR/ManiVer"
cd $REPO_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniver_env

export PYTHONPATH=$REPO_DIR/src:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

nvidia-smi

# Only rl_zero and think (sft running on quadro1)
MODELS=("olmo3_rl_zero" "olmo3_think")
BATCH_SIZE=4
NUM_SAMPLES=500

echo "Models: ${MODELS[@]}"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================="
    echo "Collecting: $MODEL"
    echo "========================================="
    echo "Start: $(date)"

    python scripts/collection/collect_logiqa_vllm_fully_optimized.py \
        $MODEL \
        --batch-size $BATCH_SIZE \
        --num-samples $NUM_SAMPLES

    OUTPUT="data/trajectories/${MODEL}/logiqa_trajectories_vllm_optimized.h5"
    if [ -f "$OUTPUT" ]; then
        python -c "
import h5py
with h5py.File('$OUTPUT', 'r') as f:
    n = len(f['is_correct'])
    c = f['is_correct'][:].sum()
    print(f'  {n} samples, {c}/{n} correct ({100*c/n:.1f}%)')
"
        ls -lh "$OUTPUT"
    fi
done

echo ""
echo "========================================="
echo "quadro2 Complete!"
echo "========================================="
echo "End: $(date)"
