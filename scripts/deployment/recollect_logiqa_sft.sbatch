#!/bin/bash
#SBATCH --job-name=logiqa_sft
#SBATCH --output=/home/ttdo/logiqa_sft_out.txt
#SBATCH --error=/home/ttdo/logiqa_sft_err.txt
#SBATCH --time=0-16:00
#SBATCH --mem=80000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=quadro1

# Single model job: olmo3_sft
# 16 hour time limit (each model takes ~14h at 400s/batch)

set -e

echo "========================================="
echo "LogiQA 0-shot: olmo3_sft"
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

MODEL="olmo3_sft"
BATCH_SIZE=4
NUM_SAMPLES=500

echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Samples: $NUM_SAMPLES"

# Delete corrupted file if exists
rm -f "data/trajectories/${MODEL}/logiqa_trajectories_vllm_optimized.h5"

python scripts/collection/collect_logiqa_vllm_fully_optimized.py \
    $MODEL \
    --batch-size $BATCH_SIZE \
    --num-samples $NUM_SAMPLES

OUTPUT="data/trajectories/${MODEL}/logiqa_trajectories_vllm_optimized.h5"
if [ -f "$OUTPUT" ]; then
    echo ""
    echo "Verifying $OUTPUT..."
    python -c "
import h5py
with h5py.File('$OUTPUT', 'r') as f:
    n = len(f['is_correct'])
    c = f['is_correct'][:].sum()
    shape = f['trajectories'].shape
    print(f'  {n} samples, {c}/{n} correct ({100*c/n:.1f}%)')
    print(f'  Shape: {shape}')
"
    ls -lh "$OUTPUT"
    echo "Collection complete for $MODEL"
fi

echo ""
echo "========================================="
echo "olmo3_sft Complete!"
echo "========================================="
echo "End: $(date)"
