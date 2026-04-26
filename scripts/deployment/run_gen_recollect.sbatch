#!/bin/bash
#SBATCH --job-name=gen_recollect
#SBATCH --output=/home/ttdo/gen_recollect_out.txt
#SBATCH --error=/home/ttdo/gen_recollect_err.txt
#SBATCH --time=0-24:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=h100

set -e

echo "========================================="
echo "Generation Trajectory Recollection (Fixed)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

WORK_DIR="/home/$USER/maniver/ManiVer"
DATA_DIR="$WORK_DIR/data/generation_trajectories"
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
CONDA_ENV="maniver_env"

MODELS=("olmo3_base" "olmo3_rl_zero")
TASKS=("gsm8k" "humaneval" "logiqa")
MAX_SAMPLES=500

cd $WORK_DIR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

export PYTHONPATH=$WORK_DIR/src:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE

# Clear old data and checkpoints (they have zero hidden states)
echo "Clearing old data with zero hidden states..."
rm -rf $DATA_DIR/olmo3_base $DATA_DIR/olmo3_rl_zero
rm -f $CHECKPOINT_DIR/generation_*.json
mkdir -p $DATA_DIR $CHECKPOINT_DIR

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================="
    echo "Model: $MODEL — Start: $(date)"
    echo "========================================="

    for TASK in "${TASKS[@]}"; do
        echo ""
        echo "--- $MODEL / $TASK ---"
        echo "Start: $(date)"

        python scripts/collection/collect_generation_trajectories.py \
            --model $MODEL \
            --task $TASK \
            --output-dir $DATA_DIR \
            --checkpoint-dir $CHECKPOINT_DIR \
            --max-samples $MAX_SAMPLES \
            --layers-even \
            || echo "WARNING: $MODEL/$TASK had errors"

        OUTPUT_FILE="$DATA_DIR/${MODEL}/${TASK}_generation.h5"
        if [ -f "$OUTPUT_FILE" ]; then
            SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
            echo "Output: $OUTPUT_FILE ($SIZE)"
            python scripts/deployment/verify_gen_data.py "$OUTPUT_FILE" 2>&1 | head -5
        fi
        echo "End: $(date)"
    done

    # Upload after each model
    echo ""
    echo "Uploading $MODEL to B2..."
    b2 account authorize 0051b5729f87ba10000000001 K005ETXoFOt7CYSNMJLQz8Y4covA02g 2>&1 | tail -1

    for TASK in "${TASKS[@]}"; do
        LOCAL="$DATA_DIR/${MODEL}/${TASK}_generation.h5"
        if [ -f "$LOCAL" ]; then
            B2_PATH="generation_trajectories/${MODEL}/${TASK}_generation.h5"
            echo "  $LOCAL -> $B2_PATH"
            b2 file upload ml-activations-store "$LOCAL" "$B2_PATH" 2>&1 | tail -2
        fi
    done

    echo "$MODEL done at $(date)"
done

echo ""
echo "========================================="
echo "All done! End: $(date)"
echo "========================================="
