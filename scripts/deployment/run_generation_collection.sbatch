#!/bin/bash
#SBATCH --job-name=gen_collect
#SBATCH --output=/home/ttdo/gen_collect_out.txt
#SBATCH --error=/home/ttdo/gen_collect_err.txt
#SBATCH --time=0-24:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=h100

#
# Generation Trajectory Collection
#
# Collects:
#   - Hidden states during generation (even layers)
#   - Attention weights (selective layers)
#   - Top-100 tokens + entropy
#
# Features:
#   - Per-sample checkpointing (resumable)
#   - Uploads to B2 after each model
#
# Usage:
#   sbatch scripts/deployment/run_generation_collection.sbatch
#
# Monitor:
#   tail -f ~/gen_collect_out.txt
#

set -e

echo "========================================="
echo "Generation Trajectory Collection (SLURM)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# === Configuration ===
WORK_DIR="/home/$USER/maniver/ManiVer"
DATA_DIR="$WORK_DIR/data/generation_trajectories"
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
CONDA_ENV="maniver_env"

# Priority order: base first, then rl_zero
MODELS=("olmo3_base" "olmo3_rl_zero")
TASKS=("gsm8k" "humaneval" "logiqa")

MAX_SAMPLES=500

# === Setup ===
cd $WORK_DIR

# Activate conda
source activate $CONDA_ENV || {
    echo "ERROR: Could not activate conda environment: $CONDA_ENV"
    exit 1
}

export PYTHONPATH=$WORK_DIR/src:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE

# Create directories
mkdir -p $DATA_DIR $CHECKPOINT_DIR

echo "Working directory: $WORK_DIR"
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# GPU check
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# === Collection Loop ===
for MODEL in "${MODELS[@]}"; do
    MODEL_START=$(date)
    echo ""
    echo "========================================="
    echo "Model: $MODEL"
    echo "Started: $MODEL_START"
    echo "========================================="
    echo ""

    for TASK in "${TASKS[@]}"; do
        echo "--- $MODEL / $TASK ---"
        echo "Start: $(date)"

        python scripts/collection/collect_generation_trajectories.py \
            --model $MODEL \
            --task $TASK \
            --output-dir $DATA_DIR \
            --checkpoint-dir $CHECKPOINT_DIR \
            --max-samples $MAX_SAMPLES \
            --layers-even \
            --resume \
            || {
                echo "WARNING: $MODEL/$TASK had errors"
                # Continue even if one task fails
            }

        # Verify output
        OUTPUT_FILE="$DATA_DIR/${MODEL}/${TASK}_generation.h5"
        if [ -f "$OUTPUT_FILE" ]; then
            SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
            echo "Output: $OUTPUT_FILE ($SIZE)"

            # Quick validation
            python -c "
import h5py
import sys
try:
    with h5py.File('$OUTPUT_FILE', 'r') as f:
        n_samples = len([k for k in f.keys() if k.startswith('sample_')])
        n_correct = sum(1 for k in f.keys() if k.startswith('sample_') and f[k].attrs.get('is_correct', False))
        print(f'  Samples: {n_samples}, Correct: {n_correct} ({n_correct/n_samples*100:.1f}%)')
except Exception as e:
    print(f'  Error reading file: {e}')
    sys.exit(1)
" || echo "  Validation failed"
        else
            echo "WARNING: Output file not found: $OUTPUT_FILE"
        fi

        echo "End: $(date)"
        echo ""
    done

    # === Upload after each model ===
    echo "========================================="
    echo "Uploading $MODEL to B2..."
    echo "========================================="

    B2_CONFIG="$WORK_DIR/configs/b2-configs.txt"

    if [ -f "$B2_CONFIG" ]; then
        # Source B2 credentials
        source $B2_CONFIG

        # Install b2 if needed
        if ! command -v b2 &> /dev/null; then
            echo "Installing B2 CLI..."
            pip install b2sdk b2 --quiet
        fi

        # Authorize
        b2 authorize-account $B2_APPLICATION_KEY_ID $B2_APPLICATION_KEY 2>/dev/null || {
            echo "ERROR: B2 authorization failed"
            continue
        }

        # Upload each task file
        for TASK in "${TASKS[@]}"; do
            LOCAL="$DATA_DIR/${MODEL}/${TASK}_generation.h5"
            if [ -f "$LOCAL" ]; then
                B2_PATH="generation/${MODEL}/${TASK}_generation.h5"
                echo "Uploading: $LOCAL -> b2://ml-activations-store/$B2_PATH"

                b2 upload-file --noProgress ml-activations-store "$LOCAL" "$B2_PATH" && \
                    echo "  Uploaded successfully" || \
                    echo "  Upload FAILED"
            fi
        done
    else
        echo "WARNING: B2 config not found at $B2_CONFIG"
        echo "Skipping upload - files remain in: $DATA_DIR/$MODEL/"
    fi

    echo ""
    echo "Model $MODEL completed at $(date)"
    echo ""
done

# === Summary ===
echo ""
echo "========================================="
echo "Collection Complete!"
echo "========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo ""

# List output files
echo "Output files:"
find $DATA_DIR -name "*.h5" -exec ls -lh {} \; 2>/dev/null || echo "  No files found"
echo ""

# Show checkpoint status
echo "Checkpoint status:"
ls -la $CHECKPOINT_DIR/generation_*.json 2>/dev/null || echo "  No checkpoints found"
echo ""

echo "Done!"
