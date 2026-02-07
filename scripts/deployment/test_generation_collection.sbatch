#!/bin/bash
#SBATCH --job-name=gen_test
#SBATCH --output=/home/ttdo/gen_test_out.txt
#SBATCH --error=/home/ttdo/gen_test_err.txt
#SBATCH --time=0-01:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=quadro2

#
# TEST: Generation Trajectory Collection (5 samples only)
#
# Validates:
#   - Hook-based activation capture works
#   - Checkpointing works
#   - HDF5 output is valid
#
# Usage:
#   sbatch scripts/deployment/test_generation_collection.sbatch
#
# Monitor:
#   tail -f ~/gen_test_out.txt
#

set -e

echo "========================================="
echo "Generation Collection TEST (5 samples)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# === Configuration ===
WORK_DIR="/home/$USER/maniver/ManiVer"
DATA_DIR="$WORK_DIR/data/generation_trajectories_test"
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
CONDA_ENV="maniver_env"

# Test with just base model and gsm8k
MODEL="olmo3_base"
TASK="gsm8k"
MAX_SAMPLES=5

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

# === Test Collection ===
echo "--- Testing: $MODEL / $TASK (5 samples) ---"
echo "Start: $(date)"

python scripts/collection/collect_generation_trajectories.py \
    --model $MODEL \
    --task $TASK \
    --output-dir $DATA_DIR \
    --checkpoint-dir $CHECKPOINT_DIR \
    --max-samples $MAX_SAMPLES \
    --layers-even \
    || {
        echo "ERROR: Collection failed!"
        exit 1
    }

# Verify output
OUTPUT_FILE="$DATA_DIR/${MODEL}/${TASK}_generation.h5"
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    echo ""
    echo "SUCCESS: Output file created: $OUTPUT_FILE ($SIZE)"
    echo ""

    # Detailed validation
    python -c "
import h5py
import numpy as np

with h5py.File('$OUTPUT_FILE', 'r') as f:
    print('=== HDF5 Structure ===')
    print(f'Top-level keys: {list(f.keys())}')

    samples = [k for k in f.keys() if k.startswith('sample_')]
    print(f'Number of samples: {len(samples)}')

    if samples:
        s = f[samples[0]]
        print(f'\\n=== Sample 0 Structure ===')
        print(f'Keys: {list(s.keys())}')

        if 'hidden_states' in s:
            hs = s['hidden_states'][:]
            print(f'hidden_states shape: {hs.shape}')
            print(f'hidden_states dtype: {hs.dtype}')

        if 'entropy' in s:
            ent = s['entropy'][:]
            print(f'entropy shape: {ent.shape}')
            print(f'entropy range: [{ent.min():.3f}, {ent.max():.3f}]')

        if 'top_100_tokens' in s:
            tok = s['top_100_tokens'][:]
            print(f'top_100_tokens shape: {tok.shape}')

        if 'attention_weights' in s:
            attn = s['attention_weights'][:]
            print(f'attention_weights shape: {attn.shape}')

        # Check attributes
        print(f'\\n=== Sample 0 Attributes ===')
        for attr in s.attrs:
            val = s.attrs[attr]
            if isinstance(val, bytes):
                val = val.decode('utf-8')[:100]
            print(f'{attr}: {val}')

print('\\n=== Validation PASSED ===')
"
else
    echo "ERROR: Output file not found: $OUTPUT_FILE"
    exit 1
fi

echo ""
echo "End: $(date)"
echo ""
echo "========================================="
echo "TEST COMPLETE - Ready for full collection"
echo "========================================="
