#!/bin/bash
#SBATCH --job-name=logiqa_recollect
#SBATCH --output=/home/ttdo/logiqa_recollect_out.txt
#SBATCH --error=/home/ttdo/logiqa_recollect_err.txt
#SBATCH --time=0-04:00
#SBATCH --mem=80000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=quadro1

#
# LogiQA 0-shot recollection with full trajectory capture (no truncation)
#
# Models: olmo3_sft, olmo3_rl_zero, olmo3_think
# MAX_SEQ_LEN = 2048 (captures full prompt + answer)
#
# Submit with:
#   sbatch scripts/deployment/recollect_logiqa_0shot.sbatch
#
# Monitor with:
#   squeue -u ttdo
#   tail -f /home/ttdo/logiqa_recollect_out.txt
#

set -e

echo "========================================="
echo "LogiQA 0-shot Recollection"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Environment setup
WORK_DIR="/home/ttdo/maniver"
REPO_DIR="$WORK_DIR/ManiVer"

echo "Working directory: $REPO_DIR"
cd $REPO_DIR

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniver_env

# Set environment variables
export PYTHONPATH=$REPO_DIR/src:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE

# Show environment info
echo ""
echo "Python: $(which python)"
python --version
echo ""

# Verify GPU
echo "GPU Status:"
nvidia-smi
echo ""

# Verify vLLM
echo "Verifying vLLM..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
echo ""

# Models to recollect (corrupted 0-shot LogiQA files)
MODELS=("olmo3_sft" "olmo3_rl_zero" "olmo3_think")
BATCH_SIZE=4  # Reduced for 48GB GPU (vs 8 for 80GB H100)
NUM_SAMPLES=500

echo "========================================="
echo "Configuration"
echo "========================================="
echo "Models: ${MODELS[@]}"
echo "Batch size: $BATCH_SIZE"
echo "Samples per model: $NUM_SAMPLES"
echo "MAX_SEQ_LEN: 2048 (full trajectory)"
echo ""

# Run collection for each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================="
    echo "Collecting: $MODEL"
    echo "========================================="
    echo "Start time: $(date)"
    echo ""

    python scripts/collection/collect_logiqa_vllm_fully_optimized.py \
        $MODEL \
        --batch-size $BATCH_SIZE \
        --num-samples $NUM_SAMPLES

    # Verify output file
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
    size_gb = shape[0] * shape[1] * shape[2] * shape[3] * 2 / (1024**3)
    print(f'  Samples: {n}')
    print(f'  Correct: {c}/{n} ({100*c/n:.1f}%)')
    print(f'  Trajectory shape: {shape}')
    print(f'  Raw size: {size_gb:.1f} GB')
"
        ls -lh "$OUTPUT"
        echo "Collection complete for $MODEL"
    else
        echo "ERROR: Output file not found for $MODEL!"
        exit 1
    fi
done

echo ""
echo "========================================="
echo "All Collections Complete!"
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Output files:"
for MODEL in "${MODELS[@]}"; do
    ls -lh "data/trajectories/${MODEL}/logiqa_trajectories_vllm_optimized.h5"
done
echo ""
echo "Next step: Upload to B2"
echo "  Option A: sbatch scripts/deployment/upload_to_b2.sbatch"
echo "  Option B: python scripts/storage/b2_upload.py"
echo ""
