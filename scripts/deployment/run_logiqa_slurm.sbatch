#!/bin/bash
#SBATCH --job-name=logiqa_vllm
#SBATCH --output=/home/<netid>/logiqa_collection_out.txt
#SBATCH --error=/home/<netid>/logiqa_collection_err.txt
#SBATCH --time=0-03:00
#SBATCH --mem=60000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=h100

#
# SLURM job file for fully optimized LogiQA collection with vLLM
#
# BEFORE SUBMITTING:
#   1. Replace <netid> with your actual network ID in lines 3-4
#   2. Run setup_slurm_env.sh first (one-time setup)
#   3. Copy B2 credentials: cp configs/b2-configs.txt /scratch/$USER/maniver/ManiVer/configs/
#
# Submit with:
#   sbatch scripts/deployment/run_logiqa_slurm.sbatch
#
# Monitor with:
#   squeue -u $USER
#   tail -f ~/logiqa_collection_out.txt
#

set -e  # Exit on any error

echo "========================================="
echo "LogiQA Collection with vLLM (SLURM)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load slurm python3

# Storage location - UPDATE THIS after running check_slurm_storage.sh
# Options:
#   /home/$USER/maniver         (if home quota allows >20GB)
#   /scratch/$USER/maniver      (if scratch is persistent)
#   /work/$USER/maniver         (if work directory exists)
WORK_DIR="/home/$USER/maniver"

REPO_DIR="$WORK_DIR/ManiVer"
CONDA_ENV_NAME="maniver_env"

echo "Working directory: $REPO_DIR"
echo "Conda environment: $CONDA_ENV_NAME"
echo ""

# Navigate to repo
cd $REPO_DIR

# Activate conda environment
source activate $CONDA_ENV_NAME

# Verify GPU is available
echo "Checking GPU..."
nvidia-smi
echo ""

# Verify Python and packages
echo "Python version:"
python --version
echo ""

echo "Verifying vLLM installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
echo ""

# Models to collect
MODELS=("olmo3_sft" "olmo3_rl_zero" "olmo3_think")
BATCH_SIZE=8
NUM_SAMPLES=500

echo "========================================="
echo "Collection Configuration"
echo "========================================="
echo "Models: ${MODELS[@]}"
echo "Batch size: $BATCH_SIZE"
echo "Samples per model: $NUM_SAMPLES"
echo ""

# Run collection for each model
for MODEL_KEY in "${MODELS[@]}"; do
    echo "========================================="
    echo "Collecting: $MODEL_KEY"
    echo "========================================="
    echo ""

    # Set PYTHONPATH
    export PYTHONPATH=$REPO_DIR/src:$PYTHONPATH

    # Disable HDF5 file locking (cluster filesystem issues)
    export HDF5_USE_FILE_LOCKING=FALSE

    # Run collection
    python scripts/collection/collect_logiqa_vllm_fully_optimized.py \
        $MODEL_KEY \
        --batch-size $BATCH_SIZE \
        --num-samples $NUM_SAMPLES

    # Check if output file was created
    OUTPUT_FILE="data/trajectories/${MODEL_KEY}/logiqa_trajectories_vllm_optimized.h5"

    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "✓ Collection completed for $MODEL_KEY"
        ls -lh "$OUTPUT_FILE"

        # Verify HDF5 file integrity
        echo "Verifying HDF5 file..."
        python -c "
import h5py
import sys

try:
    with h5py.File('$OUTPUT_FILE', 'r') as f:
        n_samples = len(f['is_correct'])
        n_correct = f['is_correct'][:].sum()
        print(f'  Samples: {n_samples}')
        print(f'  Correct: {n_correct}/{n_samples} ({n_correct/n_samples*100:.1f}%)')
        print(f'  Trajectory shape: {f[\"trajectories\"].shape}')
        print('  ✓ File is valid')
except Exception as e:
    print(f'  ✗ Error reading file: {e}')
    sys.exit(1)
"
        echo ""
    else
        echo "✗ ERROR: Output file not found for $MODEL_KEY"
        exit 1
    fi
done

echo "========================================="
echo "All Collections Complete!"
echo "========================================="
echo ""

# Upload to B2
echo "========================================="
echo "Uploading to Backblaze B2"
echo "========================================="
echo ""

# Check B2 credentials
B2_CONFIG="configs/b2-configs.txt"

if [ ! -f "$B2_CONFIG" ]; then
    echo "✗ ERROR: B2 config file not found: $B2_CONFIG"
    echo "Please copy it from local machine:"
    echo "  scp configs/b2-configs.txt ai_inst:$REPO_DIR/configs/"
    exit 1
fi

# Install B2 CLI (if not already installed)
if ! command -v b2 &> /dev/null; then
    echo "Installing B2 CLI..."
    pip install b2sdk
fi

# Authorize B2
echo "Authorizing B2..."
source $B2_CONFIG

b2 authorize-account $B2_APPLICATION_KEY_ID $B2_APPLICATION_KEY

# Upload trajectories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
B2_PREFIX="slurm_${TIMESTAMP}"

echo "Uploading trajectories to b2://ml-activations-store/$B2_PREFIX/..."
echo ""

for MODEL_KEY in "${MODELS[@]}"; do
    LOCAL_FILE="data/trajectories/${MODEL_KEY}/logiqa_trajectories_vllm_optimized.h5"
    B2_FILE="${B2_PREFIX}/trajectories/${MODEL_KEY}/logiqa_trajectories_vllm_optimized.h5"

    echo "Uploading $MODEL_KEY..."
    b2 upload-file \
        --noProgress \
        ml-activations-store \
        "$LOCAL_FILE" \
        "$B2_FILE"

    echo "  ✓ Uploaded: b2://ml-activations-store/$B2_FILE"
done

echo ""
echo "========================================="
echo "Upload Complete!"
echo "========================================="
echo ""
echo "B2 location: b2://ml-activations-store/$B2_PREFIX/trajectories/"
echo ""
echo "To download:"
echo "  b2 download-file-by-name ml-activations-store $B2_PREFIX/trajectories/<model>/<file> <local_path>"
echo ""

# Print summary
echo "========================================="
echo "Job Summary"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $START_TIME"
echo "End time: $(date)"
echo ""
echo "Models collected: ${MODELS[@]}"
echo "Output location (cluster): $REPO_DIR/data/trajectories/"
echo "Output location (B2): b2://ml-activations-store/$B2_PREFIX/trajectories/"
echo ""
echo "✓ All done!"
