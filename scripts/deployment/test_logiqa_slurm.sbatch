#!/bin/bash
#SBATCH --job-name=logiqa_test
#SBATCH --output=/home/<netid>/logiqa_test_out.txt
#SBATCH --error=/home/<netid>/logiqa_test_err.txt
#SBATCH --time=0-00:30
#SBATCH --mem=60000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=h100

#
# TEST JOB for vLLM LogiQA collection
# Runs 10 samples on 1 model to validate setup (~10 minutes)
#
# BEFORE SUBMITTING:
#   1. Replace <netid> with your actual network ID in lines 3-4
#   2. Run check_slurm_storage.sh to determine storage location
#   3. Update WORK_DIR below based on storage check results
#   4. Copy B2 credentials: cp configs/b2-configs.txt <WORK_DIR>/ManiVer/configs/
#
# Submit with:
#   sbatch scripts/deployment/test_logiqa_slurm.sbatch
#

set -e  # Exit on any error

echo "========================================="
echo "LogiQA Collection TEST (SLURM)"
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
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || {
    echo "ERROR: vLLM not installed!"
    echo "Run setup_slurm_env.sh first"
    exit 1
}
echo ""

# Test configuration
MODEL_KEY="olmo3_sft"  # Just one model for testing
BATCH_SIZE=4
NUM_SAMPLES=10  # Small test

echo "========================================="
echo "TEST Configuration"
echo "========================================="
echo "Model: $MODEL_KEY"
echo "Batch size: $BATCH_SIZE"
echo "Samples: $NUM_SAMPLES (test only)"
echo ""

# Set PYTHONPATH
export PYTHONPATH=$REPO_DIR/src:$PYTHONPATH

# Disable HDF5 file locking (cluster filesystem issues)
export HDF5_USE_FILE_LOCKING=FALSE

# Run collection
echo "Starting collection..."
python scripts/collection/collect_logiqa_vllm_fully_optimized.py \
    $MODEL_KEY \
    --batch-size $BATCH_SIZE \
    --num-samples $NUM_SAMPLES

# Check if output file was created
OUTPUT_FILE="data/trajectories/${MODEL_KEY}/logiqa_trajectories_vllm_optimized.h5"

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ Collection completed successfully!"
    ls -lh "$OUTPUT_FILE"

    # Verify HDF5 file integrity
    echo ""
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
    echo "✗ ERROR: Output file not found: $OUTPUT_FILE"
    exit 1
fi

echo "========================================="
echo "TEST PASSED!"
echo "========================================="
echo ""
echo "Output: $OUTPUT_FILE"
echo ""
echo "Ready for full collection. To run:"
echo "  sbatch scripts/deployment/run_logiqa_slurm.sbatch"
echo ""
echo "Job completed at: $(date)"
