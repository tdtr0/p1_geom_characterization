#!/bin/bash
#SBATCH --job-name=b2_upload
#SBATCH --output=/home/ttdo/b2_upload_out.txt
#SBATCH --error=/home/ttdo/b2_upload_err.txt
#SBATCH --time=0-02:00
#SBATCH --mem=8000
#SBATCH --nodelist=orion

#
# Upload collected trajectories to Backblaze B2
# This job runs on CPU-only node (no GPU needed)
#
# BEFORE SUBMITTING:
#   1. Ensure B2 config exists at configs/b2-configs.txt
#   2. Collection job must be complete
#
# Submit with:
#   sbatch scripts/deployment/upload_to_b2.sbatch
#
# Monitor:
#   tail -f /home/ttdo/b2_upload_out.txt
#

set -e

echo "========================================="
echo "B2 Upload Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Environment setup
WORK_DIR="/home/ttdo/maniver"
REPO_DIR="$WORK_DIR/ManiVer"

cd $REPO_DIR
source activate maniver_env

# Check B2 config exists
B2_CONFIG="configs/b2-configs.txt"
if [ ! -f "$B2_CONFIG" ]; then
    echo "ERROR: B2 config not found: $B2_CONFIG"
    echo ""
    echo "Please copy it from local machine:"
    echo "  scp configs/b2-configs.txt ai_inst:/home/ttdo/maniver/ManiVer/configs/"
    exit 1
fi

# Check that trajectory files exist
echo "Checking for trajectory files..."
MODELS=("olmo3_sft" "olmo3_rl_zero" "olmo3_think")
for MODEL in "${MODELS[@]}"; do
    FILE="data/trajectories/${MODEL}/logiqa_trajectories_vllm_optimized.h5"
    if [ -f "$FILE" ]; then
        echo "  Found: $FILE ($(ls -lh "$FILE" | awk '{print $5}'))"
    else
        echo "  WARNING: Not found: $FILE"
    fi
done
echo ""

# Install b2 CLI if not available
if ! command -v b2 &> /dev/null; then
    echo "Installing B2 CLI..."
    pip install b2sdk b2
fi

# Load B2 credentials and authorize
echo "Authorizing B2..."
source $B2_CONFIG
b2 authorize-account $B2_KEY_ID $B2_APP_KEY
echo "B2 authorized"
echo ""

# Upload with timestamp prefix
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_PREFIX="logiqa_0shot_recollect_${TIMESTAMP}"

echo "========================================="
echo "Uploading trajectories"
echo "========================================="
echo "Remote prefix: $REMOTE_PREFIX"
echo ""

# Upload each model's trajectory
for MODEL in "${MODELS[@]}"; do
    LOCAL_FILE="data/trajectories/${MODEL}/logiqa_trajectories_vllm_optimized.h5"
    REMOTE_PATH="${REMOTE_PREFIX}/trajectories/${MODEL}/logiqa_trajectories_vllm_optimized.h5"

    if [ -f "$LOCAL_FILE" ]; then
        echo "Uploading: $MODEL"
        echo "  Local: $LOCAL_FILE"
        echo "  Remote: b2://ml-activations-store/$REMOTE_PATH"

        b2 upload-file \
            --noProgress \
            ml-activations-store \
            "$LOCAL_FILE" \
            "$REMOTE_PATH"

        echo "  Done!"
        echo ""
    else
        echo "SKIP: $LOCAL_FILE not found"
    fi
done

echo "========================================="
echo "Upload Complete!"
echo "========================================="
echo "End: $(date)"
echo ""
echo "Files uploaded to: b2://ml-activations-store/$REMOTE_PREFIX/"
echo ""
echo "To download later:"
echo "  b2 download-file-by-name ml-activations-store $REMOTE_PREFIX/trajectories/<model>/logiqa_trajectories_vllm_optimized.h5 <local_path>"
echo ""
