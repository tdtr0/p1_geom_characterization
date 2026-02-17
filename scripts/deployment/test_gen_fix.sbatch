#!/bin/bash
#SBATCH --job-name=test_gen_fix
#SBATCH --output=/home/ttdo/test_gen_fix_out.txt
#SBATCH --error=/home/ttdo/test_gen_fix_err.txt
#SBATCH --time=0-00:30
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=h100

set -e
echo "=== Test: Verify hidden state capture fix ==="
echo "Start: $(date)"

cd /home/$USER/maniver/ManiVer
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniver_env

export PYTHONPATH=/home/$USER/maniver/ManiVer/src:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE

TEST_DIR="/home/$USER/maniver/ManiVer/data/generation_trajectories_test"
rm -rf $TEST_DIR
mkdir -p $TEST_DIR

python scripts/collection/collect_generation_trajectories.py \
    --model olmo3_base \
    --task gsm8k \
    --output-dir $TEST_DIR \
    --checkpoint-dir /tmp/gen_test_ckpt \
    --max-samples 5 \
    --layers-even

echo ""
echo "=== Verifying hidden states ==="
python3 scripts/deployment/verify_gen_data.py $TEST_DIR/olmo3_base/gsm8k_generation.h5

echo ""
echo "=== Done: $(date) ==="
