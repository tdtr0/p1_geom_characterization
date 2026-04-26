#!/bin/bash
#SBATCH --job-name=vel_pca
#SBATCH --output=/home/ttdo/vel_pca_out.txt
#SBATCH --error=/home/ttdo/vel_pca_err.txt
#SBATCH --time=0-04:00
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=orion

set -e
echo "=== Phase D: Velocity-Space PCA ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

cd /home/$USER/maniver/ManiVer
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniver_env

export PYTHONPATH=/home/$USER/maniver/ManiVer/src:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE

python experiments/generation_dynamics/analyze_generation_dynamics.py \
    --data-dir data/generation_trajectories \
    --output-dir experiments/generation_dynamics/results \
    --phases d

echo ""
echo "=== Done: $(date) ==="
