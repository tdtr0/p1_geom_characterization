#!/bin/bash
#SBATCH --job-name=gen_analysis
#SBATCH --output=/home/ttdo/gen_analysis_out.txt
#SBATCH --error=/home/ttdo/gen_analysis_err.txt
#SBATCH --time=0-04:00
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=orion

set -e
echo "=== Generation Dynamics Analysis (Phases A+B+C) ==="
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
    --phases abc

echo ""
echo "=== Done: $(date) ==="
