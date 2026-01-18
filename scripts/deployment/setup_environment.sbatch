#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=/home/<netid>/setup_out.txt
#SBATCH --error=/home/<netid>/setup_err.txt
#SBATCH --time=0-01:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:0

#
# SLURM job to set up Python environment and install dependencies
# This runs on a compute node (NOT login node)
#
# BEFORE SUBMITTING:
#   1. Replace <netid> with your actual network ID in lines 3-4
#

set -e

echo "========================================="
echo "ManiVer Environment Setup"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

WORK_DIR="/home/$USER/maniver"

echo "Working directory: $WORK_DIR"
echo ""

# Activate conda
source /home/$USER/miniconda3/bin/activate maniver_env

# Install dependencies
echo "Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing vLLM..."
pip install vllm

echo ""
echo "Installing HuggingFace and data processing..."
pip install transformers accelerate
pip install datasets
pip install h5py
pip install pyyaml
pip install tqdm
pip install numpy

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Installed packages:"
pip list | grep -E "(torch|vllm|transformers|h5py)"
echo ""
echo "End time: $(date)"
