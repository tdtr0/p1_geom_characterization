#!/bin/bash
#
# One-time setup script for SLURM cluster
# Run this manually BEFORE submitting jobs
#
# Usage:
#   bash setup_slurm_env.sh
#

set -e

echo "========================================="
echo "SLURM Environment Setup for ManiVer"
echo "========================================="
echo ""

# Load modules
echo "Loading modules..."
module load slurm python3

# Storage location - IMPORTANT: Update this based on check_slurm_storage.sh results
# Options:
#   /home/$USER/maniver         (if home quota allows >20GB)
#   /scratch/$USER/maniver      (if scratch is persistent)
#   /work/$USER/maniver         (if work directory exists)
WORK_DIR="/home/$USER/maniver"

echo "Working directory: $WORK_DIR"
echo ""
echo "IMPORTANT: Verify this location has >20GB available space"
echo "Run check_slurm_storage.sh to check quotas"
echo ""

# Create directories
mkdir -p $WORK_DIR
cd $WORK_DIR

# Clone repository (if not already cloned)
if [ ! -d "ManiVer" ]; then
    echo "Cloning repository..."
    git clone https://github.com/yourusername/ManiVer.git
    cd ManiVer
else
    echo "Repository already exists, pulling latest..."
    cd ManiVer
    git pull
fi

# Create conda environment (if not already created)
CONDA_ENV_NAME="maniver_env"

if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists"
else
    echo "Creating conda environment..."
    conda create -n $CONDA_ENV_NAME python=3.10 -y
fi

# Activate environment
echo "Activating conda environment..."
source activate $CONDA_ENV_NAME

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# vLLM (critical for fast generation)
pip install vllm

# HuggingFace and data processing
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
echo "Next steps:"
echo "  1. Edit run_logiqa_slurm.sbatch to set your <netid>"
echo "  2. Submit job: sbatch scripts/deployment/run_logiqa_slurm.sbatch"
echo "  3. Monitor: squeue -u \$USER"
echo "  4. Check output: tail -f ~/logiqa_collection_out.txt"
echo ""
echo "Working directory: $WORK_DIR/ManiVer"
echo "Conda environment: $CONDA_ENV_NAME"
echo ""
