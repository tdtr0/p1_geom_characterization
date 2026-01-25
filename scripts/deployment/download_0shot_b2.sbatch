#!/bin/bash
#SBATCH --job-name=b2_download
#SBATCH --partition=defq
#SBATCH --nodelist=orion
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/home/ttdo/b2_download_out.txt
#SBATCH --error=/home/ttdo/b2_download_err.txt

# Download 0-shot trajectory data from B2
# This is CPU-only, no GPU needed

echo "=== B2 Download Job Started ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

cd /home/ttdo/maniver/ManiVer
bash scripts/deployment/download_0shot_b2.sh

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"
