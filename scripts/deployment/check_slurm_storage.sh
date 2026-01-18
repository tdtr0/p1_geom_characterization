#!/bin/bash
#
# Check available storage on SLURM cluster
# Run this on the LOGIN NODE to determine where to store data
#
# Usage:
#   bash check_slurm_storage.sh
#

echo "========================================="
echo "SLURM Storage Check"
echo "========================================="
echo ""

# Check home directory quota
echo "Home directory: $HOME"
df -h $HOME
echo ""
quota -s 2>/dev/null || echo "(quota command not available)"
echo ""

# Check if /scratch exists and its size
if [ -d "/scratch" ]; then
    echo "Scratch directory: /scratch/$USER"
    df -h /scratch 2>/dev/null || echo "(/scratch not mounted or accessible)"
    echo ""

    # Check if it's temporary storage
    echo "Checking /scratch retention policy..."
    ls -la /scratch/ 2>/dev/null | head -20
    echo ""
else
    echo "/scratch directory does not exist"
    echo ""
fi

# Check if there's a project/work directory
if [ -d "/work/$USER" ]; then
    echo "Work directory: /work/$USER"
    df -h /work/$USER
    echo ""
fi

if [ -d "/project/$USER" ]; then
    echo "Project directory: /project/$USER"
    df -h /project/$USER
    echo ""
fi

# Estimate space needed
echo "========================================="
echo "Space Requirements"
echo "========================================="
echo ""
echo "Estimated space needed for collection:"
echo "  - 3 models × 500 samples × ~5GB each = ~15GB"
echo "  - With safety margin: ~20GB recommended"
echo ""
echo "After collection, data is uploaded to B2 and can be deleted from cluster."
echo ""

# Recommendations
echo "========================================="
echo "Recommendations"
echo "========================================="
echo ""

HOME_AVAIL=$(df -h $HOME | tail -1 | awk '{print $4}')
echo "Home directory available: $HOME_AVAIL"

if [ -d "/scratch/$USER" ]; then
    SCRATCH_AVAIL=$(df -h /scratch 2>/dev/null | tail -1 | awk '{print $4}')
    echo "Scratch directory available: $SCRATCH_AVAIL"
fi

echo ""
echo "IMPORTANT: Verify with system documentation whether /scratch/"
echo "is temporary (deleted after job ends) or persistent."
echo ""
echo "Safest option: Use /home/$USER/maniver/ if quota allows >20GB"
echo ""
