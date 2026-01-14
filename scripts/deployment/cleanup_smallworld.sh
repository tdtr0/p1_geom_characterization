#!/bin/bash
# Cleanup SmallWorldMasking directory artifacts
# Expected cleanup: ~38-45 GB

set -e

SMALLWORLD_DIR="$HOME/SmallWorldMasking"

echo "========================================"
echo "SmallWorldMasking Cleanup Script"
echo "========================================"
echo ""
echo "This will delete ~38-45 GB of artifacts:"
echo "  - vast_backups/ (3.9 GB) - old checkpoints"
echo "  - clean/classification/pregenerated_masks/ (5.9 GB) - duplicate masks"
echo "  - clean/saturation_investigation/results/ (5.8 GB) - old investigation"
echo "  - clean/pregenerated_masks/ (7.6 GB) - regenerable mask tensors"
echo "  - clean/final_dit_evals/older runs/ (~15 GB) - Aug-Oct 2025 runs"
echo ""
read -p "Proceed with cleanup? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

cd "$SMALLWORLD_DIR"

echo ""
echo "Starting cleanup..."

# 1. Delete vast_backups (3.9 GB)
if [ -d "vast_backups" ]; then
    echo "Deleting vast_backups/..."
    du -sh vast_backups
    rm -rf vast_backups
    echo "✓ Deleted vast_backups"
fi

# 2. Delete clean/classification/pregenerated_masks (5.9 GB - duplicate)
if [ -d "clean/classification/pregenerated_masks" ]; then
    echo "Deleting clean/classification/pregenerated_masks/..."
    du -sh clean/classification/pregenerated_masks
    rm -rf clean/classification/pregenerated_masks
    echo "✓ Deleted clean/classification/pregenerated_masks"
fi

# 3. Delete clean/saturation_investigation/results (5.8 GB)
if [ -d "clean/saturation_investigation/results" ]; then
    echo "Deleting clean/saturation_investigation/results/..."
    du -sh clean/saturation_investigation/results
    rm -rf clean/saturation_investigation/results
    echo "✓ Deleted clean/saturation_investigation/results"
fi

# 4. Delete clean/pregenerated_masks (7.6 GB - regenerable)
if [ -d "clean/pregenerated_masks" ]; then
    echo "Deleting clean/pregenerated_masks/..."
    du -sh clean/pregenerated_masks
    rm -rf clean/pregenerated_masks
    echo "✓ Deleted clean/pregenerated_masks"
fi

# 5. Delete older final_dit_evals runs (Aug-Oct 2025)
if [ -d "clean/final_dit_evals" ]; then
    echo "Deleting old final_dit_evals runs..."

    # List all runs sorted by date
    cd clean/final_dit_evals

    # Delete runs before 2025-11-01 (keep only Nov-Dec runs)
    for run_dir in run_202508* run_202509* run_202510*; do
        if [ -d "$run_dir" ]; then
            echo "  Deleting $run_dir..."
            du -sh "$run_dir"
            rm -rf "$run_dir"
        fi
    done

    cd "$SMALLWORLD_DIR"
    echo "✓ Deleted old final_dit_evals runs"
fi

# 6. Delete large log files in clean/classification (150 MB)
if [ -d "clean/classification" ]; then
    echo "Deleting large log files..."
    cd clean/classification
    rm -f controls_sweep.log ctrl_sweep.log pa_sweep.log 2>/dev/null || true
    cd "$SMALLWORLD_DIR"
    echo "✓ Deleted large log files"
fi

echo ""
echo "========================================"
echo "Cleanup complete!"
echo "========================================"
echo ""

# Show new disk usage
df -h "$HOME" | grep -E 'Filesystem|/dev/'
echo ""
du -sh "$SMALLWORLD_DIR"
