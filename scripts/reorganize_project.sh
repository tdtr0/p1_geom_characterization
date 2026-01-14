#!/bin/bash
#
# Reorganize ManiVer project structure
#
# This script creates a clean directory structure and moves files accordingly.
#

set -e

echo "========================================================================"
echo "Reorganizing ManiVer Project Structure"
echo "========================================================================"
echo ""

# Create new directory structure
echo "Creating directory structure..."

mkdir -p docs/plans
mkdir -p docs/guides
mkdir -p docs/paper
mkdir -p scripts/collection
mkdir -p scripts/analysis
mkdir -p scripts/storage
mkdir -p scripts/deployment
mkdir -p data/checkpoints
mkdir -p data/logs
mkdir -p data/trajectories

# Move phase plans to docs/plans
echo "Moving phase plans..."
mv -f PHASE*_PLAN.md docs/plans/ 2>/dev/null || true
mv -f PHASE*_DETAILED_PLAN.md docs/plans/ 2>/dev/null || true
mv -f phase1_implementation_plan.md docs/plans/ 2>/dev/null || true
mv -f archive_transfer_correlation_plan.md docs/plans/ 2>/dev/null || true
mv -f master_algorithm.md docs/plans/ 2>/dev/null || true

# Move pipeline and setup docs to docs/guides
echo "Moving setup guides..."
mv -f PHASE2_PIPELINE.md docs/guides/ 2>/dev/null || true
mv -f configs/B2_SETUP.md docs/guides/ 2>/dev/null || true
mv -f scripts/B2_QUICKSTART.md docs/guides/ 2>/dev/null || true

# Move paper materials
echo "Moving paper materials..."
if [ -d "paper" ]; then
    mv paper/* docs/paper/ 2>/dev/null || true
    rmdir paper 2>/dev/null || true
fi

# Move collection scripts
echo "Organizing collection scripts..."
mv -f scripts/collect_*.py scripts/collection/ 2>/dev/null || true
mv -f scripts/run_labeled_collection.sh scripts/collection/ 2>/dev/null || true
mv -f scripts/run_with_restart.sh scripts/collection/ 2>/dev/null || true
mv -f scripts/tmux_collect.sh scripts/collection/ 2>/dev/null || true
mv -f scripts/run_phase2_pipeline.sh scripts/collection/ 2>/dev/null || true
mv -f scripts/test_phase2_pipeline.sh scripts/collection/ 2>/dev/null || true

# Move analysis scripts
echo "Organizing analysis scripts..."
mv -f scripts/run_analysis.py scripts/analysis/ 2>/dev/null || true
mv -f scripts/curvature_and_stats.py scripts/analysis/ 2>/dev/null || true
mv -f scripts/check_layer_smoothness.py scripts/analysis/ 2>/dev/null || true
mv -f scripts/verify_pipeline.py scripts/analysis/ 2>/dev/null || true

# Move storage scripts
echo "Organizing storage scripts..."
mv -f scripts/b2_*.py scripts/storage/ 2>/dev/null || true
mv -f scripts/setup_b2_on_vastai.sh scripts/storage/ 2>/dev/null || true

# Move deployment scripts
echo "Organizing deployment scripts..."
mv -f scripts/vast_launcher.py scripts/deployment/ 2>/dev/null || true
mv -f scripts/cleanup_smallworld.sh scripts/deployment/ 2>/dev/null || true

# Move logs
echo "Moving logs..."
mv -f *.log data/logs/ 2>/dev/null || true
mv -f logs/*.log data/logs/ 2>/dev/null || true
rmdir logs 2>/dev/null || true

# Move checkpoints
echo "Moving checkpoints..."
if [ -d "checkpoints" ]; then
    mv checkpoints/*.json data/checkpoints/ 2>/dev/null || true
    rmdir checkpoints 2>/dev/null || true
fi

# Move GPU cache
echo "Moving GPU cache..."
mv -f gpu_names_cache.json configs/ 2>/dev/null || true

echo ""
echo "✓ Reorganization complete!"
echo ""
echo "New structure:"
echo "  docs/"
echo "    plans/          - Phase plans and algorithm docs"
echo "    guides/         - Setup and usage guides"
echo "    paper/          - Research paper materials"
echo "  scripts/"
echo "    collection/     - Data collection scripts"
echo "    analysis/       - Analysis scripts"
echo "    storage/        - B2 upload/download"
echo "    deployment/     - vast.ai management"
echo "  data/"
echo "    checkpoints/    - Collection checkpoints"
echo "    logs/           - Collection logs"
echo "    trajectories/   - Activation data (gitignored)"
echo ""
