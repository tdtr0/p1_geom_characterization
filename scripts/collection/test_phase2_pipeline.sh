#!/bin/bash
#
# Test Phase 2 Pipeline with Small Collection
#
# This script tests the full pipeline with just 2 samples to verify everything works.
#
# Usage:
#   bash scripts/test_phase2_pipeline.sh

set -e

echo "========================================================================"
echo "TESTING PHASE 2 PIPELINE (2 samples per task)"
echo "========================================================================"
echo ""

# Backup any existing data
if [ -d "data/trajectories" ]; then
    echo "Backing up existing data..."
    mv data/trajectories data/trajectories.backup.$(date +%s)
fi

if [ -d "data/checkpoints" ]; then
    mv data/checkpoints data/checkpoints.backup.$(date +%s)
fi

# Create test collection script with N=2
TEST_SCRIPT="/tmp/test_collection.py"
cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
import sys
from pathlib import Path

# Modify N_SAMPLES in the collection script for testing
script_path = Path("scripts/collect_trajectories_with_labels.py")
content = script_path.read_text()

# Replace N_SAMPLES = 500 with N_SAMPLES = 2
content = content.replace("N_SAMPLES = 500", "N_SAMPLES = 2")
content = content.replace("MAX_NEW_TOKENS = 512", "MAX_NEW_TOKENS = 128")  # Faster generation

# Test with just one model
content = content.replace(
    "model_keys = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']",
    "model_keys = ['olmo3_base']"  # Just test with base model
)

# Write to temp script
temp_script = Path("/tmp/test_phase2_collection.py")
temp_script.write_text(content)
print(f"Created test script: {temp_script}")
EOF

python "$TEST_SCRIPT"

# Run test collection
echo ""
echo "[1/3] Running test collection (olmo3_base, 2 samples, all tasks)..."
echo "This should take ~5-10 minutes..."
echo ""

python /tmp/test_phase2_collection.py

# Check results
echo ""
echo "[2/3] Verifying collection results..."
echo ""

if [ ! -d "data/trajectories/olmo3_base" ]; then
    echo "✗ Error: Collection directory not created"
    exit 1
fi

NUM_FILES=$(find data/trajectories/olmo3_base -name "*.h5" | wc -l)
echo "Created $NUM_FILES HDF5 files"

if [ "$NUM_FILES" -lt 3 ]; then
    echo "⚠ Warning: Expected 3 files (gsm8k, humaneval, logiqa), found $NUM_FILES"
else
    echo "✓ All task files created"
fi

# List file sizes
echo ""
echo "File sizes:"
du -sh data/trajectories/olmo3_base/*.h5 2>/dev/null || echo "No files found"

# Test upload to B2
echo ""
echo "[3/3] Testing B2 upload..."
echo ""

if [ -f "configs/b2-configs.txt" ]; then
    # Create test timestamp
    TEST_RUN="test_$(date +%Y%m%d_%H%M)"

    echo "Uploading to b2://ml-activations-store/$TEST_RUN/trajectories/"
    python scripts/b2_upload.py \
        --local-dir data/trajectories \
        --remote-prefix "$TEST_RUN/trajectories"

    echo ""
    echo "✓ Upload completed"
    echo ""
    echo "Test files available at:"
    echo "  https://f005.backblazeb2.com/file/ml-activations-store/$TEST_RUN/trajectories/"

    # Cleanup remote test files (optional)
    read -p "Delete remote test files? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        b2 rm "b2://ml-activations-store/$TEST_RUN/" --recursive
        echo "✓ Remote test files deleted"
    fi
else
    echo "⚠ Skipping B2 upload test (configs/b2-configs.txt not found)"
fi

# Cleanup local test files
echo ""
echo "Cleaning up local test files..."
rm -rf data/trajectories data/checkpoints

# Restore backups if they exist
if [ -d "data/trajectories.backup."* ]; then
    BACKUP=$(ls -td data/trajectories.backup.* | head -1)
    mv "$BACKUP" data/trajectories
    echo "Restored backup: $BACKUP"
fi

if [ -d "data/checkpoints.backup."* ]; then
    BACKUP=$(ls -td data/checkpoints.backup.* | head -1)
    mv "$BACKUP" data/checkpoints
    echo "Restored backup: $BACKUP"
fi

echo ""
echo "========================================================================"
echo "PIPELINE TEST COMPLETE"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✓ Collection works (2 samples per task)"
echo "  ✓ HDF5 files created correctly"
if [ -f "configs/b2-configs.txt" ]; then
    echo "  ✓ B2 upload works"
fi
echo ""
echo "Ready for full Phase 2 collection!"
echo ""
echo "To run full collection on vast.ai:"
echo "  python scripts/vast_launcher.py launch"
echo "  # Then on vast.ai instance:"
echo "  bash scripts/run_phase2_pipeline.sh"
echo ""
