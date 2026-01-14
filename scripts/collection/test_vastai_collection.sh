#!/bin/bash
#
# Test Phase 2 Collection on vast.ai (N=2 samples)
#
# This script runs a MINIMAL collection to verify:
# - All dependencies installed
# - GPUs accessible
# - Models load correctly
# - B2 upload works
# - No critical bugs
#
# Usage (on vast.ai instance):
#   cd /workspace/maniver
#   bash scripts/collection/test_vastai_collection.sh

set -e

echo "========================================================================"
echo "TEST RUN: Phase 2 Collection (N=2 samples per task)"
echo "========================================================================"
echo ""
echo "⚠️  This is a TEST RUN with minimal data"
echo "    - 2 samples per task (not 500)"
echo "    - Faster generation (128 tokens max, not 512)"
echo "    - 4 models × 3 tasks = 12 files"
echo "    - Expected duration: ~20-30 minutes"
echo "    - Expected cost: ~$2"
echo ""

# Verify we're on vast.ai (has /workspace)
if [ ! -d "/workspace" ]; then
    echo "Error: Not on vast.ai instance (/workspace not found)"
    echo "This script is meant to run on vast.ai, not locally"
    exit 1
fi

cd /workspace/maniver || exit 1

# Check dependencies
echo "========================================================================"
echo "[1/5] Checking Dependencies"
echo "========================================================================"
echo ""

echo "Checking Python packages..."
python3 -c "import torch; import transformers; import h5py; print('✓ Core packages OK')"

echo "Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "✓ Found $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -lt 4 ]; then
    echo "⚠️  Warning: Expected 4 GPUs, found $NUM_GPUS"
    echo "   Collection will still work but may be slower"
fi

echo ""

# Check B2
echo "========================================================================"
echo "[2/5] Checking B2 Configuration"
echo "========================================================================"
echo ""

if ! command -v b2 &> /dev/null; then
    echo "Installing B2 CLI..."
    pip install -q b2
fi

if [ -f "configs/b2-configs.txt" ]; then
    source <(grep -v '^#' configs/b2-configs.txt | sed 's/^/export /')
    b2 authorize-account "$B2_KEY_ID" "$B2_APP_KEY" > /dev/null 2>&1
    echo "✓ B2 authorized"
    echo "  Bucket: $B2_BUCKET_NAME"
else
    echo "⚠️  Warning: configs/b2-configs.txt not found"
    echo "   Upload will not work"
fi

echo ""

# Create test collection script
echo "========================================================================"
echo "[3/5] Creating Test Collection Script"
echo "========================================================================"
echo ""

TEST_SCRIPT="/tmp/test_collection.py"

cat > "$TEST_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Test collection script with N=2 samples
"""
import sys
from pathlib import Path

# Read original script
script_path = Path("scripts/collection/collect_trajectories_with_labels.py")
if not script_path.exists():
    print(f"Error: {script_path} not found")
    sys.exit(1)

content = script_path.read_text()

# Modifications for test run
modifications = {
    "N_SAMPLES = 500": "N_SAMPLES = 2",  # Only 2 samples
    "MAX_NEW_TOKENS = 512": "MAX_NEW_TOKENS = 128",  # Faster generation
    "MAX_SEQ_LEN = 512": "MAX_SEQ_LEN = 256",  # Shorter sequences
    "model_keys = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']":
        "model_keys = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']",  # Keep 4 models
}

for old, new in modifications.items():
    if old in content:
        content = content.replace(old, new)
        print(f"✓ Modified: {old[:50]}...")
    else:
        print(f"⚠️  Warning: Could not find: {old[:50]}...")

# Write test script
test_script = Path("/tmp/test_phase2_collection.py")
test_script.write_text(content)
print(f"\n✓ Test script created: {test_script}")
PYTHON_EOF

python3 "$TEST_SCRIPT"

echo ""

# Run test collection
echo "========================================================================"
echo "[4/5] Running Test Collection"
echo "========================================================================"
echo ""
echo "Starting collection..."
echo "  Models: olmo3_base, olmo3_sft, olmo3_rl_zero, olmo3_think"
echo "  Tasks: gsm8k, logiqa, humaneval"
echo "  Samples: 2 per task"
echo "  Max tokens: 128 (faster than production)"
echo ""
echo "This will take ~20-30 minutes. Progress will be shown below."
echo ""

# Clean any existing test data
rm -rf data/trajectories/* data/checkpoints/* data/logs/*
mkdir -p data/trajectories data/checkpoints data/logs

# Set log file
LOG_FILE="data/logs/test_collection_$(date +%Y%m%d_%H%M%S).log"

# Run collection with logging
python3 /tmp/test_phase2_collection.py 2>&1 | tee "$LOG_FILE"

COLLECTION_EXIT_CODE=${PIPESTATUS[0]}

echo ""

if [ $COLLECTION_EXIT_CODE -ne 0 ]; then
    echo "✗ Collection FAILED with exit code $COLLECTION_EXIT_CODE"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 "$LOG_FILE"
    exit $COLLECTION_EXIT_CODE
fi

echo "✓ Collection completed successfully"
echo ""

# Verify results
echo "========================================================================"
echo "[5/5] Verifying Results"
echo "========================================================================"
echo ""

echo "Files created:"
find data/trajectories -name "*.h5" -exec ls -lh {} \; | tee /tmp/test_files.txt

NUM_FILES=$(find data/trajectories -name "*.h5" | wc -l)
echo ""
echo "Total files: $NUM_FILES (expected: 12)"

if [ "$NUM_FILES" -ne 12 ]; then
    echo "⚠️  Warning: Expected 12 files, found $NUM_FILES"
fi

# Verify one file in detail
echo ""
echo "Verifying olmo3_base/gsm8k_trajectories.h5..."
python3 << 'VERIFY_EOF'
import h5py
import numpy as np
from pathlib import Path

file_path = Path("data/trajectories/olmo3_base/gsm8k_trajectories.h5")
if not file_path.exists():
    print(f"✗ File not found: {file_path}")
    exit(1)

try:
    with h5py.File(file_path, 'r') as f:
        n_samples = len(f['is_correct'])
        n_correct = f['is_correct'][:].sum()
        n_incorrect = (~f['is_correct'][:]).sum()
        shape = f['trajectories'].shape
        has_nan = np.isnan(f['trajectories'][:]).any()

        print(f"✓ File opened successfully")
        print(f"  Samples: {n_samples} (expected: 2)")
        print(f"  Correct: {n_correct}")
        print(f"  Incorrect: {n_incorrect}")
        print(f"  Trajectory shape: {shape}")
        print(f"  Has NaN: {has_nan}")

        if n_samples != 2:
            print(f"⚠️  Warning: Expected 2 samples, found {n_samples}")
        if has_nan:
            print(f"✗ ERROR: Trajectories contain NaN values!")
            exit(1)

        print("\n✓ File verification passed")
except Exception as e:
    print(f"✗ Error verifying file: {e}")
    exit(1)
VERIFY_EOF

VERIFY_EXIT_CODE=$?

echo ""

if [ $VERIFY_EXIT_CODE -ne 0 ]; then
    echo "✗ Verification FAILED"
    exit 1
fi

# Test B2 upload (if configured)
if [ -n "$B2_BUCKET_NAME" ]; then
    echo ""
    echo "Testing B2 upload..."
    TEST_PREFIX="test_vastai_$(date +%Y%m%d_%H%M)"

    python3 scripts/storage/b2_upload.py \
        --local-dir data/trajectories \
        --remote-prefix "$TEST_PREFIX/trajectories"

    UPLOAD_EXIT_CODE=$?

    if [ $UPLOAD_EXIT_CODE -eq 0 ]; then
        echo "✓ B2 upload successful"
        echo "  Location: b2://$B2_BUCKET_NAME/$TEST_PREFIX/trajectories/"

        # Ask to delete test data from B2
        echo ""
        echo "Delete test data from B2? (Saves storage costs)"
        read -p "Delete test files? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            b2 rm "b2://$B2_BUCKET_NAME/$TEST_PREFIX/" --recursive || true
            echo "✓ Test data deleted from B2"
        else
            echo "Test data kept in B2: b2://$B2_BUCKET_NAME/$TEST_PREFIX/"
        fi
    else
        echo "✗ B2 upload FAILED"
        echo "  This needs to be fixed before production run"
    fi
else
    echo ""
    echo "⚠️  Skipping B2 upload test (not configured)"
fi

echo ""
echo "========================================================================"
echo "TEST COMPLETE"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✓ Dependencies OK"
echo "  ✓ GPUs accessible ($NUM_GPUS GPUs)"
echo "  ✓ Collection completed"
echo "  ✓ Files verified (12 HDF5 files)"
if [ -n "$B2_BUCKET_NAME" ] && [ $UPLOAD_EXIT_CODE -eq 0 ]; then
    echo "  ✓ B2 upload works"
fi
echo ""
echo "Next steps:"
echo "  1. Review logs: cat $LOG_FILE"
echo "  2. If everything looks good, proceed with full collection"
echo "  3. Run: bash scripts/collection/run_phase2_pipeline.sh"
echo ""
echo "⚠️  Remember: This was just a TEST with 2 samples"
echo "    Full collection will take 5-11 hours with 500 samples"
echo ""
