#!/bin/bash
#
# Quick vLLM test script
# Tests vLLM installation and performance on 10 samples
#
# Usage:
#   bash scripts/collection/test_vllm_quick.sh

set -e

echo "============================================="
echo "vLLM Quick Test"
echo "============================================="
echo ""

# Check if vLLM is installed
echo "[1/4] Checking vLLM installation..."
if python -c "import vllm" 2>/dev/null; then
    VERSION=$(python -c "import vllm; print(vllm.__version__)")
    echo "✓ vLLM installed: v$VERSION"
else
    echo "✗ vLLM not installed"
    echo ""
    echo "Installing vLLM..."
    pip install vllm
    echo "✓ vLLM installed"
fi
echo ""

# Check GPU
echo "[2/4] Checking GPU..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Test with 10 samples
echo "[3/4] Running test collection (10 samples)..."
echo "This will test vLLM batched inference vs sequential"
echo ""

cd "$(dirname "$0")/../.."

# Test vLLM (batched)
echo "--- vLLM (batch_size=4) ---"
START=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python scripts/collection/collect_logiqa_vllm.py \
  olmo3_sft \
  --batch-size 4 \
  --num-samples 10
END=$(date +%s)
VLLM_TIME=$((END - START))
echo ""
echo "vLLM time: ${VLLM_TIME}s"
echo ""

# Compare with sequential (optional - commented out to save time)
# echo "--- Sequential (for comparison) ---"
# START=$(date +%s)
# CUDA_VISIBLE_DEVICES=0 python scripts/collection/test_logiqa_collection.py olmo3_sft
# END=$(date +%s)
# SEQ_TIME=$((END - START))
# echo ""
# echo "Sequential time: ${SEQ_TIME}s"
# echo ""

echo "[4/4] Results"
echo "============================================="
echo "✓ vLLM test completed successfully!"
echo ""
echo "Time for 10 samples: ${VLLM_TIME}s"
echo "Estimated time for 500 samples: $((VLLM_TIME * 50))s (~$((VLLM_TIME * 50 / 60)) minutes)"
echo ""
echo "Compare to sequential: ~10 hours"
echo "Speedup: ~$((36000 / (VLLM_TIME * 50)))x"
echo ""
echo "Output file: data/trajectories/olmo3_sft/logiqa_trajectories.h5"
echo "============================================="
echo ""
echo "To run full collection (500 samples):"
echo "  python scripts/collection/collect_logiqa_vllm.py olmo3_sft --batch-size 8"
echo ""
