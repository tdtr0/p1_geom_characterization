#!/bin/bash
#
# Find Pareto Optimal GPU on vast.ai for Phase 2 Collection
#
# This script searches vast.ai for the best GPU options based on:
# - Performance per dollar (dlperf_usd)
# - Availability
# - Memory requirements (24GB+ per GPU)
# - Multi-GPU setups (4× preferred)
#
# Usage:
#   bash scripts/deployment/find_optimal_gpu.sh

set -e

echo "========================================================================"
echo "Finding Pareto Optimal GPU for Phase 2 Collection"
echo "========================================================================"
echo ""
echo "Requirements:"
echo "  - Memory: 24+ GB VRAM per GPU"
echo "  - Multi-GPU: 4× preferred (process 1 model per GPU)"
echo "  - Bandwidth: 200+ Mbps up/down"
echo "  - Reliability: >95%"
echo ""

# Check if vastai is installed
if ! command -v vastai &> /dev/null; then
    echo "Error: vastai CLI not found"
    echo "Install with: pip install vastai"
    exit 1
fi

# Check if logged in
if ! vastai show user &> /dev/null; then
    echo "Error: Not logged in to vast.ai"
    echo "Login with: vastai set api-key <your_key>"
    exit 1
fi

echo "Searching vast.ai for optimal GPUs..."
echo ""

# Priority 1: 4× RTX 5090 (best speed/cost ratio)
echo "========================================================================"
echo "[1/6] Searching for 4× RTX 5090 (Best speed/cost ratio)"
echo "========================================================================"
RESULTS_5090=$(vastai search offers "gpu_name = RTX_5090 num_gpus = 4 gpu_ram >= 24 reliability > 0.95 inet_down > 200" --order dlperf_usd 2>/dev/null | head -5)

if [ -n "$RESULTS_5090" ]; then
    echo "$RESULTS_5090"
    echo ""
    echo "✓ Found 4× RTX 5090 options!"
    echo "  Estimated: 5.5 hours, ~$18 total"
    echo "  Recommendation: BEST CHOICE if available <$4/hr"
else
    echo "✗ No 4× RTX 5090 available"
fi
echo ""

# Priority 2: 4× RTX 3090 (best value)
echo "========================================================================"
echo "[2/6] Searching for 4× RTX 3090 (Best value)"
echo "========================================================================"
RESULTS_3090=$(vastai search offers "gpu_name = RTX_3090 num_gpus = 4 gpu_ram >= 24 reliability > 0.95 inet_down > 200" --order dlperf_usd 2>/dev/null | head -5)

if [ -n "$RESULTS_3090" ]; then
    echo "$RESULTS_3090"
    echo ""
    echo "✓ Found 4× RTX 3090 options!"
    echo "  Estimated: 11 hours, ~$15 total"
    echo "  Recommendation: EXCELLENT VALUE - Cheapest multi-GPU option"
else
    echo "✗ No 4× RTX 3090 available"
fi
echo ""

# Priority 3: 4× RTX 4090
echo "========================================================================"
echo "[3/6] Searching for 4× RTX 4090"
echo "========================================================================"
RESULTS_4090=$(vastai search offers "gpu_name = RTX_4090 num_gpus = 4 gpu_ram >= 24 reliability > 0.95 inet_down > 200" --order dlperf_usd 2>/dev/null | head -5)

if [ -n "$RESULTS_4090" ]; then
    echo "$RESULTS_4090"
    echo ""
    echo "✓ Found 4× RTX 4090 options!"
    echo "  Estimated: 8 hours, ~$19 total"
else
    echo "✗ No 4× RTX 4090 available"
fi
echo ""

# Priority 4: 4× A100 40GB
echo "========================================================================"
echo "[4/6] Searching for 4× A100 40GB"
echo "========================================================================"
RESULTS_A100=$(vastai search offers "gpu_name = A100_PCIE_40GB num_gpus = 4 reliability > 0.95 inet_down > 200" --order dlperf_usd 2>/dev/null | head -5)

if [ -n "$RESULTS_A100" ]; then
    echo "$RESULTS_A100"
    echo ""
    echo "✓ Found 4× A100 40GB options!"
    echo "  Estimated: 7.25 hours, ~$38 total"
else
    echo "✗ No 4× A100 40GB available"
fi
echo ""

# Priority 5: Single RTX 5090 (best single GPU)
echo "========================================================================"
echo "[5/6] Searching for Single RTX 5090 (Best single GPU value)"
echo "========================================================================"
RESULTS_5090_SINGLE=$(vastai search offers "gpu_name = RTX_5090 num_gpus = 1 gpu_ram >= 24 reliability > 0.95 inet_down > 200" --order dlperf_usd 2>/dev/null | head -5)

if [ -n "$RESULTS_5090_SINGLE" ]; then
    echo "$RESULTS_5090_SINGLE"
    echo ""
    echo "✓ Found single RTX 5090 options!"
    echo "  Estimated: 22 hours, ~$18 total"
    echo "  Recommendation: Best single GPU option"
else
    echo "✗ No single RTX 5090 available"
fi
echo ""

# Priority 6: 4× H100 (fastest)
echo "========================================================================"
echo "[6/6] Searching for 4× H100 (Fastest)"
echo "========================================================================"
RESULTS_H100=$(vastai search offers "gpu_name = H100 num_gpus = 4 reliability > 0.95 inet_down > 200" --order dlperf_usd 2>/dev/null | head -5)

if [ -n "$RESULTS_H100" ]; then
    echo "$RESULTS_H100"
    echo ""
    echo "✓ Found 4× H100 options!"
    echo "  Estimated: 4.5 hours, ~$54 total"
    echo "  Recommendation: Fastest option (if budget allows)"
else
    echo "✗ No 4× H100 available"
fi
echo ""

# Summary
echo "========================================================================"
echo "RECOMMENDATION SUMMARY"
echo "========================================================================"
echo ""
echo "Based on availability above, choose in this order:"
echo ""
echo "1. 4× RTX 5090    → 5.5 hrs, ~$18   (Best speed/cost)"
echo "2. 4× RTX 3090    → 11 hrs, ~$15    (Best value)"
echo "3. 4× RTX 4090    → 8 hrs, ~$19     (Good middle ground)"
echo "4. Single RTX 5090 → 22 hrs, ~$18   (Best single GPU)"
echo "5. 4× A100 40GB   → 7 hrs, ~$38     (Cloud standard)"
echo "6. 4× H100        → 4.5 hrs, ~$54   (Fastest)"
echo ""
echo "To launch the best available option:"
echo "  python scripts/deployment/vast_launcher.py launch --sort-by cost"
echo ""
echo "For detailed GPU comparison:"
echo "  cat GPU_OPTIMIZATION.md"
echo ""
