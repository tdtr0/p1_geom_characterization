#!/bin/bash
# Quick status check for Phase 2 collection
# Usage: bash check_collection_status.sh

SSH_CMD="ssh -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p 7116 root@24.39.63.70"

echo "========================================"
echo "Phase 2 Collection Status - $(date)"
echo "========================================"
echo ""

# GPU Status
echo "GPU Status:"
$SSH_CMD "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null

echo ""
echo "Running Processes:"
$SSH_CMD "ps aux | grep 'collect_traj' | grep -v grep | awk '{print \"  PID:\", \$2, \"CPU:\", \$3\"%\", \"Model:\", \$NF}'" 2>/dev/null

echo ""
echo "File Progress:"
$SSH_CMD "find /workspace/maniver/data/trajectories -name '*.h5' -exec ls -lh {} \; 2>/dev/null | awk '{print \"  \", \$NF, \$5}'" 2>/dev/null

echo ""
echo "Total Size:"
$SSH_CMD "du -sh /workspace/maniver/data/trajectories 2>/dev/null" 2>/dev/null

echo ""
echo "========================================"
