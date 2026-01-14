#!/bin/bash
# Monitor Phase 2 collection progress on vast.ai

INSTANCE_ID="30008731"
SSH_HOST="ssh9.vast.ai"
SSH_PORT="18730"
WORKSPACE="/workspace/maniver"

echo "Monitoring ManiVer Phase 2 Collection"
echo "Instance: $INSTANCE_ID"
echo "Press Ctrl+C to stop monitoring (collection will continue)"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "ManiVer Phase 2 Collection - Live Monitor"
    echo "Time: $(date)"
    echo "=========================================="
    echo ""

    # GPU status
    echo "=== GPU Status ==="
    ssh -o ConnectTimeout=5 root@$SSH_HOST -p $SSH_PORT "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits" 2>/dev/null || echo "Cannot connect to instance"
    echo ""

    # Collection progress
    echo "=== Collection Progress ==="
    ssh -o ConnectTimeout=5 root@$SSH_HOST -p $SSH_PORT "cd $WORKSPACE && find data/trajectories -name '*.h5' 2>/dev/null | wc -l | xargs echo 'HDF5 files created:'" 2>/dev/null
    ssh -o ConnectTimeout=5 root@$SSH_HOST -p $SSH_PORT "cd $WORKSPACE && du -sh data/trajectories 2>/dev/null | awk '{print \"Total size: \" \$1}'" 2>/dev/null
    echo ""

    # Latest log entries
    echo "=== Latest Log Entries (last 10 lines) ==="
    ssh -o ConnectTimeout=5 root@$SSH_HOST -p $SSH_PORT "cd $WORKSPACE && tail -10 data/logs/*.log 2>/dev/null | tail -10" 2>/dev/null || echo "No logs yet"
    echo ""

    # Correctness rates
    echo "=== Correctness Rates ==="
    ssh -o ConnectTimeout=5 root@$SSH_HOST -p $SSH_PORT "cd $WORKSPACE && grep -h 'Correctness rate' data/logs/*.log 2>/dev/null | tail -4" 2>/dev/null || echo "No correctness data yet"
    echo ""

    echo "Refreshing in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done
