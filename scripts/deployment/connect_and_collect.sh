#!/bin/bash
# Direct connection script - run this in your terminal

INSTANCE_ID="30008731"
SSH_HOST="ssh9.vast.ai"
SSH_PORT="18730"
SSH_KEY="~/.ssh/id_ed25519"

echo "=========================================="
echo "Connecting to vast.ai instance..."
echo "Instance: $INSTANCE_ID"
echo "=========================================="
echo ""

# Try connecting (keep retrying for 2 minutes if SSH key not ready)
MAX_ATTEMPTS=24
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "Attempt $ATTEMPT/$MAX_ATTEMPTS: Connecting..."

    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i $SSH_KEY root@$SSH_HOST -p $SSH_PORT "echo 'Connected!'" 2>/dev/null; then
        echo "✓ SSH connection successful!"
        break
    fi

    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        echo "ERROR: Could not connect after $MAX_ATTEMPTS attempts"
        echo "The SSH key may not be properly configured."
        echo ""
        echo "Try connecting manually:"
        echo "  ssh -i $SSH_KEY root@$SSH_HOST -p $SSH_PORT"
        exit 1
    fi

    echo "  Connection failed, retrying in 5 seconds..."
    sleep 5
    ATTEMPT=$((ATTEMPT + 1))
done

echo ""
echo "=========================================="
echo "Connected! Starting Phase 2 collection..."
echo "=========================================="
echo ""

# Execute the full pipeline on the remote instance
ssh -o StrictHostKeyChecking=no -i $SSH_KEY root@$SSH_HOST -p $SSH_PORT << 'ENDSSH'
set -e

cd /workspace/maniver

echo "=== Checking workspace setup ==="
pwd
ls -la

echo ""
echo "=== Verifying GPUs ==="
nvidia-smi -L

echo ""
echo "=== Checking B2 CLI ==="
b2 get-account-info | head -5

echo ""
echo "=========================================="
echo "Starting TEST collection (N=2 samples)"
echo "This will take ~30 minutes and cost ~$1"
echo "=========================================="
echo ""

bash scripts/collection/test_vastai_collection.sh

echo ""
echo "=========================================="
echo "Test collection complete!"
echo "=========================================="
echo ""
echo "Checking results..."
ls -lh data/trajectories/
echo ""
tail -50 data/logs/test_collection_*.log | grep -E '(PASSED|ERROR|Files created)'

echo ""
echo "=========================================="
echo "Do you want to proceed with FULL production?"
echo "  - 500 samples × 3 tasks × 4 models"
echo "  - ~5.5 hours"
echo "  - ~$10"
echo "=========================================="
read -p "Continue with full production? [y/N]: " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Production cancelled. Test data remains in data/trajectories/"
    echo ""
    echo "To destroy instance and stop charges:"
    echo "  vastai destroy instance 30008731 --api-key \$(cat ~/.config/vastai/vast_api_key)"
    exit 0
fi

echo ""
echo "=========================================="
echo "Starting FULL production collection"
echo "=========================================="
echo ""

# Clean test data
rm -rf data/trajectories/* data/checkpoints/* data/logs/*

# Run full pipeline
bash scripts/collection/run_phase2_pipeline.sh

echo ""
echo "=========================================="
echo "COLLECTION COMPLETE!"
echo "=========================================="
echo ""

ENDSSH

echo ""
echo "=========================================="
echo "Collection finished on vast.ai!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Verify B2 upload: python scripts/storage/b2_download.py --list-only"
echo "  2. Destroy instance: vastai destroy instance $INSTANCE_ID --api-key \$(cat ~/.config/vastai/vast_api_key)"
echo ""
