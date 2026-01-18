#!/bin/bash
set -e

# ManiVer Phase 2 Collection on vast.ai
# Instance: 30008731
# SSH: root@ssh9.vast.ai:18730

INSTANCE_ID="30008731"
SSH_HOST="ssh9.vast.ai"
SSH_PORT="18730"
WORKSPACE="/workspace/maniver"

echo "=========================================="
echo "ManiVer Phase 2 Collection Automation"
echo "=========================================="
echo "Instance: $INSTANCE_ID"
echo "SSH: root@$SSH_HOST:$SSH_PORT"
echo ""

# Function to run remote command
run_remote() {
    ssh -o StrictHostKeyChecking=no root@$SSH_HOST -p $SSH_PORT "$@"
}

# Step 1: Verify instance is ready
echo "[Step 1/6] Checking instance status..."
run_remote "echo 'Instance connected!' && cd $WORKSPACE && echo 'Workspace ready!'"
if [ $? -ne 0 ]; then
    echo "ERROR: Cannot connect to instance or workspace not ready"
    echo "Wait 2-3 minutes for setup to complete, then try again"
    exit 1
fi

# Step 2: Verify GPU
echo ""
echo "[Step 2/6] Verifying GPUs..."
run_remote "nvidia-smi -L"

# Step 3: Verify B2 setup
echo ""
echo "[Step 3/6] Verifying B2 CLI setup..."
run_remote "cd $WORKSPACE && b2 get-account-info | head -5"

# Step 4: Run test collection (N=2, ~30 mins, ~$1)
echo ""
echo "[Step 4/6] Running test collection (N=2 samples)..."
echo "This will take ~30 minutes and cost ~\$1"
read -p "Proceed with test? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Test cancelled"
    exit 1
fi

run_remote "cd $WORKSPACE && bash scripts/collection/test_vastai_collection.sh"

# Step 5: Check test results
echo ""
echo "[Step 5/6] Checking test results..."
run_remote "cd $WORKSPACE && ls -lh data/trajectories/ && echo '' && tail -50 data/logs/test_collection_*.log | grep -E '(ERROR|PASSED|Files created|B2 upload)'"

echo ""
echo "Test results summary:"
run_remote "cd $WORKSPACE && ls data/trajectories/*.h5 2>/dev/null | wc -l | xargs echo 'HDF5 files created:'"

# Ask to proceed with full collection
echo ""
echo "[Step 6/6] Ready for full production collection"
echo "This will:"
echo "  - Collect 500 samples × 3 tasks × 4 models = 6,000 samples"
echo "  - Take ~5.5 hours"
echo "  - Cost ~\$10"
echo "  - Auto-upload to B2"
echo ""
read -p "Proceed with FULL production run? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Production run cancelled"
    echo ""
    echo "To destroy instance and stop charges:"
    echo "  vastai destroy instance $INSTANCE_ID --api-key \$(cat ~/.config/vastai/vast_api_key)"
    exit 0
fi

# Clean test data and run production
echo ""
echo "Starting FULL production collection..."
run_remote "cd $WORKSPACE && rm -rf data/trajectories/* data/checkpoints/* data/logs/* && bash scripts/collection/run_phase2_pipeline.sh"

echo ""
echo "=========================================="
echo "Collection complete!"
echo "=========================================="
echo ""
echo "To verify B2 upload:"
echo "  python scripts/storage/b2_download.py --list-only"
echo ""
echo "To destroy instance:"
echo "  vastai destroy instance $INSTANCE_ID --api-key \$(cat ~/.config/vastai/vast_api_key)"
echo ""
