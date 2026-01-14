# Manual Collection Run Instructions

## Instance Details
- **Instance ID**: 30009742
- **SSH**: root@ssh2.vast.ai:19742
- **GPU**: 4× RTX 5090
- **Cost**: $1.43/hr
- **Status**: RUNNING

## Step 1: Connect to Instance

Open a terminal and run:

```bash
ssh -i ~/.ssh/id_ed25519 root@ssh2.vast.ai -p 19742
```

If this fails with "Permission denied", the instance may not have your SSH key yet. Try:

```bash
# Add SSH key to instance
vastai attach ssh 30009742 570124 --api-key $(cat ~/.config/vastai/vast_api_key)

# Reboot instance
vastai reboot instance 30009742 --api-key $(cat ~/.config/vastai/vast_api_key)

# Wait 30 seconds, then try connecting again
sleep 30
ssh -i ~/.ssh/id_ed25519 root@ssh2.vast.ai -p 19742
```

## Step 2: Setup Workspace

Once connected, run:

```bash
# Check GPUs
nvidia-smi -L

# Clone repo
cd /workspace
git clone https://github.com/tdtr0/p1_geom_characterization.git maniver
cd maniver

# Install dependencies
pip install torch transformers datasets h5py scipy scikit-learn tqdm pyyaml b2

# Create directories
mkdir -p data/trajectories data/checkpoints data/logs

# Setup B2 CLI
bash scripts/storage/setup_b2_on_vastai.sh
```

## Step 3: Run TEST Collection

```bash
cd /workspace/maniver
bash scripts/collection/test_vastai_collection.sh
```

This will:
- Collect N=2 samples (all models, all tasks)
- Take ~30 minutes
- Cost ~$0.75
- Verify everything works before full run

## Step 4: Check Test Results

```bash
# List generated files
ls -lh data/trajectories/

# Check logs
tail -100 data/logs/test_collection_*.log

# Look for "PASSED" messages
grep "PASSED" data/logs/test_collection_*.log
```

## Step 5: Run FULL Production Collection

If test passed:

```bash
# Clean test data
rm -rf data/trajectories/* data/checkpoints/* data/logs/*

# Run full collection (500 samples, ~5.5 hrs, ~$8)
bash scripts/collection/run_phase2_pipeline.sh
```

This will:
- Collect 500 samples × 3 tasks × 4 models = 6,000 samples
- Take ~5.5 hours
- Cost ~$8
- Auto-upload to B2

## Step 6: Monitor Progress

In a **separate terminal** on your local machine:

```bash
# Watch logs in real-time
ssh -i ~/.ssh/id_ed25519 root@ssh2.vast.ai -p 19742 "tail -f /workspace/maniver/data/logs/phase2_collection_*.log"

# Or monitor with the script:
cd ~/CascadeProjects/ManiVer/main
./monitor_collection.sh  # (update instance ID inside script to 30009742)
```

## Step 7: Verify and Cleanup

After collection completes:

```bash
# Check B2 upload
python scripts/storage/b2_download.py --list-only

# Destroy instance (IMPORTANT - stops charges!)
vastai destroy instance 30009742 --api-key $(cat ~/.config/vastai/vast_api_key)
```

## Troubleshooting

### SSH Connection Fails
- Make sure instance is "running" (not "loading"):
  `vastai show instances --api-key $(cat ~/.config/vastai/vast_api_key)`
- Try rebooting: `vastai reboot instance 30009742 --api-key $(cat ~/.config/vastai/vast_api_key)`
- Wait 30 seconds after reboot before connecting

### Dependencies Fail to Install
- Check internet connectivity on instance: `ping -c 3 8.8.8.8`
- Try with sudo: `sudo pip install ...`

### B2 Upload Fails
- Verify credentials: `b2 get-account-info`
- Re-run setup: `bash scripts/storage/setup_b2_on_vastai.sh`
- Manual upload: `python scripts/storage/b2_upload.py`

## Cost Tracking

- Test run: ~$0.75 (30 mins)
- Full run: ~$8.00 (5.5 hrs)
- **Total**: ~$8.75

Current meter started at: ~19:30 on 2026-01-14
