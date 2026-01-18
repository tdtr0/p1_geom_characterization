# Phase 2 Execution Plan - PRODUCTION RUN

**Status**: 🔴 **DRAFT - AWAITING APPROVAL**
**Target**: 4× RTX 5090 on vast.ai
**Cost Estimate**: $18 total (5.5 hours)
**Models**: 4 models (olmo3_base, olmo3_sft, olmo3_rl_zero, olmo3_think)

---

## ⚠️ SAFETY PROTOCOL

**RULE 1**: Never rent instance without explicit user approval
**RULE 2**: Always test on small dataset (N=2) first
**RULE 3**: Monitor logs continuously during collection
**RULE 4**: Destroy instance immediately if errors detected
**RULE 5**: Verify B2 upload before destroying instance

---

## Pre-Flight Checklist

### Local Machine (Before Renting Instance)

- [ ] **B2 CLI authorized**: `b2 get-account-info` shows correct account
- [ ] **vast.ai logged in**: `vastai show user` works
- [ ] **Credits available**: At least $20 in vast.ai account
- [ ] **Git repo up to date**: Latest scripts pushed to GitHub
- [ ] **Test script ready**: `scripts/collection/test_vastai_collection.sh` exists
- [ ] **User approval obtained**: USER SAYS GO

### After Renting Instance (Before Full Collection)

- [ ] **Instance SSH accessible**: `vastai ssh <id>` works
- [ ] **Repository cloned**: `/workspace/maniver` exists
- [ ] **Dependencies installed**: `pip list | grep transformers` shows packages
- [ ] **B2 configured**: `b2 ls b2://ml-activations-store/` works
- [ ] **GPU visible**: `nvidia-smi` shows 4× RTX 5090
- [ ] **Test run passed**: N=2 collection completed successfully
- [ ] **Test upload to B2**: Test data uploaded successfully
- [ ] **Logs visible locally**: Can tail logs from local machine

### During Full Collection

- [ ] **Monitor every 2 hours**: Check logs for errors
- [ ] **Checkpoints saving**: `data/checkpoints/` updating every 25 samples
- [ ] **Correctness rates reasonable**: Not all 0% or 100%
- [ ] **Disk space OK**: At least 100 GB free
- [ ] **GPU utilization high**: >80% GPU usage

### After Collection Complete

- [ ] **All 12 files created**: 4 models × 3 tasks = 12 HDF5 files
- [ ] **B2 upload successful**: All files in B2 bucket
- [ ] **Verify file integrity**: Download one file and check it opens
- [ ] **Logs uploaded to B2**: Logs backed up
- [ ] **Instance destroyed**: No ongoing charges

---

## Execution Steps

### Phase 0: Preparation (Local Machine)

**Duration**: 10 minutes
**Location**: Local machine

```bash
# 1. Verify everything is ready
cd /Users/thanhdo/CascadeProjects/ManiVer/main

# 2. Check B2
b2 get-account-info
b2 ls b2://ml-activations-store/

# 3. Check vast.ai
vastai show user
vastai show instances  # Should be empty or no active instances

# 4. Push latest code to GitHub (so vast.ai can clone it)
git status
git add -A
git commit -m "Phase 2 production run: 4 models on 4× RTX 5090"
git push origin master

# 5. Find best GPU
bash scripts/deployment/find_optimal_gpu.sh
```

**Stop here and get user approval before proceeding!**

---

### Phase 1: Test Run (Small Dataset)

**Duration**: 30 minutes
**Cost**: ~$2 (test only)
**Location**: vast.ai instance

#### Step 1.1: Rent Instance

```bash
# Search for 4× RTX 5090 (or fallback to 4× RTX 3090)
python scripts/deployment/vast_launcher.py search --sort-by cost

# Note the best offer ID, then rent it
# User confirms instance before renting
python scripts/deployment/vast_launcher.py launch
```

**Wait 2-3 minutes for instance to start**

#### Step 1.2: Verify Setup

```bash
# Get instance ID from output
INSTANCE_ID=<from_output>

# SSH into instance
vastai ssh $INSTANCE_ID

# Verify setup
cd /workspace/maniver
ls -la  # Should show all repo files
nvidia-smi  # Should show 4 GPUs
b2 get-account-info  # Should show authorized
```

#### Step 1.3: Run Test Collection (N=2)

```bash
# Run test with N=2 samples per task
bash scripts/collection/test_vastai_collection.sh

# Expected output:
# - 4 models × 3 tasks = 12 HDF5 files
# - Each file ~100 MB (2 samples only)
# - Takes ~20-30 minutes total
# - Logs to data/logs/test_collection_*.log
```

**Monitor from local machine:**
```bash
# In another terminal on local machine
vastai ssh $INSTANCE_ID "tail -f /workspace/maniver/data/logs/test_collection_*.log"
```

#### Step 1.4: Verify Test Results

```bash
# On vast.ai instance
cd /workspace/maniver

# Check files created
find data/trajectories -name "*.h5" -exec ls -lh {} \;

# Should show 12 files, ~100 MB each

# Check one file
python3 << 'EOF'
import h5py
f = h5py.File('data/trajectories/olmo3_base/gsm8k_trajectories.h5', 'r')
print(f"Samples: {len(f['is_correct'])}")
print(f"Correct: {f['is_correct'][:].sum()}")
print(f"Incorrect: {(~f['is_correct'][:]).sum()}")
f.close()
EOF

# Should show: 2 samples with some correct/incorrect split
```

#### Step 1.5: Test B2 Upload

```bash
# Upload test results
python scripts/storage/b2_upload.py \
  --local-dir data/trajectories \
  --remote-prefix test_phase2_$(date +%Y%m%d_%H%M)

# Verify upload
b2 ls b2://ml-activations-store/test_phase2_*/
```

**If test passed**: Proceed to Phase 2
**If test failed**: Debug, fix, destroy instance, start over

---

### Phase 2: Full Production Collection

**Duration**: 5.5 hours (4× RTX 5090) or 11 hours (4× RTX 3090)
**Cost**: $18 (5090) or $15 (3090)
**Location**: Same vast.ai instance OR new instance

#### Step 2.1: Clean Test Data

```bash
# On vast.ai instance
cd /workspace/maniver

# Remove test data
rm -rf data/trajectories/*
rm -rf data/checkpoints/*
rm -rf data/logs/*

# Verify clean
du -sh data/trajectories  # Should show ~0 MB
```

#### Step 2.2: Run Full Collection

```bash
# Start full collection with all 4 models, 500 samples per task
bash scripts/collection/run_phase2_pipeline.sh

# This will:
# 1. Setup B2 CLI (already done from test)
# 2. Run collection (4 models × 3 tasks × 500 samples)
# 3. Auto-upload to B2 with timestamp
# 4. Ask to cleanup local storage
```

**Expected timeline**:
- **Hour 0-1**: Model loading and first 100 samples
- **Hour 1-3**: GSM8K collection (math)
- **Hour 3-4**: LogiQA collection (logic)
- **Hour 4-5.5**: HumanEval collection (code)

#### Step 2.3: Monitor Progress (Local Machine)

```bash
# Terminal 1: Watch logs
vastai ssh $INSTANCE_ID "tail -f /workspace/maniver/data/logs/phase2_collection_*.log"

# Terminal 2: Watch GPU usage
watch -n 30 "vastai ssh $INSTANCE_ID nvidia-smi"

# Terminal 3: Watch disk usage
watch -n 60 "vastai ssh $INSTANCE_ID 'df -h /workspace'"

# Check checkpoints every hour
vastai ssh $INSTANCE_ID "cat /workspace/maniver/data/checkpoints/labeled_olmo3_base_gsm8k.json"
```

**Monitor for**:
- Correctness rates (should be 20-70% depending on model/task)
- No NaN errors
- Disk space not running out (<100 GB free = warning)
- No repeated errors in logs

#### Step 2.4: Collection Complete

**Pipeline will automatically**:
1. ✓ Upload to B2 (timestamped: `phase2_YYYYMMDD_HHMM/`)
2. ✓ Upload checkpoints and logs
3. ? Ask if you want to delete local copies

**You manually verify**:
```bash
# Check B2 upload
b2 ls b2://ml-activations-store/phase2_*/trajectories/

# Should show:
# olmo3_base/gsm8k_trajectories.h5
# olmo3_base/logiqa_trajectories.h5
# olmo3_base/humaneval_trajectories.h5
# ... (12 files total, ~56 GB)

# Verify one file
mkdir -p /tmp/verify
cd /tmp/verify
python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/thanhdo/CascadeProjects/ManiVer/main')
from scripts.storage.b2_download import download_single_file, load_b2_config

config = load_b2_config()
# Download one file to verify
# (Add download code here)
EOF
```

---

### Phase 3: Cleanup and Verification

**Duration**: 10 minutes
**Cost**: $0 (on local machine)

#### Step 3.1: Destroy Instance

```bash
# Destroy instance IMMEDIATELY after B2 upload confirms
python scripts/deployment/vast_launcher.py destroy $INSTANCE_ID

# Verify no instances running
vastai show instances
# Should be empty or all stopped
```

#### Step 3.2: Download and Verify (Local)

```bash
# On local machine
cd /Users/thanhdo/CascadeProjects/ManiVer/main

# List what we collected
python scripts/storage/b2_download.py --list-only

# Download one file to verify integrity
python scripts/storage/b2_download.py \
  --remote-prefix phase2_YYYYMMDD_HHMM/trajectories \
  --local-dir /tmp/phase2_verify \
  --file trajectories/olmo3_base/gsm8k_trajectories.h5

# Verify file
python3 << 'EOF'
import h5py
import numpy as np

f = h5py.File('/tmp/phase2_verify/gsm8k_trajectories.h5', 'r')
print(f"Samples: {len(f['is_correct'])}")
print(f"Correct: {f['is_correct'][:].sum()}")
print(f"Incorrect: {(~f['is_correct'][:]).sum()}")
print(f"Shape: {f['trajectories'].shape}")
print(f"Has NaN: {np.isnan(f['trajectories'][:]).any()}")
f.close()
print("\n✓ File verified successfully!")
EOF
```

#### Step 3.3: Generate Summary

```bash
# Create summary of collection
cat > /tmp/phase2_summary.md << EOF
# Phase 2 Collection Summary

**Date**: $(date)
**Instance**: 4× RTX 5090 (or GPU type used)
**Duration**: X hours
**Cost**: \$XX

## Files Collected

- 4 models × 3 tasks = 12 HDF5 files
- Total size: ~56 GB
- Location: b2://ml-activations-store/phase2_YYYYMMDD_HHMM/

## Correctness Rates

| Model | GSM8K | LogiQA | HumanEval |
|-------|-------|--------|-----------|
| olmo3_base | XX% | XX% | XX% |
| olmo3_sft | XX% | XX% | XX% |
| olmo3_rl_zero | XX% | XX% | XX% |
| olmo3_think | XX% | XX% | XX% |

## Next Steps

1. Compute path signatures (Week 3)
2. Test H1 hypothesis (Week 4)
3. If H1 succeeds, proceed to H2 cross-domain test
EOF

cat /tmp/phase2_summary.md
```

---

## Failure Scenarios and Recovery

### Scenario 1: Collection Fails Mid-Run

**Symptoms**: Script crashes, errors in logs, NaN values

**Recovery**:
1. Don't destroy instance yet
2. Check logs: `cat data/logs/phase2_collection_*.log | tail -100`
3. Check last checkpoint: `cat data/checkpoints/labeled_*_*.json`
4. Fix issue (update code, push to GitHub, git pull on instance)
5. Re-run pipeline (will resume from checkpoint)

### Scenario 2: B2 Upload Fails

**Symptoms**: Upload errors, connection timeouts

**Recovery**:
1. Keep instance alive
2. Re-authorize B2: `bash scripts/storage/setup_b2_on_vastai.sh`
3. Retry upload: `python scripts/storage/b2_upload.py`
4. Verify: `b2 ls b2://ml-activations-store/phase2_*/`

### Scenario 3: Out of Disk Space

**Symptoms**: "No space left on device" errors

**Recovery**:
1. Check usage: `df -h /workspace`
2. Upload existing data to B2
3. Delete uploaded data: `rm -rf data/trajectories/*`
4. Resume collection (from checkpoint)

### Scenario 4: GPU Out of Memory

**Symptoms**: CUDA OOM errors

**Recovery**:
1. Check which model failed (from logs)
2. This shouldn't happen with 4× 24GB GPUs for 7B models
3. If it does, reduce MAX_SEQ_LEN or MAX_NEW_TOKENS in collection script
4. Re-run

---

## Cost Breakdown

### Test Run (N=2)
- Duration: 30 minutes
- Cost: ~$2
- Purpose: Verify everything works

### Production Run (N=500)
- Duration: 5.5 hours (4× RTX 5090) or 11 hours (4× RTX 3090)
- Cost: $18 (5090) or $15 (3090)
- Purpose: Full data collection

### Total Expected
- **Best case**: $20 (test + production on 5090)
- **Worst case**: $25 (if test fails, need to retry)

---

## Sign-Off

**Prepared by**: Claude (AI Assistant)
**Reviewed by**: [Awaiting User Review]
**Approved by**: [Awaiting User Approval]

**Status**: 🔴 **DO NOT EXECUTE - AWAITING APPROVAL**

---

## User Approval

Before proceeding, user must confirm:

- [ ] I have reviewed the execution plan
- [ ] I understand the costs ($18-25 total)
- [ ] I have $20+ credits in vast.ai account
- [ ] B2 is configured correctly
- [ ] Git repo is up to date
- [ ] I approve test run (N=2, ~$2)
- [ ] I will monitor logs during collection
- [ ] I understand to destroy instance when done

**User signature**: ________________________
**Date**: ________________________

**After approval, proceed to Phase 0.**
