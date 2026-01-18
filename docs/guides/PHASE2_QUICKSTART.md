# Phase 2 Quick Start Guide

**🔴 Status: READY FOR EXECUTION** (pending user approval)

---

## TL;DR - What We're Doing

**Goal**: Collect 6,000 trajectory samples (500 × 3 tasks × 4 models)
**GPU**: 4× RTX 5090 (or 4× RTX 3090 fallback)
**Duration**: 5-11 hours
**Cost**: $15-18
**Safety**: Test first (N=2, $2), then full run

---

## Three-Command Execution

### 1. Pre-Flight Check (Local Machine - 5 mins)

```bash
# Open PRE_FLIGHT_CHECKLIST.md and verify all items
cat PRE_FLIGHT_CHECKLIST.md

# Quick automated check
bash scripts/deployment/find_optimal_gpu.sh

# Push latest code
git push origin master
```

**Stop here until user approves!**

---

### 2. Test Run (vast.ai - 30 mins, $2)

```bash
# Launch instance
python scripts/deployment/vast_launcher.py launch --sort-by cost

# Note instance ID: ________

# SSH in
vastai ssh <instance_id>

# Run test
cd /workspace/maniver
bash scripts/collection/test_vastai_collection.sh

# If test passes, proceed to step 3
# If test fails, fix and retry
```

---

### 3. Full Production Run (vast.ai - 5-11 hrs, $15-18)

```bash
# Still on vast.ai instance
cd /workspace/maniver

# Clean test data
rm -rf data/trajectories/* data/checkpoints/* data/logs/*

# Start production run
bash scripts/collection/run_phase2_pipeline.sh

# Monitor from local machine (new terminal):
vastai ssh <instance_id> "tail -f /workspace/maniver/data/logs/phase2_collection_*.log"

# When done (5-11 hours later):
# - Pipeline auto-uploads to B2
# - You confirm and destroy instance
python scripts/deployment/vast_launcher.py destroy <instance_id>
```

---

## Key Files

| File | Purpose | When to Use |
|------|---------|-------------|
| [PHASE2_EXECUTION_PLAN.md](PHASE2_EXECUTION_PLAN.md) | **Detailed execution plan** | Read before starting |
| [PRE_FLIGHT_CHECKLIST.md](PRE_FLIGHT_CHECKLIST.md) | **Step-by-step checklist** | Fill out before renting |
| [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) | GPU analysis | Understand costs |
| [VASTAI_GUIDE.md](VASTAI_GUIDE.md) | vast.ai setup | First-time setup |

---

## Safety Features

✅ **Test run first** - Only $2, catches bugs before full run
✅ **Checkpointing** - Every 25 samples, can resume if interrupted
✅ **Auto-upload** - Automatic B2 upload when done
✅ **Monitoring** - Real-time log tailing from local machine
✅ **Cost cap** - Destroy instance immediately if issues

---

## What's Being Collected

**Models** (4):
- `olmo3_base` - Baseline model
- `olmo3_sft` - SFT trained
- `olmo3_rl_zero` - Pure RL
- `olmo3_think` - Full pipeline (SFT+DPO+RLVR)

**Tasks** (3):
- `gsm8k` - Math word problems (500 samples)
- `logiqa` - Logical reasoning (500 samples)
- `humaneval` - Python coding (500 samples)

**Output**:
- 12 HDF5 files (4 models × 3 tasks)
- ~56 GB total
- Stored in B2: `phase2_YYYYMMDD_HHMM/trajectories/`

---

## Timeline Breakdown

```
Hour 0    : Rent instance, run test (~30 min)
Hour 0.5  : Start full collection
Hour 1-3  : GSM8K collection (math)
Hour 3-4  : LogiQA collection (logic)
Hour 4-6  : HumanEval collection (code)
Hour 6    : Upload to B2 (~30 min)
Hour 6.5  : Destroy instance, verify
```

**Total**: 6-12 hours depending on GPU

---

## Cost Breakdown

| Item | Time | Cost |
|------|------|------|
| Test run (N=2) | 30 min | ~$2 |
| Full run (4× RTX 5090) | 5.5 hrs | ~$18 |
| Full run (4× RTX 3090) | 11 hrs | ~$15 |
| **Total (best case)** | **6 hrs** | **$20** |
| **Total (expected)** | **6-12 hrs** | **$17-20** |

---

## Monitoring Checklist

**Every 2 hours, check**:

1. **Logs updating**: `tail -f` shows progress
2. **No errors**: No repeated error messages
3. **GPU busy**: >80% utilization in `nvidia-smi`
4. **Disk space**: >100 GB free
5. **Checkpoints**: Files updating in `data/checkpoints/`

**Bad signs** (destroy instance if seen):
- ❌ Logs frozen for >30 minutes
- ❌ Repeated CUDA OOM errors
- ❌ All correctness = 0% or 100%
- ❌ Disk space <50 GB

---

## Failure Recovery

### If Test Fails
1. Check logs: `cat data/logs/test_collection_*.log | tail -100`
2. Fix issue locally, push to GitHub
3. On vast.ai: `git pull`
4. Re-run test

### If Production Fails Mid-Run
1. DON'T destroy instance yet
2. Check checkpoints: `cat data/checkpoints/labeled_*_*.json`
3. Fix issue, push, pull
4. Re-run pipeline (auto-resumes from checkpoint)

### If B2 Upload Fails
1. Keep instance alive
2. Re-authorize: `bash scripts/storage/setup_b2_on_vastai.sh`
3. Re-upload: `python scripts/storage/b2_upload.py`
4. Verify: `b2 ls b2://ml-activations-store/`

---

## Final Verification

After completion:

```bash
# On local machine
python scripts/storage/b2_download.py --list-only

# Should show:
# phase2_YYYYMMDD_HHMM/trajectories/
#   olmo3_base/gsm8k_trajectories.h5 (4.7 GB)
#   olmo3_base/logiqa_trajectories.h5 (4.7 GB)
#   olmo3_base/humaneval_trajectories.h5 (4.7 GB)
#   ... (12 files total, ~56 GB)

# Download one to verify
python scripts/storage/b2_download.py \
  --remote-prefix phase2_YYYYMMDD_HHMM/trajectories \
  --local-dir /tmp/verify \
  --file trajectories/olmo3_base/gsm8k_trajectories.h5

# Check integrity
python3 -c "
import h5py
f = h5py.File('/tmp/verify/gsm8k_trajectories.h5', 'r')
print(f'✓ Samples: {len(f[\"is_correct\"])} (expected 500)')
print(f'✓ Correct: {f[\"is_correct\"][:].sum()}')
print(f'✓ Shape: {f[\"trajectories\"].shape}')
f.close()
"
```

---

## User Approval Required

**Before executing, confirm**:

- [ ] I have reviewed [PHASE2_EXECUTION_PLAN.md](PHASE2_EXECUTION_PLAN.md)
- [ ] I have $20+ credits in vast.ai account
- [ ] I understand the costs ($17-20 total)
- [ ] B2 is configured and working
- [ ] I have 6-12 hours to monitor
- [ ] I know how to destroy instance
- [ ] I approve test run ($2)

**Once approved, proceed to Step 1 above.**

---

## Next Steps After Collection

1. **Verify data quality** (Week 2 end)
2. **Compute path signatures** (Week 3)
3. **Test H1 hypothesis** (Week 4)
   - Within-domain classification
   - Success: >65% accuracy
4. **If H1 succeeds → H2** (Phase 3)
   - Cross-domain transfer
   - This is the critical test!

---

## Quick Reference Commands

```bash
# Find best GPU
bash scripts/deployment/find_optimal_gpu.sh

# Launch instance
python scripts/deployment/vast_launcher.py launch

# SSH
vastai ssh <id>

# Test (on vast.ai)
bash scripts/collection/test_vastai_collection.sh

# Full run (on vast.ai)
bash scripts/collection/run_phase2_pipeline.sh

# Monitor (local)
vastai ssh <id> "tail -f /workspace/maniver/data/logs/*.log"

# Destroy
python scripts/deployment/vast_launcher.py destroy <id>

# Verify
python scripts/storage/b2_download.py --list-only
```

---

**Status**: Ready to execute pending user approval ✓
