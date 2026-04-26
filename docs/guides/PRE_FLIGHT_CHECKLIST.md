# Pre-Flight Checklist - Phase 2 Production Run

**Date**: _______________
**Operator**: _______________

---

## 🔴 STOP - Read This First

**This checklist MUST be completed before renting any vast.ai instance.**

Estimated cost: $18-25
Estimated time: 6-12 hours
Risk: Low (if checklist followed)

---

## Part 1: Local Machine Setup

### 1.1 B2 Storage

- [ ] **B2 CLI installed**: `which b2` shows path
- [ ] **B2 authorized**: `b2 get-account-info` shows account ID
- [ ] **B2 bucket accessible**: `b2 ls b2://ml-activations-store/` works
- [ ] **Credentials file exists**: `cat configs/b2-configs.txt` shows key
- [ ] **Free space in B2**: Have room for 56 GB upload

**Test command**:
```bash
b2 get-account-info && b2 ls b2://ml-activations-store/
```

**Expected**: Shows account info and bucket listing (may be empty)

---

### 1.2 vast.ai Setup

- [ ] **vast.ai CLI installed**: `which vastai` shows path
- [ ] **vast.ai logged in**: `vastai show user` shows user info
- [ ] **Credits available**: At least $25 in account
- [ ] **No running instances**: `vastai show instances` is empty OR all stopped

**Test command**:
```bash
vastai show user && vastai show instances
```

**Expected**: Shows your username and no active instances

---

### 1.3 Git Repository

- [ ] **All changes committed**: `git status` shows "nothing to commit"
- [ ] **Pushed to GitHub**: `git push origin master` is up to date
- [ ] **Test scripts exist**:
  - `scripts/collection/test_vastai_collection.sh` ✓
  - `scripts/collection/run_phase2_pipeline.sh` ✓
  - `scripts/storage/b2_upload.py` ✓

**Test command**:
```bash
git status && ls scripts/collection/test_vastai_collection.sh
```

**Expected**: "nothing to commit, working tree clean" and file exists

---

### 1.4 Model Configuration

- [ ] **4 models configured**: `grep "model_keys" scripts/collection/collect_trajectories_with_labels.py`
- [ ] **Shows**: `['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']`
- [ ] **NOT deepseek_r1** (we're only using 4 models for this run)

**Test command**:
```bash
grep "model_keys = " scripts/collection/collect_trajectories_with_labels.py
```

**Expected**: Shows exactly 4 models in list

---

## Part 2: Find Best GPU

### 2.1 Search for Optimal GPU

- [ ] **Run GPU finder**: `bash scripts/deployment/find_optimal_gpu.sh`
- [ ] **Found 4× RTX 5090 OR 4× RTX 3090**: At least one multi-GPU option available
- [ ] **Price acceptable**: <$4/hr for 4× setup
- [ ] **Reliability >95%**: Shown in search results

**Test command**:
```bash
bash scripts/deployment/find_optimal_gpu.sh
```

**Expected**: Shows available GPUs with prices

---

### 2.2 Select Instance

Write down the selected instance details:

- **GPU Type**: ________________ (e.g., 4× RTX 5090)
- **Offer ID**: ________________ (from search results)
- **$/hour**: ________________
- **Expected duration**: ________________ hours
- **Expected cost**: $________________

**Approval**: I confirm the above costs are acceptable [ ]

---

## Part 3: Before Renting Instance

### 3.1 Final Verification

- [ ] **All Part 1 items checked**: Go back and verify
- [ ] **All Part 2 items checked**: Go back and verify
- [ ] **Read execution plan**: `PHASE2_EXECUTION_PLAN.md` reviewed
- [ ] **Know how to destroy**: `python scripts/deployment/vast_launcher.py destroy <id>`
- [ ] **Have 6-12 hours free**: To monitor collection

**Critical**: Do NOT rent instance if any item above is unchecked!

---

### 3.2 Emergency Contacts

**If something goes wrong**:
1. Check logs (see monitoring section)
2. DO NOT PANIC - data is checkpointed
3. Can always destroy instance and retry
4. Checkpoints allow resuming from where you left off

**Destroy command** (memorize this):
```bash
python scripts/deployment/vast_launcher.py destroy <instance_id>
```

---

## Part 4: Rent Instance (MONEY STARTS HERE)

### 4.1 Launch Instance

- [ ] **Confirmed all above items**: Triple-checked
- [ ] **Ready to monitor**: Have terminals ready
- [ ] **Launch command ready**:
  ```bash
  python scripts/deployment/vast_launcher.py launch --sort-by cost
  ```

**ACTION**: Run launch command now

```bash
python scripts/deployment/vast_launcher.py launch --sort-by cost
```

- [ ] **Note instance ID**: _________________
- [ ] **Note SSH command**: _________________

**Wait 2-3 minutes for instance to start...**

---

### 4.2 Verify Instance Setup

SSH into instance:
```bash
vastai ssh <instance_id>
```

**On vast.ai instance**, run these checks:

- [ ] **Repository cloned**: `ls /workspace/maniver/` shows files
- [ ] **In correct directory**: `pwd` shows `/workspace/maniver`
- [ ] **GPUs visible**: `nvidia-smi` shows all GPUs
- [ ] **Dependencies installed**: `python3 -c "import torch; import transformers"`
- [ ] **B2 configured**: `b2 get-account-info` works

**Test commands** (on vast.ai):
```bash
cd /workspace/maniver
nvidia-smi
python3 -c "import torch; import transformers; import h5py; print('OK')"
b2 get-account-info
```

**Expected**: All commands succeed, no errors

---

## Part 5: Test Run (N=2)

### 5.1 Run Test

**On vast.ai instance**:

- [ ] **Start test**: `bash scripts/collection/test_vastai_collection.sh`
- [ ] **Monitor on local**: Open new terminal, tail logs
- [ ] **Test duration**: ~20-30 minutes

**Monitor command** (from local machine):
```bash
vastai ssh <instance_id> "tail -f /workspace/maniver/data/logs/test_collection_*.log"
```

---

### 5.2 Test Results

- [ ] **Test completed**: No fatal errors
- [ ] **12 files created**: `find /workspace/maniver/data/trajectories -name "*.h5" | wc -l` shows 12
- [ ] **No NaN values**: Verification script passed
- [ ] **B2 upload worked**: Test data uploaded successfully
- [ ] **Correctness rates reasonable**: Not all 0% or 100%

**If test failed**:
- [ ] Debug issue
- [ ] Fix code locally, push to GitHub
- [ ] `cd /workspace/maniver && git pull`
- [ ] Re-run test

**If test passed**: Proceed to Part 6

---

## Part 6: Full Production Run

### 6.1 Clean Test Data

**On vast.ai instance**:

- [ ] **Clean test data**:
  ```bash
  cd /workspace/maniver
  rm -rf data/trajectories/*
  rm -rf data/checkpoints/*
  rm -rf data/logs/*
  ```
- [ ] **Verify clean**: `du -sh data/trajectories` shows ~0 MB

---

### 6.2 Start Production Collection

- [ ] **Run pipeline**:
  ```bash
  bash scripts/collection/run_phase2_pipeline.sh
  ```
- [ ] **Note start time**: _________________
- [ ] **Expected end time**: _________________ (start + 5-11 hours)

---

### 6.3 Monitoring Setup (Local Machine)

Open 3 terminals on local machine:

**Terminal 1 - Logs**:
```bash
vastai ssh <instance_id> "tail -f /workspace/maniver/data/logs/phase2_collection_*.log"
```

**Terminal 2 - GPU Usage**:
```bash
watch -n 30 "vastai ssh <instance_id> nvidia-smi"
```

**Terminal 3 - Disk Space**:
```bash
watch -n 60 "vastai ssh <instance_id> 'df -h /workspace'"
```

- [ ] **All monitors running**: 3 terminals active

---

### 6.4 Hourly Checks

**Every 2 hours, check**:

- [ ] **Still running**: Logs are updating
- [ ] **No errors**: No repeated error messages
- [ ] **Checkpoints saving**: Files in `data/checkpoints/` updating
- [ ] **Correctness rates OK**: Between 10-80% (varies by model/task)
- [ ] **Disk space OK**: >100 GB free
- [ ] **GPU utilization high**: >80% in nvidia-smi

**Check command** (on vast.ai):
```bash
cd /workspace/maniver
ls -lth data/checkpoints/ | head -5
cat data/checkpoints/labeled_olmo3_base_gsm8k.json
```

---

## Part 7: Collection Complete

### 7.1 Verify Upload

- [ ] **Pipeline finished**: Final "COMPLETE" message shown
- [ ] **B2 upload started**: Shows "Uploading to B2..." in logs
- [ ] **Upload completed**: "✓ Upload completed successfully"
- [ ] **Note B2 location**: `phase2_YYYYMMDD_HHMM/` prefix recorded

**B2 Location**: _________________________________

---

### 7.2 Verify Files in B2

**On local machine**:

```bash
python scripts/storage/b2_download.py --list-only
```

- [ ] **12 HDF5 files visible**: 4 models × 3 tasks
- [ ] **Total ~56 GB**: Reasonable file sizes
- [ ] **Checkpoints uploaded**: In B2 under same prefix
- [ ] **Logs uploaded**: In B2 under same prefix

---

### 7.3 DESTROY INSTANCE

**⚠️ CRITICAL - DO NOT FORGET**:

- [ ] **Destroy NOW**: `python scripts/deployment/vast_launcher.py destroy <instance_id>`
- [ ] **Verify destroyed**: `vastai show instances` shows no active instances
- [ ] **Note end time**: _________________
- [ ] **Calculate actual cost**: _________________ hours × $/hr = $_________________

**Destroy command**:
```bash
python scripts/deployment/vast_launcher.py destroy <instance_id>
```

---

## Part 8: Final Verification

### 8.1 Download and Verify One File

**On local machine**:

```bash
# Download one file to verify
python scripts/storage/b2_download.py \
  --remote-prefix phase2_YYYYMMDD_HHMM/trajectories \
  --local-dir /tmp/phase2_verify \
  --file trajectories/olmo3_base/gsm8k_trajectories.h5

# Verify
python3 -c "
import h5py
f = h5py.File('/tmp/phase2_verify/gsm8k_trajectories.h5', 'r')
print(f'Samples: {len(f[\"is_correct\"])}')
print(f'Correct: {f[\"is_correct\"][:].sum()}')
print(f'Shape: {f[\"trajectories\"].shape}')
f.close()
"
```

- [ ] **File downloads**: No errors
- [ ] **File opens**: h5py can read it
- [ ] **500 samples**: Correct count
- [ ] **No NaN**: Data looks good

---

### 8.2 Generate Summary

- [ ] **Total cost**: $_________________
- [ ] **Total duration**: _________________ hours
- [ ] **Files collected**: 12 / 12 ✓
- [ ] **Ready for analysis**: YES / NO

---

## Sign-Off

**Phase 2 collection**: COMPLETE ✓

**Prepared by**: _________________
**Date**: _________________
**Time**: _________________

**Next steps**:
1. [ ] Compute path signatures (Week 3)
2. [ ] Test H1 hypothesis (Week 4)
3. [ ] Update project status in claude.md

---

## Notes / Issues Encountered

_________________________________________________________________________

_________________________________________________________________________

_________________________________________________________________________

_________________________________________________________________________
