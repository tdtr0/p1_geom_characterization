# SLURM Quick Start Guide

Fast reference for running LogiQA collection on H100 cluster.

## Setup (One-Time)

```bash
# 1. SSH to cluster
ssh ai_inst

# 2. Check storage quotas
bash scripts/deployment/check_slurm_storage.sh

# 3. Update WORK_DIR in scripts if needed
# Edit these files to change storage location:
#   - scripts/deployment/setup_slurm_env.sh (line 26)
#   - scripts/deployment/test_logiqa_slurm.sbatch (line 33)
#   - scripts/deployment/run_logiqa_slurm.sbatch (line 44)

# 4. Run setup
bash scripts/deployment/setup_slurm_env.sh

# 5. Copy B2 credentials
# From local machine:
scp configs/b2-configs.txt ai_inst:/home/$USER/maniver/ManiVer/configs/

# 6. Edit job files to set <netid>
nano scripts/deployment/test_logiqa_slurm.sbatch  # Lines 3-4
nano scripts/deployment/run_logiqa_slurm.sbatch   # Lines 3-4
```

## Test Run (10 samples, ~10 min)

```bash
# Submit test job
cd /home/$USER/maniver/ManiVer
sbatch scripts/deployment/test_logiqa_slurm.sbatch

# Note the job ID (e.g., 12345)

# Check status
squeue -u $USER

# Monitor progress
bash scripts/deployment/monitor_slurm_job.sh 12345

# Or follow logs live
tail -f ~/logiqa_test_out.txt
```

## Full Run (500 samples × 3 models, ~1.5-2 hrs)

```bash
# Submit full collection
sbatch scripts/deployment/run_logiqa_slurm.sbatch

# Note the job ID

# Monitor
bash scripts/deployment/monitor_slurm_job.sh <job_id>

# Or follow live
tail -f ~/logiqa_collection_out.txt
```

## Monitoring Approach (for Claude)

I'll check on the job periodically by:

1. SSH to ai_inst
2. Run monitor script: `bash scripts/deployment/monitor_slurm_job.sh <job_id>`
3. Report status to you
4. If errors occur, read logs and diagnose

**Monitoring intervals**:
- Test job: Check every 5 minutes
- Full job: Check every 15-30 minutes

## Expected Output

### Test job (10 samples)
```
✓ Collection script started
✓ vLLM model loaded
✓ HuggingFace model loaded
Progress:   olmo3_sft/logiqa: 100%|██████████| 3/3 [00:45<00:00]
✓ TEST JOB PASSED!
```

### Full job (3 models × 500 samples)
```
✓ Collection script started
✓ vLLM model loaded (olmo3_sft)
Progress:   olmo3_sft/logiqa: 100%|██████████| 63/63 [12:15<00:00]
✓ Collection complete, uploading to B2...
✓ Uploaded: olmo3_sft
[Repeat for olmo3_rl_zero and olmo3_think]
✓ JOB COMPLETED SUCCESSFULLY!
```

## After Completion

Data will be:
1. **On cluster**: `/home/$USER/maniver/ManiVer/data/trajectories/`
2. **On B2**: `b2://ml-activations-store/slurm_YYYYMMDD_HHMMSS/trajectories/`

Download from B2:
```bash
# From local machine
python scripts/storage/b2_download.py \
    --remote-prefix slurm_YYYYMMDD_HHMMSS/trajectories \
    --local-dir data/trajectories
```

## Troubleshooting

### Job stuck in queue (PD status)
- h100 node may be busy
- Check: `squeue -p main` to see all jobs
- Wait or remove `--nodelist=h100` to use any available GPU

### Out of memory
- Reduce batch size in job file: `BATCH_SIZE=4` (instead of 8)

### Job failed
```bash
# Check error log
cat ~/logiqa_collection_err.txt

# Common issues:
# - vLLM not installed: Run setup_slurm_env.sh
# - B2 credentials missing: Copy b2-configs.txt
# - Out of disk space: Check with check_slurm_storage.sh
```

### Cancel job
```bash
scancel <job_id>
```

## Storage Cleanup

After successful B2 upload, you can delete cluster data:

```bash
# Verify B2 upload succeeded first!
rm -rf /home/$USER/maniver/ManiVer/data/trajectories/*
```

## File Sizes

| Model | Samples | File Size |
|-------|---------|-----------|
| olmo3_sft | 500 | ~5-6 GB |
| olmo3_rl_zero | 500 | ~5-6 GB |
| olmo3_think | 500 | ~5-6 GB |
| **Total** | **1500** | **~15-18 GB** |

With B2 upload running in parallel, peak storage is ~15-20GB.
