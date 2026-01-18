# SLURM Cluster Guide for LogiQA Collection

Complete guide for running fully optimized vLLM collection on SLURM cluster with H100 GPUs.

## Quick Start (3 Steps)

```bash
# 1. SSH to login node
ssh ai_inst

# 2. One-time setup (clone repo, create conda env, install deps)
bash scripts/deployment/setup_slurm_env.sh

# 3. Submit job (AFTER editing <netid> in sbatch file)
sbatch scripts/deployment/run_logiqa_slurm.sbatch
```

**Total time**: ~1.5-2 hours for 3 models × 500 samples
**Expected speedup**: 8-10x vs sequential collection

---

## Architecture Overview

```
Login Node (ai_inst)           Compute Node (h100)              Backblaze B2
─────────────────────          ───────────────────              ────────────
Submit jobs only               GPU inference + collection        Persistent storage
(NO compute here!)             /scratch/$USER/maniver/          b2://ml-activations-store/
                               ├── vLLM: Fast generation
                               ├── HF: Activation hooks
                               └── Auto-upload to B2 →
```

## Performance Expectations

| Method | GPU Util | Time/Sample | Total (500 samples) |
|--------|----------|-------------|---------------------|
| Sequential | 60-70% | 90s | 12.5 hrs |
| Batched HF | 0-10% ❌ | 40-50s | 5.5 hrs |
| Optimized HF | 70-86% | 20s | 2.8 hrs |
| **vLLM Fully Optimized** | **90-95%** ✅ | **10-12s** | **1.5-2 hrs** |

**Memory usage on H100**:
- vLLM model: 14GB
- HuggingFace model: 14GB
- Activation buffers: 15GB
- **Total**: ~43GB / 80GB available

---

## Detailed Setup Instructions

### Step 1: SSH to Login Node

```bash
ssh ai_inst
```

**⚠️ CRITICAL**: This is a **login node** - you can ONLY:
- Edit files
- Submit jobs (`sbatch`)
- Monitor jobs (`squeue`)

**NEVER** run Python scripts, model inference, or any compute on the login node!

### Step 2: One-Time Environment Setup

Run the setup script (this clones repo, creates conda env, installs dependencies):

```bash
# On login node
cd /scratch/$USER/maniver
bash setup_slurm_env.sh
```

This will:
1. Load `slurm` and `python3` modules
2. Clone ManiVer repository to `/scratch/$USER/maniver/ManiVer`
3. Create conda environment `maniver_env`
4. Install dependencies: PyTorch, vLLM, HuggingFace, h5py, etc.

**Expected time**: 10-15 minutes

### Step 3: Copy B2 Credentials

The job file auto-uploads to B2 after collection. Copy your B2 credentials:

```bash
# On local machine
scp /Users/thanhdo/CascadeProjects/ManiVer/main/configs/b2-configs.txt \
    ai_inst:/scratch/$USER/maniver/ManiVer/configs/
```

Verify:

```bash
# On login node
cat /scratch/$USER/maniver/ManiVer/configs/b2-configs.txt
```

Should contain:
```bash
export B2_APPLICATION_KEY_ID=<your_key_id>
export B2_APPLICATION_KEY=<your_key>
```

### Step 4: Edit SLURM Job File

Edit `scripts/deployment/run_logiqa_slurm.sbatch` and replace `<netid>`:

```bash
cd /scratch/$USER/maniver/ManiVer
nano scripts/deployment/run_logiqa_slurm.sbatch
```

Change lines 3-4:
```bash
#SBATCH --output=/home/<netid>/logiqa_collection_out.txt
#SBATCH --error=/home/<netid>/logiqa_collection_err.txt
```

To (replace with your actual netid):
```bash
#SBATCH --output=/home/yournetid/logiqa_collection_out.txt
#SBATCH --error=/home/yournetid/logiqa_collection_err.txt
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

### Step 5: Submit Job

```bash
sbatch scripts/deployment/run_logiqa_slurm.sbatch
```

You should see:
```
Submitted batch job 12345
```

Take note of the job ID!

---

## Monitoring Jobs

### Check Job Status

```bash
# See all your jobs
squeue -u $USER

# Sample output:
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
12345      main logiqa_v yourname  R       5:23      1 h100
```

**Status codes**:
- `PD` (Pending): Waiting in queue
- `R` (Running): Currently executing
- `CG` (Completing): Finishing up
- `CD` (Completed): Done

### Monitor Progress

```bash
# Watch stdout (collection progress)
tail -f ~/logiqa_collection_out.txt

# Watch stderr (errors)
tail -f ~/logiqa_collection_err.txt
```

Press `Ctrl+C` to stop tailing.

### Check GPU Utilization

If you need to check GPU usage (advanced):

```bash
# SSH to compute node (only if job is running)
ssh h100

# Check GPU
nvidia-smi

# Exit compute node
exit
```

**Note**: Only do this if absolutely necessary - login node is preferred.

---

## Expected Output

### During Collection

You should see (in `~/logiqa_collection_out.txt`):

```
=========================================
LogiQA Collection with vLLM (SLURM)
=========================================
Job ID: 12345
Node: h100
Start time: Sat Jan 18 15:30:00 PST 2026

Working directory: /scratch/yourname/maniver/ManiVer
Conda environment: maniver_env

Checking GPU...
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|   0  H100 80GB       On   | 00000000:3B:00.0 Off |                    0 |
| N/A   32C    P0    68W / 700W |  43000MiB / 81559MiB |     92%      Default |
+-------------------------------+----------------------+----------------------+

Loading models: allenai/OLMo-3-7B-Think-SFT
  Loading vLLM model...
  vLLM loaded: ready for fast generation
  Loading HuggingFace model for activations...
  HF loaded: 32 layers, d_model=4096

=========================================
Collecting: olmo3_sft
=========================================

  olmo3_sft/logiqa: 100%|██████████| 63/63 [01:15<00:00, 12.5s/batch]

✓ Collection completed for olmo3_sft
  Samples: 500
  Correct: 245/500 (49.0%)
  Trajectory shape: (500, 512, 16, 4096)
  ✓ File is valid
```

### After Completion

```
=========================================
Upload Complete!
=========================================

B2 location: b2://ml-activations-store/slurm_20260118_173045/trajectories/

To download:
  b2 download-file-by-name ml-activations-store slurm_20260118_173045/trajectories/<model>/<file> <local_path>

=========================================
Job Summary
=========================================
Job ID: 12345
Node: h100
Start time: Sat Jan 18 15:30:00 PST 2026
End time: Sat Jan 18 17:15:23 PST 2026

Models collected: olmo3_sft olmo3_rl_zero olmo3_think
Output location (cluster): /scratch/yourname/maniver/ManiVer/data/trajectories/
Output location (B2): b2://ml-activations-store/slurm_20260118_173045/trajectories/

✓ All done!
```

---

## Troubleshooting

### Job Won't Start (PD Status)

**Cause**: h100 node is busy or requested resources unavailable.

**Solution**: Wait, or modify job file to allow other nodes:

```bash
# Remove line 7 (--nodelist=h100) to allow any node
#SBATCH --nodelist=h100  # <-- Delete or comment out
```

### Out of Memory Error

**Cause**: Batch size too large for GPU.

**Solution**: Edit job file, reduce `BATCH_SIZE`:

```bash
BATCH_SIZE=4  # Instead of 8
```

### B2 Upload Fails

**Cause**: B2 credentials not found or invalid.

**Solution**:

1. Check credentials exist:
   ```bash
   cat /scratch/$USER/maniver/ManiVer/configs/b2-configs.txt
   ```

2. Copy from local if missing:
   ```bash
   scp configs/b2-configs.txt ai_inst:/scratch/$USER/maniver/ManiVer/configs/
   ```

3. Test authorization:
   ```bash
   source /scratch/$USER/maniver/ManiVer/configs/b2-configs.txt
   b2 authorize-account $B2_APPLICATION_KEY_ID $B2_APPLICATION_KEY
   b2 ls ml-activations-store
   ```

### Collection Hangs or Crashes

**Check logs**:

```bash
# Stdout
tail -100 ~/logiqa_collection_out.txt

# Stderr
tail -100 ~/logiqa_collection_err.txt
```

**Cancel job**:

```bash
scancel <job_id>
```

**Resubmit** after fixing issue.

---

## Performance Optimization

### GPU Bottlenecks Fixed

This script fixes **all 6 bottlenecks** identified during development:

1. ✅ **GPU→CPU transfer**: Tensors stay on GPU until final write
2. ✅ **Sequential forward passes**: Batched activation collection (1 pass vs 8)
3. ✅ **Blocking HDF5 writes**: Async I/O with threading (overlapped with GPU)
4. ✅ **Memory fragmentation**: Explicit cleanup after each batch
5. ✅ **CPU tokenization**: vLLM handles tokenization on GPU
6. ✅ **Slow HF generation**: vLLM is 3-5x faster (10-15s vs 40-50s per batch)

### Expected Timeline

| Model | Samples | Batches | Time per Batch | Total Time |
|-------|---------|---------|----------------|------------|
| olmo3_sft | 500 | 63 | 12-15s | 12-16 min |
| olmo3_rl_zero | 500 | 63 | 12-15s | 12-16 min |
| olmo3_think | 500 | 63 | 12-15s | 12-16 min |

**Total**: ~36-48 minutes collection + ~10-15 minutes upload = **1.5-2 hours**

Compare to:
- Sequential: 12.5 hours (8-10x slower ❌)
- Optimized HF: 2.8 hours (2x slower)

---

## File Locations

### On Cluster

```
/scratch/$USER/maniver/ManiVer/
├── configs/
│   └── b2-configs.txt              # B2 credentials (you must copy)
├── data/
│   └── trajectories/
│       ├── olmo3_sft/
│       │   └── logiqa_trajectories_vllm_optimized.h5
│       ├── olmo3_rl_zero/
│       │   └── logiqa_trajectories_vllm_optimized.h5
│       └── olmo3_think/
│           └── logiqa_trajectories_vllm_optimized.h5
└── scripts/
    ├── collection/
    │   └── collect_logiqa_vllm_fully_optimized.py
    └── deployment/
        ├── setup_slurm_env.sh
        └── run_logiqa_slurm.sbatch
```

### On B2

```
b2://ml-activations-store/
└── slurm_YYYYMMDD_HHMMSS/
    └── trajectories/
        ├── olmo3_sft/
        │   └── logiqa_trajectories_vllm_optimized.h5
        ├── olmo3_rl_zero/
        │   └── logiqa_trajectories_vllm_optimized.h5
        └── olmo3_think/
            └── logiqa_trajectories_vllm_optimized.h5
```

---

## Download from B2

After job completes, download to local machine:

```bash
# List available runs
b2 ls ml-activations-store

# Download specific run
b2 sync b2://ml-activations-store/slurm_20260118_173045/trajectories/ \
    data/trajectories/
```

Or use the Python script:

```bash
python scripts/storage/b2_download.py \
    --remote-prefix slurm_20260118_173045/trajectories \
    --local-dir data/trajectories
```

---

## Cost Estimate

**SLURM cluster**: Typically free for academic users (check with your institution)

If you were to run this on vast.ai with H100:
- H100: $2.50/hr
- Time: 1.5-2 hrs
- **Cost**: $3.75-$5.00

Compare to sequential on RTX 4090:
- RTX 4090: $0.60/hr
- Time: 45-55 hrs (for 3 models)
- **Cost**: $27-33

**SLURM advantage**: 8-10x faster, potentially free!

---

## Safety Checklist

Before submitting job, verify:

- [ ] Ran `setup_slurm_env.sh` (one-time setup)
- [ ] Copied B2 credentials to cluster
- [ ] Edited `<netid>` in sbatch file
- [ ] On login node (NOT compute node)
- [ ] Using `sbatch` (NOT running Python directly)

**Remember**: NEVER run compute jobs on login node - always use `sbatch`!
