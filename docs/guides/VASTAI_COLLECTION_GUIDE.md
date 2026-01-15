# Practical Guide: GPU Collection on vast.ai with B2 Upload

This guide documents lessons learned from Phase 2 trajectory collection, including pitfalls and solutions for renting GPUs on vast.ai, running large-scale collection, and uploading to Backblaze B2.

## Overview

**What we did**: Collected 52GB of activation trajectories for 4 models × 3 tasks on vast.ai, uploaded to Backblaze B2.

**Total cost**: ~$12 for 8 hours on 4× RTX 5090

**Key learnings**: HDF5 file locking, parallel GPU execution, B2 storage caps, and auto-monitoring.

---

## 1. Renting on vast.ai

### Finding the Right Instance

```bash
# Search for multi-GPU instances
python scripts/deployment/vast_launcher.py search --sort-by cost

# Key filters to consider:
# - GPU count: 4+ for parallel collection
# - VRAM: 24GB+ per GPU (7B models need ~15GB)
# - Disk: 300GB+ for trajectories
# - Network: 200+ Mbps for B2 uploads
```

### Pitfall: Disk Space is Charged Separately

vast.ai charges for disk storage separately (~$0.20/GB/month). Always check:
- Instance disk size vs your data needs
- You CANNOT add disk space after launch
- Request 50-100GB more than you need

### Pitfall: SSH Key Issues

If you get "Permission denied (publickey)":
```bash
# Add your SSH key to vast.ai
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"

# Attach to instance
vastai attach ssh <instance_id>

# Wait 30-60 seconds before connecting
```

### Pitfall: Instance Might Not Start

Some instances fail to start or become unresponsive. If this happens:
1. Wait 2-3 minutes
2. Try `vastai ssh <instance_id>` directly
3. If still failing, destroy and try another instance

---

## 2. Setting Up the Collection Environment

### On vast.ai Instance

```bash
# Clone repo
cd /workspace
git clone https://github.com/YOUR_REPO.git maniver
cd maniver

# Install dependencies
pip3 install torch transformers h5py tqdm datasets accelerate

# Set up paths
export PYTHONPATH=/workspace/maniver/src:$PYTHONPATH
```

### Pitfall: Missing `accelerate` Library

If you get `ValueError: device_map requires accelerate`:
```bash
pip3 install accelerate
```

### Pitfall: Flash Attention Not Available

Flash Attention 2 may not be installed. The code falls back automatically, but for speed:
```bash
pip3 install flash-attn --no-build-isolation
```

---

## 3. HDF5 File Locking Issues

### The Problem

When running multiple processes writing HDF5 files simultaneously:
```
BlockingIOError: [Errno 11] Unable to synchronously create file
(unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')
```

Or corruption errors:
```
OSError: Can't synchronously write data (filter returned failure during read)
```

### The Solution

**Always disable HDF5 file locking**:
```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

**Stagger process starts** to avoid race conditions:
```bash
# In parallel script, add delays between starts
delay=0
for model in model1 model2 model3 model4; do
    sleep $delay
    CUDA_VISIBLE_DEVICES=$gpu HDF5_USE_FILE_LOCKING=FALSE \
        python3 collect.py --model $model &
    delay=10  # 10 second delay between starts
done
```

### Pitfall: Corrupted Files from Crashes

If a process crashes during HDF5 write, the file may be corrupted. Always:
1. Delete corrupted files before restarting: `rm -rf data/trajectories/MODEL_NAME/`
2. Delete checkpoints: `rm -f data/checkpoints/labeled_MODEL_*.json`
3. Then restart collection

---

## 4. Parallel Collection on Multiple GPUs

### GPU Assignment

Use `CUDA_VISIBLE_DEVICES` to isolate each model to one GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python3 collect.py --model model1 &
CUDA_VISIBLE_DEVICES=1 python3 collect.py --model model2 &
CUDA_VISIBLE_DEVICES=2 python3 collect.py --model model3 &
CUDA_VISIBLE_DEVICES=3 python3 collect.py --model model4 &
```

### Pitfall: device_map="auto" with Single GPU

When `CUDA_VISIBLE_DEVICES` limits to 1 GPU, the model sees only that GPU. Use:
```python
if torch.cuda.device_count() == 1:
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda()
else:
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", torch_dtype=torch.float16)
```

### Monitoring Progress

```bash
# Watch GPU utilization
watch nvidia-smi

# Check file sizes growing
watch -n 60 'find data/trajectories -name "*.h5" -exec ls -lh {} \;'

# Tail all logs
tail -f data/logs/*_collection_*.log
```

---

## 5. Uploading to Backblaze B2

### Setup B2 CLI

```bash
pip3 install b2
b2 account authorize <key_id> <app_key>
```

### Config File Format

Create `configs/b2-configs.txt`:
```
B2_KEY_ID=your_key_id
B2_BUCKET_NAME=your_bucket_name
B2_APP_KEY=your_app_key
```

**Pitfall**: Don't include extra lines or comments - they cause parse errors.

### Upload Command

```bash
# Sync entire directory
b2 sync --replace-newer data/trajectories b2://bucket-name/trajectories/

# Upload single file
b2 file upload bucket-name local/path/file.h5 remote/path/file.h5
```

### Pitfall: Storage Cap Exceeded

If you get `StorageCapExceeded`:
1. Go to https://secure.backblaze.com/b2_caps.htm
2. Increase storage cap (requires payment method)
3. Changes may take a few minutes to propagate

### Upload Speed

From vast.ai to B2: typically 50-200 MB/s (depends on instance location)
- US instances to US B2: fastest
- European instances to US B2: slower

---

## 6. Auto-Monitoring and Recovery

### Monitor Script

Create a script that:
1. Checks GPU memory (near-zero when done)
2. Counts valid files (>500MB each)
3. Auto-restarts crashed processes
4. Triggers B2 upload when complete

See: `scripts/collection/auto_complete_and_upload.sh`

### Key Checks

```bash
# Check if processes are running
pgrep -f "collect_trajectories" | wc -l

# Check GPU memory (should be near-zero when done)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}'

# Count valid files (>500MB)
find data/trajectories -name "*.h5" -size +500M | wc -l
```

---

## 7. Termination Checklist

Before destroying the vast.ai instance:

1. **Verify all files created**:
   ```bash
   find data/trajectories -name "*.h5" -exec ls -lh {} \;
   ```

2. **Verify B2 upload complete**:
   ```bash
   b2 ls --long --recursive b2://bucket-name/trajectories/
   ```

3. **Compare local vs B2 sizes**:
   ```bash
   # Local
   du -sh data/trajectories

   # B2 (check file count and total size)
   b2 ls --recursive b2://bucket-name/trajectories/ | wc -l
   ```

4. **Then destroy**:
   ```bash
   vastai destroy <instance_id>
   ```

---

## 8. Cost Optimization

### GPU Selection for 7B Models

| GPU | VRAM | $/hr | Recommendation |
|-----|------|------|----------------|
| RTX 3090 | 24GB | $0.35 | Budget option |
| RTX 4090 | 24GB | $0.60 | Good value |
| 4× RTX 3090 | 96GB | $1.20 | Best for parallel |
| A100 40GB | 40GB | $1.30 | Fast, single GPU |
| 4× RTX 5090 | 128GB | $1.40 | What we used |

### Time Estimates (500 samples × 3 tasks per model)

- Sequential (1 model at a time): ~28 hours
- Parallel (4 GPUs, 4 models): ~7 hours
- Cost difference: Similar total, but parallel finishes faster

### Scaling to Larger Models (e.g., DeepSeek-R1 8B)

For 8B+ models:
- Need 32GB+ VRAM (A100 40GB or 2× RTX 3090 with model splitting)
- Consider gradient checkpointing to reduce memory
- Expect 1.5-2× longer collection times

---

## 9. Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Permission denied (publickey)` | SSH key not attached | `vastai create ssh-key` + `vastai attach ssh` |
| `Unable to lock file` | HDF5 locking conflict | `export HDF5_USE_FILE_LOCKING=FALSE` |
| `filter returned failure during read` | Corrupted HDF5 file | Delete file and checkpoint, restart |
| `StorageCapExceeded` | B2 storage limit | Increase cap in B2 dashboard |
| `device_map requires accelerate` | Missing library | `pip3 install accelerate` |
| `ModuleNotFoundError: task_data` | PYTHONPATH not set | `export PYTHONPATH=/path/to/src:$PYTHONPATH` |

---

## 10. Quick Reference Commands

```bash
# === vast.ai ===
vastai search offers --sort-by cost          # Find instances
vastai create instance <offer_id>            # Launch
vastai ssh <instance_id>                     # Connect
vastai destroy <instance_id>                 # Terminate

# === Collection ===
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/workspace/maniver/src:$PYTHONPATH
python3 scripts/collection/collect_trajectories_with_labels.py \
    --models olmo3_base --tasks gsm8k --n_samples 500

# === B2 Upload ===
b2 account authorize <key_id> <app_key>
b2 sync data/trajectories b2://bucket/trajectories/
b2 ls --long --recursive b2://bucket/trajectories/

# === Monitoring ===
watch nvidia-smi
tail -f data/logs/*.log
find data/trajectories -name "*.h5" -exec ls -lh {} \;
```

---

## Summary

**What worked well**:
- Parallel collection on 4 GPUs (4× speedup)
- B2 storage for persistent data
- Checkpoint recovery from crashes

**What to watch out for**:
- HDF5 file locking (ALWAYS disable it)
- B2 storage caps (add payment method first)
- Stagger parallel starts to avoid race conditions
- Verify upload before terminating instance

**Next time**:
- Use `auto_complete_and_upload.sh` for hands-off collection
- Consider A100 for single-GPU simplicity
- Budget 50% more time than estimated
