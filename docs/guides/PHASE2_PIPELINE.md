# Phase 2 Pipeline: Collect → Upload → Analyze

Complete automated pipeline for Phase 2 trajectory collection with Backblaze B2 storage.

## Architecture

```
Local Machine          vast.ai Instance              Backblaze B2
(Setup/Control)        (GPU Collection)              (Persistent Storage)
┌──────────────┐      ┌────────────────────┐       ┌────────────────────┐
│ Launch       │ ───► │ Auto-setup         │       │                    │
│ instance     │      │ Clone repo         │       │ Timestamped runs   │
│              │      │ Install deps       │       │ phase2_YYYYMMDD/   │
│              │      │ Setup B2 CLI       │       │   trajectories/    │
│              │      ├────────────────────┤       │   checkpoints/     │
│              │      │ Run collection     │       │   logs/            │
│              │      │ 500 samples × 3    │  ──►  │                    │
│              │      │ tasks × 4 models   │upload │ ~56 GB per run     │
│              │      │ ~60-85 GPU hrs     │       │                    │
│              │      ├────────────────────┤       │                    │
│              │      │ Auto-upload to B2  │       │                    │
│              │      │ Cleanup local      │       │                    │
│              │      │ Destroy instance   │       │                    │
└──────────────┘      └────────────────────┘       └────────────────────┘
                                                     ▲
                                                     │ Download for analysis
                                                     │
                                           ┌────────────────────┐
                                           │ Analysis Machine   │
                                           │ (Local/vast.ai)    │
                                           │                    │
                                           │ Compute signatures │
                                           │ Test H1/H2         │
                                           └────────────────────┘
```

## Quick Start

### 1. One-Time Setup (Local Machine)

```bash
# Install B2 CLI (already done if you followed B2_SETUP.md)
pip install b2

# Authorize B2 (credentials from configs/b2-configs.txt)
b2 authorize-account <key_id> <app_key>

# Test pipeline with 2 samples (optional but recommended)
bash scripts/test_phase2_pipeline.sh
```

### 2. Launch vast.ai Instance

```bash
# Search for best instance
python scripts/vast_launcher.py search --sort-by cost

# Launch instance (auto-setup, installs deps, clones repo)
python scripts/vast_launcher.py launch --sort-by cost

# Wait for setup to complete (~2-3 minutes)
python scripts/vast_launcher.py status
```

### 3. Run Collection Pipeline on vast.ai

```bash
# SSH into instance
vastai ssh <instance_id>

# Run full Phase 2 pipeline (collect + upload)
cd /workspace/maniver
bash scripts/run_phase2_pipeline.sh

# This will:
# - Collect 500 samples × 3 tasks × 4 models (~60-85 GPU hours)
# - Upload to B2 automatically
# - Ask if you want to cleanup local storage
```

### 4. Download & Analyze (Later)

```bash
# On your analysis machine (local or new vast.ai)

# List available runs
python scripts/b2_download.py --list-only

# Download specific run
python scripts/b2_download.py --remote-prefix phase2_20260114_1552/trajectories

# Analyze
python scripts/verify_trajectories.py
python scripts/compute_signatures.py
python scripts/test_h1.py
```

### 5. Cleanup

```bash
# On vast.ai instance (after upload confirms)
# Delete local trajectories if prompted by pipeline
# OR manually: rm -rf data/trajectories/*

# Destroy instance when done
python scripts/vast_launcher.py destroy <instance_id>
```

## Scripts Reference

### Collection Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| [collect_trajectories_with_labels.py](scripts/collect_trajectories_with_labels.py) | Core collection script | Direct invocation (manual) |
| [run_phase2_pipeline.sh](scripts/run_phase2_pipeline.sh) | **Integrated pipeline** (collect + upload) | **Recommended for vast.ai** |
| [test_phase2_pipeline.sh](scripts/test_phase2_pipeline.sh) | Test with 2 samples | Test locally before full run |

### Storage Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| [b2_upload.py](scripts/b2_upload.py) | Upload to B2 | Manual upload or retry |
| [b2_download.py](scripts/b2_download.py) | Download from B2 | Get data for analysis |
| [setup_b2_on_vastai.sh](scripts/setup_b2_on_vastai.sh) | Auto B2 setup | Called by onstart script |

### Instance Management

| Script | Purpose | Usage |
|--------|---------|-------|
| [vast_launcher.py](scripts/vast_launcher.py) | Search/launch/destroy instances | Primary interface |

## Configuration

### B2 Credentials

File: [configs/b2-configs.txt](configs/b2-configs.txt)

```
B2_KEY_ID=0051b5729f87ba10000000001
B2_BUCKET_NAME=ml-activations-store
B2_APP_KEY=K005ETXoFOt7CYSNMJLQz8Y4covA02g
CLOUDFLARE_DOMAIN=activations.maniact.space
```

**Important**: Update f002 → f005 if needed (check with `b2 get-account-info`)

### Model Configuration

File: [configs/models.yaml](configs/models.yaml)

Models for Phase 2:
- `olmo3_base`: Base model (no post-training)
- `olmo3_sft`: SFT only
- `olmo3_rl_zero`: Pure RL (no SFT)
- `olmo3_think`: Full pipeline (SFT+DPO+RLVR)

### Collection Settings

In [collect_trajectories_with_labels.py](scripts/collect_trajectories_with_labels.py):

```python
N_SAMPLES = 500           # Samples per task
MAX_SEQ_LEN = 512         # Trajectory length
MAX_NEW_TOKENS = 512      # Generation length
LAYERS_TO_COLLECT = [0,2,4,...,30]  # Even layers only
```

## Pipeline Options

### Full Collection (Default)

```bash
bash scripts/run_phase2_pipeline.sh
```

**Collects**: All 4 models × 3 tasks × 500 samples = 6,000 samples
**Time**: ~60-85 GPU hours
**Storage**: ~56 GB

### Partial Collection

```bash
# Specific models only
bash scripts/run_phase2_pipeline.sh --models olmo3_base,olmo3_rl_zero

# Specific tasks only
bash scripts/run_phase2_pipeline.sh --tasks gsm8k,logiqa

# Custom run name
bash scripts/run_phase2_pipeline.sh --run-name pilot_run_1

# Keep local copies after upload
bash scripts/run_phase2_pipeline.sh --keep-local
```

### Manual Collection (No Auto-Upload)

```bash
# Just collect, no upload
python scripts/collect_trajectories_with_labels.py

# Upload later
python scripts/b2_upload.py --remote-prefix phase2_$(date +%Y%m%d)
```

## Expected Results

### Collection Progress

```
==============================================================================
Model: olmo3_base (allenai/Olmo-3-1025-7B)
==============================================================================

Task: gsm8k
  Collecting 500 samples (from 0)...
  Current: 0 correct, 0 incorrect
  olmo3_base/gsm8k: 100%|████████| 500/500 [3:45:23<00:00, 27.05s/it, correct=324, incorrect=176]
  Completed: 324 correct, 176 incorrect
  File size: 4.7 GB

Task: logiqa
  ...
```

### Upload Confirmation

```
[3/4] Uploading trajectories to Backblaze B2...

Total size to upload: 56G
Syncing to b2://ml-activations-store/phase2_20260114_1552/trajectories/

✓ Upload completed successfully

Files available at:
  Direct B2: https://f005.backblazeb2.com/file/ml-activations-store/phase2_20260114_1552/trajectories/
  Cloudflare CDN: https://activations.maniact.space/phase2_20260114_1552/trajectories/
```

### Final Summary

```
Summary:
  Run name: phase2_20260114_1552
  Log file: logs/phase2_collection_20260114_1552.log
  B2 location: b2://ml-activations-store/phase2_20260114_1552/

To download on another machine:
  python scripts/b2_download.py --remote-prefix phase2_20260114_1552/trajectories
```

## Data Format

### HDF5 Structure

Each trajectory file (`{model}/{task}_trajectories.h5`):

```python
{
    'trajectories': (500, 512, 16, 4096),  # float16, gzip compressed
    'sequence_lengths': (500,),             # int32
    'is_correct': (500,),                   # bool
    'prompts': (500,),                      # string
    'model_outputs': (500,),                # string
    'ground_truth': (500,),                 # string
}
```

**Metadata attributes**:
- `model`: Model key (e.g., 'olmo3_base')
- `task`: Task name (e.g., 'gsm8k')
- `n_samples`: 500
- `layers`: [0, 2, 4, ..., 30]
- `collection_date`: ISO timestamp

### Directory Structure in B2

```
ml-activations-store/
├── phase2_20260114_1552/
│   ├── trajectories/
│   │   ├── olmo3_base/
│   │   │   ├── gsm8k_trajectories.h5      (4.7 GB)
│   │   │   ├── logiqa_trajectories.h5     (4.7 GB)
│   │   │   └── humaneval_trajectories.h5  (4.7 GB)
│   │   ├── olmo3_sft/
│   │   ├── olmo3_rl_zero/
│   │   └── olmo3_think/
│   ├── checkpoints/
│   │   ├── labeled_olmo3_base_gsm8k.json
│   │   └── ...
│   └── logs/
│       └── phase2_collection_20260114_1552.log
```

## Troubleshooting

### Collection Fails Mid-Run

```bash
# Check checkpoint
cat data/checkpoints/labeled_olmo3_base_gsm8k.json
# Shows: {"completed_samples": 275, "n_correct": 180, "n_incorrect": 95}

# Resume from checkpoint (automatic)
bash scripts/run_phase2_pipeline.sh
```

### Upload Fails

```bash
# Re-authorize B2
b2 authorize-account <key_id> <app_key>

# Retry upload manually
python scripts/b2_upload.py --remote-prefix phase2_$(date +%Y%m%d_%H%M)
```

### Out of Disk Space

```bash
# Check usage
df -h /workspace

# Delete temporary files
rm -rf offload/  # Model offload cache
rm -rf logs/*.log  # Old logs

# Upload and delete existing trajectories
python scripts/b2_upload.py
rm -rf data/trajectories/*
```

### GPU Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Collection script auto-handles multi-GPU with device_map="auto"
# If still OOM, reduce MAX_SEQ_LEN in collect_trajectories_with_labels.py
```

### Download Slow from B2

```bash
# Use Cloudflare CDN instead (once DNS configured)
# Or increase threads
python scripts/b2_download.py --threads 8

# Or download specific files only
python scripts/b2_download.py --file trajectories/olmo3_base/gsm8k_trajectories.h5
```

## Cost Estimation

### vast.ai Instance

| Component | Cost | Notes |
|-----------|------|-------|
| GPU hours | $0.20-0.40/hr | RTX 3090/4090 |
| Total for 85 hrs | **$17-34** | Full Phase 2 |

### Backblaze B2 Storage

| Component | Cost | Notes |
|-----------|------|-------|
| Storage | $5/TB/month | ~56 GB = $0.28/month |
| Egress (direct) | $10/TB | If downloading directly |
| Egress (Cloudflare) | **FREE** | Once DNS configured |

**Total estimated cost**: $17-34 for collection + $0.28/month storage

## Next Steps After Collection

1. **Verify Data Quality**
   ```bash
   python scripts/verify_trajectories.py
   # Checks for NaN, balanced classes, file integrity
   ```

2. **Compute Path Signatures** (Week 3 of Phase 2)
   ```bash
   python scripts/compute_signatures.py
   # Uses signatory library, PCA projection
   ```

3. **Test H1 Hypothesis** (Week 4 of Phase 2)
   ```bash
   python scripts/test_h1.py
   # Within-domain classification: correct vs incorrect
   # Success criterion: >65% accuracy across model/task combinations
   ```

4. **Proceed to Phase 3** (if H1 succeeds)
   - Test H2: Cross-domain transfer
   - Train on GSM8K, test on HumanEval/LogiQA

## Documentation

- [B2_SETUP.md](configs/B2_SETUP.md): Detailed B2 + Cloudflare setup
- [B2_QUICKSTART.md](scripts/B2_QUICKSTART.md): Quick command reference
- [PHASE2_DETAILED_PLAN.md](PHASE2_DETAILED_PLAN.md): Complete Phase 2 specification
- [CLAUDE.md](CLAUDE.md): Project overview and workflow

## Support

If you encounter issues:

1. Check logs: `logs/phase2_collection_*.log`
2. Verify checkpoints: `data/checkpoints/`
3. Test with small run: `bash scripts/test_phase2_pipeline.sh`
4. Review documentation above

---

**Ready to start?**

```bash
# Test locally first (recommended)
bash scripts/test_phase2_pipeline.sh

# Then launch full collection
python scripts/vast_launcher.py launch
# On vast.ai: bash scripts/run_phase2_pipeline.sh
```
