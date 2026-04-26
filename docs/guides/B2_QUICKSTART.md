# B2 Storage Quick Reference

## Setup Complete ✅

All scripts are ready to use! B2 CLI is authorized on your local machine.

## Quick Commands

### Upload Trajectories (Local Machine)

```bash
# Upload all trajectories
python scripts/b2_upload.py

# Upload with custom path
python scripts/b2_upload.py --local-dir data/trajectories --remote-prefix phase2/run1

# Dry run first
python scripts/b2_upload.py --dry-run
```

### Download Trajectories (vast.ai or Local)

```bash
# First-time setup on new machine
bash scripts/setup_b2_on_vastai.sh

# Download all trajectories
python scripts/b2_download.py

# List what's available
python scripts/b2_download.py --list-only

# Download specific prefix
python scripts/b2_download.py --remote-prefix phase2/run1
```

### Manual Operations

```bash
# List all files
b2 ls b2://ml-activations-store/

# Upload single file
b2 upload-file ml-activations-store local.pkl remote/path.pkl

# Download single file
b2 download-file-by-name ml-activations-store remote/path.pkl local.pkl

# Sync directory (like rsync)
b2 sync data/trajectories/ b2://ml-activations-store/trajectories/
```

## Test Results

✅ Upload test: SUCCESS
✅ Download test: SUCCESS
✅ Public access: SUCCESS
⚠️ Cloudflare CDN: DNS not configured yet (see [B2_SETUP.md](../configs/B2_SETUP.md))

## File Access

**Direct B2 URL (costs egress):**
```
https://f005.backblazeb2.com/file/ml-activations-store/<path>
```

**Cloudflare CDN (free egress, after DNS setup):**
```
https://activations.maniact.space/<path>
```

## Cost Estimate

- **Storage:** ~$5/TB/month
- **Egress (direct B2):** $10/TB
- **Egress (via Cloudflare):** FREE

Expected: $5-10/month (1-2 TB storage + Cloudflare CDN)

## Typical Workflow

### On vast.ai GPU Instance

```bash
# 1. Clone repo and setup
git clone <repo> /workspace/maniver
cd /workspace/maniver
bash scripts/setup_b2_on_vastai.sh

# 2. Download existing data (if needed)
python scripts/b2_download.py --remote-prefix phase2/existing

# 3. Run collection
python scripts/collect_trajectories_with_labels.py --n-samples 500

# 4. Upload results
python scripts/b2_upload.py --remote-prefix phase2/$(date +%Y%m%d_%H%M)

# 5. Destroy instance (data is safe in B2!)
```

### On Local Machine (Analysis)

```bash
# 1. Download latest trajectories
python scripts/b2_download.py --remote-prefix phase2/20260114_1552

# 2. Analyze
jupyter notebook notebooks/analyze.ipynb

# 3. Upload analysis results
python scripts/b2_upload.py --local-dir results/ --remote-prefix analysis/20260114
```

## Directory Structure in B2

```
ml-activations-store/
├── trajectories/           # Main trajectory storage
│   ├── gsm8k/
│   │   ├── olmo3_base/
│   │   ├── olmo3_rl_zero/
│   │   └── ...
│   ├── humaneval/
│   └── logiqa/
├── phase2/                 # Phase 2 experiments
│   ├── 20260114/          # Dated collections
│   └── pilot/
├── analysis/              # Analysis results
└── test/                  # Test uploads
```

## Troubleshooting

**"Not authorized"**
```bash
b2 authorize-account <key_id> <app_key>
```

**"Bucket not found"**
```bash
# Check bucket name in configs/b2-configs.txt
b2 list-buckets
```

**Slow upload/download**
```bash
# Increase threads
b2 sync --threads 8 source dest
```

## Next Steps

1. [ ] Configure Cloudflare DNS for free egress (see [B2_SETUP.md](../configs/B2_SETUP.md))
2. [ ] Test full trajectory upload (~56 GB)
3. [ ] Add compression before upload
4. [ ] Implement streaming upload during collection
