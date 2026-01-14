# Backblaze B2 + Cloudflare CDN Setup

## Current Status

✅ **Completed:**
- B2 CLI installed and authorized locally
- Bucket `ml-activations-store` is configured and public
- Test file uploaded successfully
- Upload/download scripts created
- Direct B2 access verified

⚠️ **Pending:**
- Cloudflare DNS configuration for `activations.maniact.space`

## Architecture

```
vast.ai Instance              Backblaze B2              Cloudflare CDN
(ephemeral)                   (persistent)              (free egress)
┌─────────────────┐          ┌─────────────────┐      ┌──────────────────┐
│ 200-300 GB SSD  │  upload  │ ml-activations- │      │ activations.     │
│                 │ ───────► │ store bucket    │ ───► │ maniact.space    │
│ data/           │          │ ~$5/TB/month    │      │ (free downloads) │
│ trajectories/   │          └─────────────────┘      └──────────────────┘
└─────────────────┘
```

## Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| B2 Storage | $5/TB/month | ~1-2 TB needed |
| B2 Egress (direct) | $10/TB | If downloading directly |
| B2 Egress (via Cloudflare) | **$0** | Free with Cloudflare CDN |
| Cloudflare CDN | **$0** | Free tier (no bandwidth limits) |

**Monthly estimate:** $5-10 (storage only, assuming Cloudflare CDN for downloads)

## Configuration Details

### B2 Credentials
- **Account ID:** 1b5729f87ba1
- **Key ID:** 0051b5729f87ba10000000001
- **Bucket:** ml-activations-store
- **Region:** us-east-005

### URLs

**Direct B2 URLs (costs egress):**
```
https://f005.backblazeb2.com/file/ml-activations-store/<path>
```

**Cloudflare CDN URLs (free egress, once DNS configured):**
```
https://activations.maniact.space/<path>
```

## Cloudflare DNS Configuration (TODO)

The domain `activations.maniact.space` currently points to Porkbun parking (44.227.65.245). To enable free egress via Cloudflare:

### Option 1: CNAME to B2 (Recommended)

1. Go to Cloudflare dashboard → DNS
2. Add CNAME record:
   - **Name:** `activations.maniact.space`
   - **Target:** `f005.backblazeb2.com`
   - **Proxy status:** Proxied (orange cloud) ☁️
   - **TTL:** Auto

3. Add Page Rule for path rewriting:
   - **URL:** `activations.maniact.space/*`
   - **Forwarding URL:** `https://f005.backblazeb2.com/file/ml-activations-store/$1`

### Option 2: Cloudflare Workers (More Control)

Create a Cloudflare Worker to proxy B2:

```javascript
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)
  const b2Url = `https://f005.backblazeb2.com/file/ml-activations-store${url.pathname}`

  return fetch(b2Url, {
    method: request.method,
    headers: request.headers
  })
}
```

### Verification

Once DNS is configured, verify Cloudflare is serving content:

```bash
# Should show CF-Ray header (indicates Cloudflare)
curl -I https://activations.maniact.space/test/b2_test.txt | grep CF-
```

Expected output:
```
CF-Ray: 8f8a8b8c8d8e8f-SJC
CF-Cache-Status: HIT
Server: cloudflare
```

## Usage

### On Local Machine (macOS)

```bash
# Upload trajectories to B2
python scripts/b2_upload.py

# Upload specific directory
python scripts/b2_upload.py --local-dir data/trajectories --remote-prefix phase2/run1

# Dry run (see what would be uploaded)
python scripts/b2_upload.py --dry-run
```

### On vast.ai GPU Instance

```bash
# First-time setup
bash scripts/setup_b2_on_vastai.sh

# Download all trajectories
python scripts/b2_download.py

# Download specific experiment
python scripts/b2_download.py --remote-prefix phase2/run1

# List available files
python scripts/b2_download.py --list-only
```

### Manual B2 CLI Commands

```bash
# List all files in bucket
b2 ls b2://ml-activations-store/

# List specific prefix
b2 ls b2://ml-activations-store/trajectories/

# Upload single file
b2 upload-file ml-activations-store local_file.pkl trajectories/file.pkl

# Download single file
b2 download-file-by-name ml-activations-store trajectories/file.pkl local_file.pkl

# Sync directory (like rsync)
b2 sync data/trajectories/ b2://ml-activations-store/trajectories/
```

## Workflow Examples

### Scenario 1: Collect on vast.ai, store permanently

```bash
# On vast.ai instance
cd /workspace/maniver

# Run collection
python scripts/collect_trajectories_with_labels.py \
  --models olmo3_base olmo3_rl_zero \
  --n-samples 500

# Upload to B2 for permanent storage
python scripts/b2_upload.py \
  --local-dir data/trajectories \
  --remote-prefix phase2/$(date +%Y%m%d)

# Destroy instance (data is safe in B2)
```

### Scenario 2: Download existing data to new instance

```bash
# On new vast.ai instance
bash scripts/setup_b2_on_vastai.sh

# Download previous trajectories
python scripts/b2_download.py \
  --remote-prefix phase2/20260114

# Continue analysis
python scripts/analyze_geometry.py
```

### Scenario 3: Local development with cloud data

```bash
# On local machine
python scripts/b2_download.py \
  --remote-prefix phase2/20260114 \
  --local-dir data/trajectories_from_cloud

# Work locally with downloaded data
jupyter notebook notebooks/explore_trajectories.ipynb
```

## Storage Estimates

| Dataset | Samples | Models | Tasks | Size |
|---------|---------|--------|-------|------|
| Small test | 10 | 1 | 1 | ~100 MB |
| Phase 2 pilot | 100 | 2 | 3 | ~5 GB |
| Full collection | 500 | 5 | 3 | ~56 GB |
| Phase 2 complete | 500 | 5 | 3 | ~56 GB |

**Total B2 storage needed:** 100-200 GB initially, growing to ~1 TB

## Troubleshooting

### Permission denied when uploading
```bash
# Re-authorize B2
b2 authorize-account <key_id> <app_key>
```

### Slow downloads on vast.ai
```bash
# Use Cloudflare CDN instead (once DNS configured)
# Or increase thread count
b2 sync --threads 8 b2://bucket/path local/path
```

### Out of disk space on vast.ai
```bash
# Upload and then delete local copy
python scripts/b2_upload.py
rm -rf data/trajectories/*

# Or stream directly during collection (future enhancement)
```

## Next Steps

1. **Complete Cloudflare DNS setup** - Configure CNAME or Worker for free egress
2. **Test Cloudflare proxy** - Verify CF-Ray headers appear
3. **Update scripts** - Add automatic fallback from Cloudflare to direct B2
4. **Add compression** - Use gzip for .pkl files before upload
5. **Implement streaming** - Upload during collection to save disk space
