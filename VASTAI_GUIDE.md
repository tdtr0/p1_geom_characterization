# vast.ai Guide: Setup, Login, and Cost Optimization

## Quick Start

### 1. Install vast.ai CLI

```bash
pip install vastai
```

### 2. Login to vast.ai

**Get your API key:**
1. Go to https://cloud.vast.ai/
2. Sign up or log in
3. Click your username (top right) → Account
4. Copy your API key

**Set API key:**
```bash
# Option 1: Set in environment (temporary)
export VASTAI_API_KEY="your_api_key_here"

# Option 2: Set permanently
vastai set api-key your_api_key_here

# Verify login
vastai show user
```

### 3. Add Billing

Before you can rent instances, add payment method:
1. Go to https://cloud.vast.ai/billing/
2. Add credit card or crypto
3. Add initial credit ($20-50 recommended)

**Pricing**: Pay-as-you-go, charged per second of usage

## GPU Cost Analysis for Phase 2

### ⚡ Quick Answer (Pareto Optimal)

**Best options in order:**

1. **4× RTX 5090** → 5.5 hrs, $18 ⭐ (Best speed/cost if available)
2. **4× RTX 3090** → 11 hrs, $15 ⭐ (Best value, most common)
3. **Single RTX 5090** → 22 hrs, $18 (Best single GPU)
4. **4× A100 40GB** → 7.25 hrs, $38 (Cloud standard)
5. **4× H100** → 4.5 hrs, $54 (Fastest)

**To find what's available right now:**
```bash
bash scripts/deployment/find_optimal_gpu.sh
```

See [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) for detailed analysis.

### Collection Performance Estimate

Phase 2 collects **6,000 samples** (500 × 3 tasks × 4 models):
- Generation: ~3 million tokens total
- Model loading: ~2 min per model
- Per-sample time depends on GPU speed

**Key metric**: Tokens per second (TPS)
**Key insight**: 4× parallel GPUs ≈ 4× faster than single GPU

### Memory Requirements

**Per model**: ~20 GB VRAM (7B model + KV cache + activations)
**For 4 models parallel**: ~80 GB total

✅ Any 4× GPU with 24+ GB per GPU is sufficient
✅ Extra VRAM doesn't help (B200's 192 GB is massive overkill)

### Single GPU Options (Sequential Processing)

| GPU | VRAM | TPS | Time (hrs) | $/hr | Total Cost | Notes |
|-----|------|-----|------------|------|------------|-------|
| **RTX 5090** | 24 GB | 150 | **22** | $0.80 | **$18** | 🏆 Best single GPU |
| H100 PCIe | 80 GB | 170 | 20 | $2.50 | $50 | Fast but expensive |
| A100 40GB | 40 GB | 115 | 30 | $1.30 | $39 | Balanced |
| RTX 4090 | 24 GB | 100 | 34 | $0.60 | $20 | Budget |
| RTX 3090 | 24 GB | 70 | 49 | $0.35 | $17 | Cheapest but slow |

### Multi-GPU Options (4× Parallel - RECOMMENDED)

**Why 4× GPUs?** Run 1 model per GPU → 4× faster!

| Setup | TPS/GPU | Time (hrs) | $/hr | Total Cost | Notes |
|-------|---------|------------|------|------------|-------|
| **4× RTX 5090** | 150 | **5.5** | $3.20 | **$18** | 🏆 Best speed/cost |
| **4× RTX 3090** | 70 | **11** | $1.40 | **$15** | 🏆 Cheapest |
| 4× RTX 4090 | 100 | 8 | $2.40 | $19 | Good middle |
| 4× A100 40GB | 115 | 7.25 | $5.20 | $38 | Cloud standard |
| 4× H100 | 190 | 4.5 | $12.00 | $54 | Fastest |
| 4× B200 | 280 | 3.25 | $22.00 | $70 | Overkill (if available) |

**Pareto Frontier**: 4× RTX 5090 dominates everything except 4× RTX 3090 (which is slower but $3 cheaper)

### Cost Calculation Details

**Formula**: Total Cost = (GPU hours) × ($/hour)

**Example for 4× RTX 3090**:
```
6,000 samples ÷ 4 models = 1,500 samples per GPU
1,500 × 512 tokens × (1/70 tokens/sec) = 11 hours per GPU
All 4 run in parallel → 11 hours total
11 hrs × $1.40/hr = $15.40 total
```

**Why multi-GPU is better**:
- Single RTX 3090: 49 hrs × $0.35/hr = $17 (slower, MORE expensive)
- 4× RTX 3090: 11 hrs × $1.40/hr = $15 (4× faster, LESS expensive)

## Finding and Launching Instances

### Method 1: Using Our Script (Recommended)

```bash
# Search for best options
python scripts/deployment/vast_launcher.py search --sort-by cost

# Launch with confirmation
python scripts/deployment/vast_launcher.py launch --sort-by cost
```

### Method 2: Manual Search

```bash
# Search for specific GPU
vastai search offers "gpu_name = RTX_3090 reliability > 0.95 inet_down > 200 disk_space > 300"

# Sort by performance per dollar
vastai search offers --order "dlperf_usd"

# Show top 10
vastai search offers "gpu_ram >= 24 reliability > 0.95" --order "dlperf_usd" | head -11
```

### Method 3: Web Interface

1. Go to https://cloud.vast.ai/templates/
2. Use "Create" tab
3. Set filters:
   - GPU: Select preferred (e.g., "RTX 4090")
   - VRAM: ≥ 24 GB
   - Bandwidth: ≥ 200 Mbps up/down
   - Reliability: ≥ 95%
4. Sort by "DLPerf/$" (performance per dollar)
5. Click "Rent" on best option

## Instance Management

### Check Running Instances

```bash
vastai show instances

# Output shows:
#   ID    STATUS   GPU      $/hr  SSH_PORT  SSH_HOST
#   12345 running  4×3090   1.20  41234     ssh5.vast.ai
```

### SSH into Instance

```bash
# Method 1: Using vast.ai CLI
vastai ssh 12345

# Method 2: Direct SSH
ssh -p 41234 root@ssh5.vast.ai -L 8080:localhost:8080

# Method 3: Use our launcher
python scripts/deployment/vast_launcher.py status
```

### Monitor Collection Progress

```bash
# SSH into instance first
vastai ssh 12345

# Monitor collection
tail -f data/logs/phase2_collection_*.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check disk usage
df -h /workspace
```

### Destroy Instance When Done

```bash
# Using our script
python scripts/deployment/vast_launcher.py destroy 12345

# Or directly
vastai destroy instance 12345
```

**IMPORTANT**: Always destroy instances when done to avoid charges!

## Recommended Workflow

### For Speed (Complete in ~1 day)

```bash
# 1. Search for H100 or 2× A100
python scripts/deployment/vast_launcher.py search --sort-by perf

# 2. Launch (will take 25-40 hours)
python scripts/deployment/vast_launcher.py launch

# 3. SSH in and start pipeline
vastai ssh <id>
cd /workspace/maniver
bash scripts/collection/run_phase2_pipeline.sh

# Expected: Done in 25-40 hours, cost: $50-75
```

### For Cost (Cheapest total)

```bash
# 1. Search for 4× RTX 3090 or single RTX 3090
python scripts/deployment/vast_launcher.py search --sort-by cost

# 2. Launch
python scripts/deployment/vast_launcher.py launch

# 3. Run collection
vastai ssh <id>
cd /workspace/maniver

# If multi-GPU (4× RTX 3090), run models in parallel:
CUDA_VISIBLE_DEVICES=0 python scripts/collection/collect_single_model.py olmo3_base &
CUDA_VISIBLE_DEVICES=1 python scripts/collection/collect_single_model.py olmo3_sft &
CUDA_VISIBLE_DEVICES=2 python scripts/collection/collect_single_model.py olmo3_rl_zero &
CUDA_VISIBLE_DEVICES=3 python scripts/collection/collect_single_model.py olmo3_think &
wait

# Then upload to B2
python scripts/storage/b2_upload.py

# Expected: 18-75 hours depending on GPU, cost: $21-33
```

### For Balance (Recommended)

```bash
# Search for A100 40GB
vastai search offers "gpu_name = A100_PCIE_40GB reliability > 0.95 inet_down > 200" --order dlperf_usd | head -11

# Launch best option
# Run standard pipeline
bash scripts/collection/run_phase2_pipeline.sh

# Expected: ~40 hours, cost: ~$52
```

## Billing and Costs

### How Charging Works

- **Per-second billing**: Charged every second instance is running
- **No minimum commitment**: Can destroy anytime
- **Auto-stop on funds**: Instance stops if credits run out (won't overcharge)

### Monitor Spending

```bash
# Check current balance
vastai show user

# See instance costs
vastai show instances
```

**Web dashboard**: https://cloud.vast.ai/billing/

### Typical Costs for Full Phase 2

| Scenario | Time | Cost | When to Use |
|----------|------|------|-------------|
| **Fastest** (H100) | 1 day | $65 | Need results ASAP |
| **Balanced** (A100 40GB) | 1.5 days | $52 | **Recommended** |
| **Budget** (RTX 3090) | 3 days | $24 | Have time, minimize cost |
| **Multi-GPU** (4× RTX 3090) | 1 day | $22 | **Best value** if available |

### Hidden Costs to Watch

✓ **Storage**: Included in hourly rate (typically $0.15/GB/month prorated)
✓ **Network**: Usually free
✓ **Idle time**: Charged even if not using GPU! Always destroy when done

### Cost-Saving Tips

1. **Test first**: Run `test_phase2_pipeline.sh` locally before launching
2. **Use checkpointing**: Collection auto-saves every 25 samples
3. **Monitor actively**: Check progress every few hours
4. **Upload immediately**: Use `run_phase2_pipeline.sh` to auto-upload and cleanup
5. **Destroy promptly**: Don't leave instances idle

## Troubleshooting

### Can't login

```bash
# Check API key is set
echo $VASTAI_API_KEY

# Or check saved key
vastai show user

# If fails, re-set key
vastai set api-key your_api_key_here
```

### No instances available

- Try different GPU types
- Increase max price filter
- Try different regions
- Check at different times (availability varies)

### Instance stuck in "loading"

- Wait 2-3 minutes (normal setup time)
- If >5 minutes, destroy and try another instance

### Collection fails mid-run

```bash
# SSH in and check logs
vastai ssh <id>
cd /workspace/maniver
tail -100 data/logs/phase2_collection_*.log

# Check checkpoints to see progress
cat data/checkpoints/labeled_olmo3_base_gsm8k.json

# Resume by re-running pipeline (auto-resumes from checkpoint)
bash scripts/collection/run_phase2_pipeline.sh
```

### Out of credits

1. Go to https://cloud.vast.ai/billing/
2. Add more credits
3. Instance will auto-resume if still allocated

## Summary

**Recommended setup for Phase 2:**

1. **Budget conscious** → Single RTX 3090 ($24, 3 days)
2. **Time conscious** → H100 ($65, 1 day)
3. **Best value** → 4× RTX 3090 ($22, 1 day) or A100 40GB ($52, 1.5 days)

**Login steps:**
```bash
pip install vastai
vastai set api-key <your_key>
vastai show user  # Verify login
```

**Launch:**
```bash
python scripts/deployment/vast_launcher.py launch --sort-by cost
```

**Monitor:**
```bash
vastai show instances
vastai ssh <id>
```

**Always destroy when done!**
```bash
python scripts/deployment/vast_launcher.py destroy <id>
```
