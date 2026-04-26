# GPU Optimization Analysis for Phase 2 Collection

## Memory Requirements (Detailed)

### Per-Model Memory Breakdown

**7B Model (OLMo-3, DeepSeek-R1 distilled)**:
- **Model weights** (float16): ~14 GB
- **KV cache** (512 tokens, float16): ~2 GB
- **Activation storage** (during collection): ~3 GB
- **CUDA overhead**: ~1 GB
- **Total per model**: ~**20 GB VRAM**

**For 4 models in parallel**:
- 4 × 20 GB = **80 GB total**

### Key Insight: Memory is NOT the bottleneck
- 4× RTX 3090 (24 GB each) = 96 GB total ✅ **Sufficient**
- 4× RTX 4090 (24 GB each) = 96 GB total ✅ **Sufficient**
- 4× A100 40GB = 160 GB total (2× what we need - overkill)
- 4× H100 80GB = 320 GB total (4× what we need - massive overkill)
- 4× B200 192GB = 768 GB total (10× what we need - absurd overkill)

**Conclusion**: Any multi-GPU setup with 24+ GB per GPU is sufficient. Extra VRAM doesn't help.

## Speed: What Actually Matters

**Bottleneck**: Token generation speed (TPS = tokens per second)

### Token Generation Performance (7B models, float16)

| GPU | Architecture | TPS (tokens/sec) | Relative Speed |
|-----|--------------|------------------|----------------|
| **B200** | Blackwell (2024) | ~250-300 | 4.3× |
| **H100 SXM** | Hopper (2023) | ~180-200 | 3.1× |
| **H100 PCIe** | Hopper (2023) | ~160-180 | 2.7× |
| **A100 80GB** | Ampere (2021) | ~120-140 | 2.0× |
| **A100 40GB** | Ampere (2021) | ~110-120 | 1.8× |
| **RTX 5090** | Ada/Blackwell (2025) | ~140-160 | 2.4× |
| **RTX 4090** | Ada (2022) | ~95-110 | 1.6× |
| **RTX 3090** | Ampere (2020) | ~70 | **1.0× (baseline)** |

*Note: TPS varies by model, these are estimates for 7B models*

## Phase 2 Collection Time Estimates

**Workload**: 6,000 samples × 512 tokens avg = 3,072,000 tokens
- Plus model loading overhead (~2 min per model)
- Plus checkpointing, I/O (~10% overhead)

### Single GPU (Sequential Processing)

| GPU | TPS | Generation Time | Overhead | Total | $/hr | Cost |
|-----|-----|-----------------|----------|-------|------|------|
| B200 | 280 | 11 hrs | +2 hrs | **13 hrs** | $4-6 | **$52-78** |
| H100 SXM | 190 | 16 hrs | +2 hrs | **18 hrs** | $3.00 | **$54** |
| H100 PCIe | 170 | 18 hrs | +2 hrs | **20 hrs** | $2.50 | **$50** |
| A100 80GB | 130 | 23 hrs | +3 hrs | **26 hrs** | $1.80 | **$47** |
| A100 40GB | 115 | 27 hrs | +3 hrs | **30 hrs** | $1.30 | **$39** |
| RTX 5090 | 150 | 20 hrs | +2 hrs | **22 hrs** | $0.80 | **$18** |
| RTX 4090 | 100 | 30 hrs | +4 hrs | **34 hrs** | $0.60 | **$20** |
| RTX 3090 | 70 | 44 hrs | +5 hrs | **49 hrs** | $0.35 | **$17** |

### Multi-GPU (4× Parallel, 1 model per GPU)

| Setup | TPS/GPU | Per-Model Time | Total | $/hr | Cost |
|-------|---------|----------------|-------|------|------|
| 4× B200 | 280 | 3.25 hrs | **3.25 hrs** | $20-24 | **$65-78** |
| 4× H100 | 190 | 4.5 hrs | **4.5 hrs** | $12.00 | **$54** |
| 4× A100 80GB | 130 | 6.5 hrs | **6.5 hrs** | $7.20 | **$47** |
| 4× A100 40GB | 115 | 7.25 hrs | **7.25 hrs** | $5.20 | **$38** |
| 4× RTX 5090 | 150 | 5.5 hrs | **5.5 hrs** | $3.20 | **$18** |
| 4× RTX 4090 | 100 | 8 hrs | **8 hrs** | $2.40 | **$19** |
| 4× RTX 3090 | 70 | 11 hrs | **11 hrs** | $1.40 | **$15** |

*Note: Parallel efficiency ~95% (minimal overhead since models are independent)*

## Pareto Frontier Analysis

### Cost vs Time Tradeoff

```
Cost ($)
100 |
 90 |
 80 |     4×B200(3h, $70)
 70 |
 60 |
 50 |   H100(18h, $54)  4×H100(4.5h, $54)
 40 |  A100(30h, $39)  4×A100(7h, $38)
 30 |
 20 | RTX5090(22h, $18)  4×5090(5.5h, $18) 4×4090(8h, $19)
 10 | RTX3090(49h, $17)  4×3090(11h, $15) ★
  0 +----+----+----+----+----+----+----+----+----+----+----+
     0    5   10   15   20   25   30   35   40   45   50   Time (hrs)
```

### Pareto Optimal Points

| GPU Setup | Time | Cost | $/hour saved | Best For |
|-----------|------|------|--------------|----------|
| **4× RTX 3090** ⭐ | **11 hrs** | **$15** | $1.36 | **Best value** - Cheapest and fast |
| **RTX 5090 (single)** | 22 hrs | $18 | $0.82 | Best single GPU value |
| **4× RTX 5090** | 5.5 hrs | $18 | $3.27 | Best speed/cost ratio |
| **4× A100 40GB** | 7.25 hrs | $38 | $5.24 | Cloud standard |
| **4× H100** | 4.5 hrs | $54 | $12.00 | Max speed, enterprise |
| **4× B200** | 3.25 hrs | $70 | $21.50 | Bleeding edge (if available) |

### Dominated (Not Pareto Optimal)

These are strictly worse than alternatives:
- ❌ Single RTX 3090: 49 hrs, $17 (dominated by 4× RTX 3090: 11 hrs, $15)
- ❌ Single A100 40GB: 30 hrs, $39 (dominated by 4× RTX 3090: 11 hrs, $15)
- ❌ Single H100: 18 hrs, $54 (dominated by 4× RTX 5090: 5.5 hrs, $18)
- ❌ 4× RTX 4090: 8 hrs, $19 (dominated by 4× RTX 5090: 5.5 hrs, $18)

## Recommendations

### For Budget Conscious ($15-20)
**Winner: 4× RTX 3090** ⭐
- **Cost**: $15
- **Time**: 11 hours (< 1 day)
- **Why**: Cheapest option, completes overnight
- **Availability**: Common on vast.ai

### For Time Conscious ($15-25)
**Winner: 4× RTX 5090**
- **Cost**: $18
- **Time**: 5.5 hours (half a day)
- **Why**: 2× faster than 4× RTX 3090, only +$3
- **Availability**: Newer, may be limited

### For Maximum Speed ($50+)
**Winner: 4× H100 or 4× B200**
- **Cost**: $54-70
- **Time**: 3-4.5 hours (done in morning)
- **Why**: Fastest possible, for urgent deadlines
- **Availability**: Enterprise GPUs, may require higher bids

### For Single GPU
**Winner: RTX 5090**
- **Cost**: $18
- **Time**: 22 hours (~1 day)
- **Why**: Best single GPU value, modern architecture
- **Availability**: Decent

## B200 Specific Analysis

**NVIDIA B200 (Blackwell, 2024)**:
- 192 GB HBM3e memory per GPU
- ~250-300 TPS for 7B models (4.3× faster than RTX 3090)
- **Massive memory overkill** for our use case (we need 20 GB, it has 192 GB)

**4× B200 Setup**:
- **Total time**: ~3.25 hours
- **Cost**: ~$65-78 (if available at $20-24/hr)
- **Memory utilization**: 80 GB / 768 GB = **10% utilized** 😱
- **Verdict**: Overkill unless you're willing to pay for absolute speed

**Is it worth it?**
- vs 4× RTX 3090: Pay **+$55**, save **7.75 hours**
- **Cost per hour saved**: $55 / 7.75 hrs = **$7/hr saved**
- Only worth it if your time is valued at >$7/hr

## Availability Check on vast.ai

### Commonly Available (as of 2026-01)
✅ **RTX 3090** - Very common, $0.35-0.50/hr
✅ **RTX 4090** - Common, $0.60-0.80/hr
✅ **A100 40GB** - Common, $1.20-1.50/hr
⚠️ **RTX 5090** - Newer, $0.80-1.20/hr (availability varies)
⚠️ **H100** - Less common, $2.50-3.50/hr
❌ **B200** - Very rare or unavailable (too new)

### Multi-GPU Availability
- **4× RTX 3090**: Common (many miners/enthusiasts)
- **4× RTX 4090**: Less common (high-end setups)
- **4× RTX 5090**: Rare (brand new)
- **4× A100**: Available (cloud providers)
- **4× H100**: Rare (enterprise only)
- **4× B200**: Extremely rare (just released)

## Practical Recommendation

### Check vast.ai for these in order:

1. **4× RTX 5090** (if available <$4/hr total) ⭐ Best speed/cost
2. **4× RTX 3090** (if 5090 not available) ⭐ Best value
3. **Single RTX 5090** (if multi-GPU not available)
4. **4× A100 40GB** (if on cloud provider credits)
5. **Single H100** (if nothing else available and you have budget)

### Search Commands

```bash
# Search for 4× RTX 5090 (best speed/cost)
vastai search offers "gpu_name = RTX_5090 num_gpus >= 4 reliability > 0.95 inet_down > 200" --order dlperf_usd

# Search for 4× RTX 3090 (best value)
vastai search offers "gpu_name = RTX_3090 num_gpus >= 4 reliability > 0.95 inet_down > 200" --order dlperf_usd

# Search for any 4× GPU with 24+ GB
vastai search offers "gpu_ram >= 24 num_gpus >= 4 reliability > 0.95 inet_down > 200" --order dlperf_usd
```

## Summary Table: Your Decision Matrix

| Priority | GPU Setup | Time | Cost | When to Choose |
|----------|-----------|------|------|----------------|
| **Cheapest** | 4× RTX 3090 | 11h | $15 | Default choice |
| **Fastest/dollar** | 4× RTX 5090 | 5.5h | $18 | If available |
| **Fastest overall** | 4× H100 | 4.5h | $54 | Urgent deadline |
| **Bleeding edge** | 4× B200 | 3.25h | $70 | Money is no object |

**Bottom line**:
- **Start with**: Search for 4× RTX 5090 (best of both worlds)
- **Fallback to**: 4× RTX 3090 (reliable and cheap)
- **Avoid**: Single GPU setups (worse value)
- **Skip**: 4× B200 unless you need results in 3 hours
