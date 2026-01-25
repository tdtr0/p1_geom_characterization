# ManiVer: Geometric Signatures of Correct Computation in LLMs

**Project**: ManiVer (Manifold Verification)
**Core Question**: Do correct solutions have distinguishable dynamical signatures, and do these signatures share structure across verifiable domains?

## Theoretical Framework (Updated 2026-01-17)

We adopt an **interpolation-centric view** (Allen-Zhu & Li, 2024):

- Transformers compute smooth functions — no "reasoning mode" vs "recall mode"
- All computation is interpolation through the representation manifold
- What differs is the *region* and *dynamics* of manifold traversal

**We avoid cognitive framing.** We don't detect "reasoning" — we characterize the geometry of correct solutions.

**Key theoretical connections**:
| Concept | Source | Our Application |
|---------|--------|-----------------|
| Everything is interpolation | Allen-Zhu & Li (2024) | Don't detect "reasoning" — characterize interpolation geometry |
| Curvature regimes | Merullo et al. (2025) | High-SV = distributed; low-SV = localized (proxy for curvature) |
| Attractor dynamics | Ren & Liu (2026) | Correct solutions find right attractors; incorrect get trapped |
| Belief state geometry | Shai et al. (2024) | Residual stream represents belief states |
| Menger curvature | Zhou et al. (Oct 2025) | Curvature captures logical structure beyond surface semantics |

## 🔄 Current Phase: Phase 2 - Trajectory Collection with Correctness Labels

**Status**: Collection 92% Complete (11/12 files)
**Objective**: Collect activation trajectories with correctness labels to test H1 (distinguishable trajectories) and H2 (domain-invariant signatures)

### Phase 2 Collection Results (2026-01-15)

**Completed**: 11/12 trajectory files uploaded to Backblaze B2 (~52GB total)

| Model | GSM8K | HumanEval | LogiQA | Status |
|-------|-------|-----------|--------|--------|
| olmo3_base | ⚠️ 7.5KB (corrupted) | ✅ 2.71GB | ✅ 13.05GB | 2/3 |
| olmo3_sft | ✅ 4.16GB | ✅ 2.71GB | ✅ 6.22GB | 3/3 |
| olmo3_rl_zero | ✅ 4.16GB | ✅ 4.97GB | ✅ 4.97GB | 3/3 |
| olmo3_think | ✅ 4.16GB | ✅ 2.71GB | ✅ 4.72GB | 3/3 |

**B2 Location**: `b2://ml-activations-store/trajectories/`

**Missing Data**: `olmo3_base/gsm8k` - corrupted due to HDF5 gzip filter error on sample 1. Need to recollect.

**Collection Details**:
- 500 samples per task (GSM8K, LogiQA), 164 samples (HumanEval)
- 16 layers collected (even layers 0-30)
- Trajectory shape: (n_samples, 512, 16, 4096)
- All models used same prompts (seed=42)
- Runtime: ~8 hours on 4× RTX 5090 ($1.43/hr)

---

## 📁 Project Structure

```
ManiVer/
├── docs/                          # All documentation
│   ├── plans/                     # Phase plans and algorithm docs
│   │   ├── PHASE1_DETAILED_PLAN.md
│   │   ├── PHASE2_DETAILED_PLAN.md
│   │   ├── PHASE3_DETAILED_PLAN.md
│   │   ├── PHASE4_DETAILED_PLAN.md
│   │   ├── PHASE5_DETAILED_PLAN.md
│   │   ├── phase1_implementation_plan.md
│   │   ├── master_algorithm.md
│   │   └── archive_transfer_correlation_plan.md
│   ├── guides/                    # Setup and usage guides
│   │   ├── PHASE2_PIPELINE.md    # Complete Phase 2 pipeline guide
│   │   ├── SLURM_QUICKSTART.md   # **SLURM quick reference**
│   │   ├── SLURM_CLUSTER_GUIDE.md  # **SLURM H100 cluster guide (detailed)**
│   │   ├── VLLM_GPU_GUIDE.md     # **vLLM GPU compatibility & testing**
│   │   ├── B2_SETUP.md           # Backblaze B2 + Cloudflare setup
│   │   └── B2_QUICKSTART.md      # Quick command reference
│   └── paper/                     # Research paper materials
│       ├── RESEARCH_PLAN.md      # Main hypotheses (H1-H5)
│       ├── LITERATURE_REVIEW_SHORT.md
│       ├── LITERATURE_REVIEW_LONG.md
│       ├── TRAJECTORY_GEOMETRY_CRITIQUE.md
│       └── geometric_compression_research_plan.md
│
├── experiments/                   # Subexperiments (self-contained)
│   ├── aha_moment/               # Phase transition at correction points
│   │   ├── README.md             # Experiment overview
│   │   ├── collect_thinking_traces.py
│   │   ├── analyze_pivot_points.py
│   │   └── data/                 # Local data for this experiment
│   └── svd_reasoning_separability/  # **NEW: Linear separability test**
│       ├── README.md             # Experiment overview
│       ├── analyze_svd_delta.py  # Main analysis script
│       ├── dashboard.sh          # Monitoring dashboard
│       └── results/              # Output plots and JSON
│
├── scripts/                       # Executable scripts
│   ├── collection/                # Data collection scripts
│   │   ├── collect_trajectories_with_labels.py  # Main Phase 2 script
│   │   ├── collect_logiqa_vllm_fully_optimized.py  # **FULLY OPTIMIZED vLLM (all 6 bottlenecks fixed)**
│   │   ├── collect_logiqa_optimized.py          # Optimized HF batching (4 bottlenecks fixed)
│   │   ├── collect_logiqa_vllm.py               # vLLM batched inference (deprecated - memory)
│   │   ├── test_vllm_quick.sh                   # Quick vLLM test (10 samples)
│   │   ├── test_optimized_quick.sh              # Quick optimized test
│   │   ├── collect_activations.py
│   │   ├── collect_single_model.py
│   │   ├── collect_single_logiqa.py             # Single-model LogiQA collection
│   │   ├── test_logiqa_collection.py            # Test LogiQA (3 samples)
│   │   ├── collect_trajectories_half_layers.py
│   │   ├── run_phase2_pipeline.sh               # **MAIN PIPELINE**
│   │   ├── test_phase2_pipeline.sh              # Test with 2 samples
│   │   ├── run_labeled_collection.sh
│   │   ├── run_with_restart.sh
│   │   └── tmux_collect.sh
│   ├── analysis/                  # Analysis scripts
│   │   ├── run_analysis.py
│   │   ├── curvature_and_stats.py
│   │   ├── check_layer_smoothness.py
│   │   └── verify_pipeline.py
│   ├── storage/                   # B2 upload/download
│   │   ├── b2_upload.py          # Upload to Backblaze B2
│   │   ├── b2_download.py        # Download from B2
│   │   └── setup_b2_on_vastai.sh # Auto B2 setup
│   ├── deployment/                # vast.ai & SLURM management
│   │   ├── check_slurm_storage.sh   # **SLURM: Check storage quotas**
│   │   ├── setup_slurm_env.sh    # **SLURM: One-time environment setup**
│   │   ├── test_logiqa_slurm.sbatch  # **SLURM: Test job (10 samples)**
│   │   ├── run_logiqa_slurm.sbatch  # **SLURM: Full job (3 models)**
│   │   ├── monitor_slurm_job.sh  # **SLURM: Monitor job progress**
│   │   ├── vast_launcher.py      # vast.ai: Search/launch/destroy instances
│   │   ├── check_collection_status.sh
│   │   ├── connect_and_collect.sh
│   │   ├── monitor_collection.sh
│   │   ├── run_collection_on_vastai.sh
│   │   └── cleanup_smallworld.sh
│   └── reorganize_project.sh      # This script reorganized everything
│
├── src/                           # Core library code
│   ├── __init__.py
│   ├── activation_collector.py    # Collect activations during generation
│   ├── checkpointing.py           # Checkpoint management
│   ├── geometric_measures.py      # Curvature, path signatures
│   └── task_data.py               # GSM8K, HumanEval, LogiQA loaders
│
├── configs/                       # Configuration files
│   ├── models.yaml                # Model definitions
│   ├── b2-configs.txt             # B2 credentials (DO NOT COMMIT)
│   └── gpu_names_cache.json       # GPU name cache
│
├── data/                          # Data storage (gitignored)
│   ├── trajectories/              # Collected activations
│   │   ├── olmo3_base/
│   │   ├── olmo3_sft/
│   │   ├── olmo3_rl_zero/
│   │   └── olmo3_think/
│   ├── checkpoints/               # Collection checkpoints
│   │   └── labeled_*.json
│   └── logs/                      # Collection logs
│       └── *.log
│
├── results/                       # Analysis results
│   ├── phase1_summary.json
│   ├── statistical_tests.json
│   └── figures/
│
├── notebooks/                     # Jupyter notebooks (exploratory)
│
├── README.md                      # Project overview
├── claude.md                      # **THIS FILE** - Instructions for Claude
├── VASTAI_GUIDE.md               # vast.ai login, costs, GPU selection
└── sync.sh                        # Sync to eyecog server
```

---

## 📋 Quick Navigation

### For Understanding the Project
- **Research Plan**: [docs/paper/RESEARCH_PLAN.md](docs/paper/RESEARCH_PLAN.md) - Main hypotheses (H1-H5)
- **Literature Reviews**:
  - [docs/paper/LITERATURE_REVIEW_SHORT.md](docs/paper/LITERATURE_REVIEW_SHORT.md) - Concise review
  - [docs/paper/LITERATURE_REVIEW_LONG.md](docs/paper/LITERATURE_REVIEW_LONG.md) - Extended analysis
- **Master Algorithm**: [docs/plans/master_algorithm.md](docs/plans/master_algorithm.md) - Complete project structure

### For Phase 2 Collection
- **Pipeline Guide**: [docs/guides/PHASE2_PIPELINE.md](docs/guides/PHASE2_PIPELINE.md) - **START HERE**
- **Phase 2 Plan**: [docs/plans/PHASE2_DETAILED_PLAN.md](docs/plans/PHASE2_DETAILED_PLAN.md) - Week-by-week breakdown
- **Main Script**: [scripts/collection/run_phase2_pipeline.sh](scripts/collection/run_phase2_pipeline.sh) - Collect + Upload
- **Collection Script**: [scripts/collection/collect_trajectories_with_labels.py](scripts/collection/collect_trajectories_with_labels.py)

### For SLURM Cluster (H100)
- **Quick Start**: [docs/guides/SLURM_QUICKSTART.md](docs/guides/SLURM_QUICKSTART.md) - **Fast reference for setup & monitoring**
- **Complete Guide**: [docs/guides/SLURM_CLUSTER_GUIDE.md](docs/guides/SLURM_CLUSTER_GUIDE.md) - Detailed documentation
- **vLLM Collection**: [scripts/collection/collect_logiqa_vllm_fully_optimized.py](scripts/collection/collect_logiqa_vllm_fully_optimized.py) - Fully optimized (all 6 bottlenecks fixed)
- **Scripts**:
  - [scripts/deployment/check_slurm_storage.sh](scripts/deployment/check_slurm_storage.sh) - Check quotas & available space
  - [scripts/deployment/setup_slurm_env.sh](scripts/deployment/setup_slurm_env.sh) - One-time environment setup
  - [scripts/deployment/test_logiqa_slurm.sbatch](scripts/deployment/test_logiqa_slurm.sbatch) - Test job (10 samples, ~10 min)
  - [scripts/deployment/run_logiqa_slurm.sbatch](scripts/deployment/run_logiqa_slurm.sbatch) - Full job (500 samples × 3 models, ~1.5-2 hrs)
  - [scripts/deployment/monitor_slurm_job.sh](scripts/deployment/monitor_slurm_job.sh) - Monitor job progress

### For vast.ai Setup
- **vast.ai Guide**: [docs/guides/VASTAI_GUIDE.md](docs/guides/VASTAI_GUIDE.md) - **Login, costs, GPU selection**
- **Collection Guide**: [docs/guides/VASTAI_COLLECTION_GUIDE.md](docs/guides/VASTAI_COLLECTION_GUIDE.md) - **Practical pitfalls & solutions**
- **GPU Optimization**: [docs/guides/GPU_OPTIMIZATION.md](docs/guides/GPU_OPTIMIZATION.md) - **Detailed pareto analysis**
- **GPU Finder**: [scripts/deployment/find_optimal_gpu.sh](scripts/deployment/find_optimal_gpu.sh) - Find best available GPU
- **Launcher**: [scripts/deployment/vast_launcher.py](scripts/deployment/vast_launcher.py)
- **B2 Setup**: [docs/guides/B2_SETUP.md](docs/guides/B2_SETUP.md)
- **B2 Quick Reference**: [docs/guides/B2_QUICKSTART.md](docs/guides/B2_QUICKSTART.md)

### For Subexperiments
- **Aha Moment Analysis**: [experiments/aha_moment/](experiments/aha_moment/) - Phase transition at correction points

### For Results
- **Phase 1 Results**: [docs/plans/phase1_implementation_plan.md](docs/plans/phase1_implementation_plan.md) lines 21-73
- **Current Results**: `results/` directory

---

## Phase 1 Summary (Complete)

Phase 1 established that RLVR and SFT models have different static geometry:
- **RL-Zero**: 98.6% subspace preservation vs base
- **SFT/Think**: ~50% preservation
- **Conclusion**: Training method affects geometry, but static analysis doesn't capture reasoning quality

## Models

| Key | Model | Training | Size |
|-----|-------|----------|------|
| `olmo3_base` | allenai/Olmo-3-1025-7B | Base | 7B |
| `olmo3_sft` | allenai/Olmo-3-7B-Think-SFT | SFT | 7B |
| `olmo3_rl_zero` | allenai/Olmo-3-7B-RL-Zero-General | RL-Zero | 7B |
| `olmo3_think` | allenai/OLMo-3-7B-Think | SFT+DPO+RLVR | 7B |
| `deepseek_r1` | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | RLVR distilled | 8B |

## Compute Requirements

### Storage Architecture
```
vast.ai Instance (ephemeral)     Backblaze B2 (persistent)
┌─────────────────────────┐      ┌─────────────────────────┐
│ /workspace/maniver/     │      │ ml-activations-store    │
│ data/trajectories/      │ ───► │ phase2_YYYYMMDD/        │
│ (200-300 GB local SSD)  │ B2   │ (~56 GB per run)        │
└─────────────────────────┘      └─────────────────────────┘
```

### GPU Requirements & Costs

| GPU | VRAM | Time (hrs) | $/hr | Total Cost | Recommendation |
|-----|------|------------|------|------------|----------------|
| **4× RTX 3090** | 96 GB | **18-20** | $1.20 | **$22-24** | 🏆 Best value |
| **A100 40GB** | 40 GB | 38-45 | $1.30 | $49-58 | Balanced |
| **H100 80GB** | 80 GB | **25-30** | $2.50 | $62-75 | Fastest |
| **RTX 4090** | 24 GB | 45-55 | $0.60 | $27-33 | Budget |
| **RTX 3090** | 24 GB | 60-75 | $0.35 | $21-26 | Cheapest (slowest) |

**Recommended**:
- **Best value**: 4× RTX 3090 (parallel, $22, 1 day)
- **Fastest**: H100 ($65, 1 day)
- **Balanced**: A100 40GB ($50, 1.5 days)

See [VASTAI_GUIDE.md](VASTAI_GUIDE.md) for detailed cost analysis and GPU selection.

### Bandwidth
- Minimum: 200 Mbps up/down for B2 uploads

---

## Development Workflow

### Architecture
```
Local Machine          eyecog Server           vast.ai
(VSCode/Claude)   <->  (Dev/Test/Code)    ->   (Heavy Compute)
      |                      |                      |
      v                      v                      v
   Edit code            Test small batch      Run full collection
   Search vast.ai       Prototype scripts     Auto-upload to B2
   Manage instances     Git push              Destroy when done
```

### Directory Mapping
- **Local**: `~/CascadeProjects/ManiVer/main/`
- **eyecog**: `~/p1_geom_characterization/`
- **vast.ai**: `/workspace/maniver/`

### Sync Commands
```bash
# Local -> eyecog (for development/testing)
./sync.sh

# eyecog -> GitHub (after testing)
ssh eyecog "cd ~/p1_geom_characterization && git add -A && git commit -m 'msg' && git push"

# vast.ai auto-pulls from GitHub on start
```

---

## 🖥️ Eyecog Server Access

### SSH Access
```bash
ssh eyecog
```

### Data Locations (Updated 2026-01-19)

**IMPORTANT**: Large trajectory data has been moved to `/data/thanhdo/` to free up /home space.

| Data | Location | Size | Notes |
|------|----------|------|-------|
| **trajectories_0shot** | `/data/thanhdo/trajectories_0shot/` | ~38GB | Symlinked from ~/p1.../data/ |
| **trajectories_8shot** | `/data/thanhdo/trajectories_8shot/` | ~57GB | Symlinked from ~/p1.../data/ |
| **trajectories** (current) | `~/p1_geom_characterization/data/trajectories/` | ~16GB | In home dir |
| **experiments** | `~/p1_geom_characterization/experiments/` | ~11GB | aha_moment data |

**Symlinks in place** - code paths still work:
```bash
~/p1_geom_characterization/data/trajectories_0shot -> /data/thanhdo/trajectories_0shot
~/p1_geom_characterization/data/trajectories_8shot -> /data/thanhdo/trajectories_8shot
```

### Corrupted Files (Deleted 2026-01-19)
These 0shot logiqa files were truncated/corrupted and have been deleted:
- `olmo3_sft/logiqa_trajectories.h5`
- `olmo3_rl_zero/logiqa_trajectories.h5`
- `olmo3_think/logiqa_trajectories.h5`

Only `olmo3_base/logiqa_trajectories.h5` survives in 0shot.

### Available Conda Environments
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base           # Main env with torch, transformers, h5py
conda activate geometric_transfer  # Alternative (5.6GB)
conda activate swin_improv    # Vision models (8.1GB)
```

### GPUs
```bash
nvidia-smi  # Check availability
# GPU 0: RTX 3090 24GB
# GPU 1: RTX 3090 24GB
```

---

## ⚠️ SLURM Cluster Access (CRITICAL SAFETY RULES)

### SSH Access
```bash
ssh ai_inst
```

### 🚨 CRITICAL SAFETY CONSTRAINTS 🚨

**THIS IS A LOGIN NODE - NEVER RUN COMPUTE JOBS DIRECTLY ON IT**

1. **ALWAYS submit jobs via SLURM** (`sbatch`, `srun`)
2. **NEVER run Python scripts directly** on the login node
3. **NEVER run model inference** on the login node
4. **ALWAYS ask user before submitting ANY job** - even test jobs
5. **Only allowed on login node**: File operations, text editing, job submission

**Violation of these rules can cause system issues and trouble for the user.**

### Available GPUs

| Node | GPUs | VRAM | Cores Available | Best For |
|------|------|------|-----------------|----------|
| **h100** | 4× H100 80GB | 80 GB | 60 | **Recommended** - Fast, large VRAM |
| quadro1/2 | 8× Quadro RTX 8000 | 48 GB | ? | Alternative if h100 busy |
| tesla1/2 | 8× Tesla V100 | 32 GB | ? | Too small for vLLM+HF |

**Recommended node**: `h100` (80GB VRAM, fastest inference)

### SLURM Job Workflow

```bash
# 1. SSH to login node
ssh ai_inst

# 2. Prepare job files (via Claude/local editing)
# - setup_slurm.sh (install dependencies)
# - run_collection.sbatch (SLURM job file)

# 3. Submit job (ALWAYS ask user first!)
sbatch run_collection.sbatch

# 4. Monitor job
squeue -u $USER
tail -f slurm-<jobid>.out

# 5. Check results after completion
ls -lh /home/$USER/maniver/ManiVer/data/trajectories/
```

### Monitoring Approach (for Claude)

When monitoring SLURM jobs, Claude will:

1. **SSH to ai_inst** (login node only)
2. **Run monitoring script**: `bash scripts/deployment/monitor_slurm_job.sh <job_id>`
3. **Check progress** from stdout logs
4. **Report status** to user periodically
5. **Diagnose errors** if job fails

**Monitoring intervals**:
- Test job (10 samples): Check every 5 minutes
- Full job (500 samples): Check every 15-30 minutes

**No email notifications** - Claude checks logs directly via SSH.

---

## Quick Start: Phase 2 Collection

### Step 1: vast.ai Login (One-Time)

```bash
# Install CLI
pip install vastai

# Login (get API key from https://cloud.vast.ai/account)
vastai set api-key <your_api_key>

# Verify
vastai show user
```

### Step 2: Launch Instance

```bash
# Search for best GPU
python scripts/deployment/vast_launcher.py search --sort-by cost

# Launch (auto-setup, clone repo, install deps)
python scripts/deployment/vast_launcher.py launch
```

### Step 3: Run Collection

```bash
# SSH into instance
vastai ssh <instance_id>

# Run full pipeline (collect + upload to B2)
cd /workspace/maniver
bash scripts/collection/run_phase2_pipeline.sh

# Monitor progress
tail -f data/logs/phase2_collection_*.log
```

### Step 4: Destroy Instance

```bash
# On local machine
python scripts/deployment/vast_launcher.py destroy <instance_id>
```

**Total time**: 18-75 hours (depending on GPU)
**Total cost**: $22-75 (see VASTAI_GUIDE.md)

---

## Cloud Storage (B2)

### Bucket Configuration
- **Bucket**: `ml-activations-store`
- **Endpoint**: `https://f005.backblazeb2.com`
- **CDN**: `https://activations.maniact.space` (once DNS configured)

### B2 Commands

```bash
# Upload trajectories
python scripts/storage/b2_upload.py

# List available runs
python scripts/storage/b2_download.py --list-only

# Download for analysis
python scripts/storage/b2_download.py --remote-prefix phase2_20260114/trajectories
```

See [docs/guides/B2_SETUP.md](docs/guides/B2_SETUP.md) for detailed setup.

---

## Notes for Claude

### ⚠️ CRITICAL: File Generation Rules

**NEVER generate files directly in `/main/` (the project root).** Always use the appropriate subdirectory:

1. **Plans/Specs** → `docs/plans/`
2. **Setup Guides** → `docs/guides/`
3. **Research Papers** → `docs/paper/`
4. **Collection Scripts** → `scripts/collection/`
5. **Analysis Scripts** → `scripts/analysis/`
6. **Storage Scripts** → `scripts/storage/`
7. **Deployment Scripts** → `scripts/deployment/`
8. **Core Library** → `src/`
9. **Config Files** → `configs/`
10. **Data** → `data/` (gitignored)
11. **Results** → `results/`
12. **Subexperiments** → `experiments/<experiment_name>/` (self-contained)

**For subexperiments**: Create a folder under `experiments/` with its own README.md, scripts, and data folder. Keep experiments self-contained.

**IMPORTANT**: Update this file (`claude.md`) when you create new files!

### Development Rules
1. **Edit locally** - Use VSCode/Claude for all code changes
2. **Test on eyecog** - Small batch (N=10) first
3. **Deploy to vast.ai** - Only after testing
4. **Always confirm** - Before renting instances

### Code Quality
- Store activations in HDF5 with gzip compression
- Use float16 for storage efficiency
- Checkpoint every 25 samples for fault tolerance
- Log correctness rates during collection

### Data Collection
- **Trajectories**: Even layers only [0,2,4,...,30] = 16 layers
- **Aggregation**: Full sequence (not last_token)
- **Correctness labels**: Required for H1/H2 tests

### Correctness Checking
- **GSM8K**: Extract `#### <number>`, exact numerical match
- **LogiQA**: Extract A/B/C/D letter, exact match
- **HumanEval**: Syntax check (full test execution deferred)

---

## Core Hypotheses (H1-H5) — Status Update (2026-01-24)

**H1**: Correct vs incorrect solutions have distinguishable trajectory dynamics — ✅ TRUE via linear probe (AUC 0.68-0.75)
**H2**: Dynamical signatures share structure across domains — ⚠️ WEAK (d ~ 0.4, not universal)
**H3**: Signatures correlate with human judgments on non-verifiable domains — Not tested
**H4**: Trajectory interventions can improve task performance — Not tested
**H5**: Correct solutions have more stable dynamics (Lyapunov) — ❌ **FAILED** (worse than linear probe baseline)

### Original Concepts

- **Trajectory**: Activation path through layers (seq_len, n_layers, d_model)
- **Path signature**: Reparameterization-invariant trajectory features
- **Correctness label**: Boolean (model answer matches ground truth)
- **Verifiable domains**: Math (GSM8K), Code (HumanEval), Logic (LogiQA)
- **Non-verifiable domains**: Philosophy, ethics, strategy (no ground truth)

### New Dynamical Systems Concepts (Phase 3)

- **Vector field**: Layer transition dynamics v(x) = x_{l+1} - x_l
- **Helmholtz decomposition**: Split into potential (gradient) + rotational (curl) flow
- **Lyapunov exponent**: Rate of trajectory divergence/convergence (stability)
- **Attractor basin**: Region of state space converging to fixed point
- **Activation regime**: High-SV (distributed) vs low-SV (localized) — proxy for curvature

### Phase 3 Analysis Methods — Results

| Method | What it Measures | Hypothesis | Status |
|--------|------------------|------------|--------|
| MARBLE vector field | Flow structure | Correct = more potential | Not tested |
| Lyapunov exponents | Trajectory stability | Correct = more stable | ❌ **FAILED** |
| Attractor analysis | Convergence targets | Different basins | Not tested |
| Activation regime | Weight direction usage | Correct = more distributed | ❌ Failed (opposite) |
| Path signatures | Trajectory shape | Correct = more structured | Not tested |
| Menger curvature | Local bending | Lower curvature | ❌ Failed (architectural) |
| **Linear probe** | Static geometry | — | ✅ **WORKS** (AUC 0.75) |

**Key finding**: Static geometry (linear probe on mean activation) beats ALL dynamical measures tested.

---

## Current Priorities

1. ✅ **Project reorganization** - Clean folder structure
2. ✅ **B2 storage setup** - Backblaze + Cloudflare CDN
3. ✅ **Pipeline scripts** - Automated collect + upload
4. ✅ **Phase 2 collection** - 11/12 files collected and uploaded to B2
5. ✅ **GPU bottleneck analysis** - Identified 4 critical bottlenecks, documented fixes
6. 🔄 **Optimized collection script** - Created, testing on 10 samples
7. 🔄 **Rerun LogiQA collection** - Using optimized script (2-3 hrs vs 8-10 hrs)
8. ⏳ **Recollect olmo3_base/gsm8k** - After LogiQA completes
9. ⏳ **H1 testing** - Within-domain classification (can start with 11 files)
10. ⏳ **H2 testing** - Cross-domain transfer (critical test)

---

## File Update Log

**2026-01-19**:
- **Added SVD Linear Separability Experiment** — Motivating negative result for dynamical analysis
  - Added `experiments/svd_reasoning_separability/` - Complete experiment folder
  - Added `experiments/svd_reasoning_separability/analyze_svd_delta.py` - SVD comparison script
  - Added `experiments/svd_reasoning_separability/dashboard.sh` - Real-time monitoring
  - Added `notebooks/working_notes/SVD_LINEAR_SEPARABILITY_FINDINGS.md` - Detailed findings
  - **Key result**: Tail eigenvectors change 3-8x MORE than top eigenvectors (opposite of separable reasoning)
  - **Interpretation**: RLVR preserves top eigenvectors (core structure), refines tail (fine-grained)
  - **Implication**: Linear methods don't capture reasoning; motivates dynamical/flow analysis
  - Updated `docs/paper/RESEARCH_PLAN.md` with new "Motivating Result" section
  - Runtime: ~12 minutes on eyecog (randomized SVD)

**2026-01-18** (evening):
- **Created fully optimized vLLM collection pipeline for SLURM cluster** (H100)
  - Added `scripts/collection/collect_logiqa_vllm_fully_optimized.py` - Fixes ALL 6 GPU bottlenecks
  - Added `scripts/deployment/setup_slurm_env.sh` - One-time SLURM environment setup
  - Added `scripts/deployment/run_logiqa_slurm.sbatch` - SLURM job file with auto-upload to B2
  - Added `docs/guides/SLURM_CLUSTER_GUIDE.md` - Complete SLURM usage guide
  - Added `docs/guides/SLURM_QUICKSTART.md` - Quick reference for setup & monitoring
  - **Key innovation**: Dual-model approach (vLLM for generation + HF for activations)
  - **Expected performance**: 1.5-2 hours for 3 models × 500 samples (vs 12.5 hrs sequential)
  - **Bottlenecks fixed**: (1) GPU-only tensors, (2) Batched activation, (3) Async I/O, (4) Memory cleanup, (5) GPU tokenization via vLLM, (6) Fast vLLM generation (3-5x speedup)
  - **Auto-upload**: Job automatically uploads to B2 after collection completes
  - **Safety**: Documented critical SLURM constraints (login node vs compute node)
- **Added storage checking and test infrastructure for SLURM**
  - Added `scripts/deployment/check_slurm_storage.sh` - Check storage quotas before setup
  - Added `scripts/deployment/test_logiqa_slurm.sbatch` - Test job (10 samples, ~10 min validation)
  - Added `scripts/deployment/monitor_slurm_job.sh` - Monitor job progress by reading logs
  - Updated all SLURM scripts to use configurable `WORK_DIR` (support /home/, /scratch/, /work/)
  - **Monitoring approach**: Claude SSH checks logs periodically, no email notifications needed

**2026-01-18** (afternoon):
- **Added Menger curvature analysis to Phase 3** (Zhou et al., 2025)
  - Added Section 6 to PHASE3_DETAILED_PLAN.md: "Menger Curvature Analysis"
  - Reference: Zhou et al. "The Geometry of Reasoning" (arXiv:2510.09782)
  - Key finding: Curvature (2nd order) captures logical structure > surface semantics
  - Tests: cross-domain curvature correlation, curvature profiles for correct vs incorrect
  - Updated Week 6 timeline: Days 5-6 for Menger curvature analysis
  - Added deliverables: `h2_menger_curvature.csv`, `h2_menger_correlation.csv`
- **Updated aha_moment experiment** to reuse Phase 2 data for Experiment B
  - Phase 2 HDF5 files contain `model_outputs` — no new collection needed
  - Experiment B now CPU-only (0 GPU hours)
  - Added pivot detection options: regex, zero-shot classifier, free LLM API

**2026-01-18** (morning):
- **Discovered and documented 4 GPU bottlenecks** during batched collection
  - Bottleneck 1: GPU→CPU transfer during generation (25% idle time)
  - Bottleneck 2: Sequential forward passes for activation collection (75% of batch time)
  - Bottleneck 3: Blocking HDF5 writes (15-20% idle time)
  - Bottleneck 4: Memory fragmentation causing slowdown over time
  - Documented in `docs/plans/master_algorithm.md` - Section "GPU Optimization Lessons"
- **Created optimized collection script** (fixes all 4 bottlenecks)
  - Added `scripts/collection/collect_logiqa_optimized.py` - GPU-optimized with pipelined execution
  - Added `scripts/collection/test_optimized_quick.sh` - Quick 10-sample test
  - Expected speedup: 4-5x vs sequential (2-3 hours vs 12.5 hours for 500 samples)
  - Key changes: GPU-only tensors, batched activation collection, async I/O, memory cleanup
  - Will rerun collection after testing completes
- **Created vLLM acceleration pipeline** (NOT USED - memory constraints)
  - Added `scripts/collection/collect_logiqa_vllm.py` - Batched inference with vLLM
  - Added `scripts/collection/test_vllm_quick.sh` - Quick 10-sample test script
  - Added `docs/guides/VLLM_GPU_GUIDE.md` - Complete GPU compatibility guide
  - Issue: Requires 28GB (vLLM + HF model), doesn't fit on 24GB RTX 4090s
  - Abandoned in favor of optimized HF batching
- **Batched LogiQA collection in progress** (inefficient, will be replaced)
  - Running: olmo3_sft (box1 GPU 0), olmo3_rl_zero (box1 GPU 1)
  - Started: ~1:15 AM, Batch 5/125 each
  - ETA: ~8-10 hours (but GPU utilization only 0-10%!)
  - Will stop and rerun with optimized script after testing

**2026-01-17** (night):
- **Integrated error-detection direction analysis into Phase 3** (Wynroe-style)
  - Added Section 5 to PHASE3_DETAILED_PLAN.md: "Error-Detection Direction Analysis"
  - Uses existing Phase 2 data (correct vs incorrect labels) — zero additional GPU time
  - Tests: direction existence, layer profile, model comparison, cross-domain transfer
  - Updated Week 7 timeline to include direction analysis (Days 1-4)
  - Added new deliverables: `h2_error_direction.csv`, `h2_direction_transfer.csv`
- Updated `experiments/aha_moment/README.md` to reflect Option C (Phase 3 integration)
  - Primary: Wynroe replication on OLMo 3 family using Phase 2 data
  - Secondary (future): Pivot token dynamics if primary analysis is promising

**2026-01-17** (evening):
- **Directory cleanup**: Moved loose files from /main/ to proper subdirectories
  - GPU_OPTIMIZATION.md, VASTAI_GUIDE.md, etc. → docs/guides/
  - PHASE2_EXECUTION_PLAN.md → docs/plans/
  - Shell scripts → scripts/deployment/
- Added `experiments/` folder for self-contained subexperiments
- Created `experiments/aha_moment/` for phase transition analysis
- Updated CLAUDE.md with explicit rule: NEVER generate files in /main/ root

**2026-01-17** (morning):
- **Major theoretical reframing**: Adopted interpolation-centric view (Allen-Zhu)
- Updated RESEARCH_PLAN.md with new theoretical framework and reframed hypotheses
- Updated PHASE3_DETAILED_PLAN.md with dynamical systems analyses:
  - MARBLE-style vector field decomposition
  - Lyapunov exponent analysis
  - Attractor analysis
  - Activation regime analysis (proxy for Goodfire curvature)
- Updated master_algorithm.md with new concepts and references
- Added connections to: Allen-Zhu (interpolation), Merullo et al. (curvature regimes), Ren & Liu (attractor dynamics), Shai et al. (belief states)
- Clarified that Goodfire K-FAC requires gradient statistics — we use structural proxies instead
- Phase 3 now spans 4 weeks (was 2 weeks)

**2026-01-15**:
- **Phase 2 collection completed** (11/12 files, ~52GB) on vast.ai 4× RTX 5090
- Uploaded all trajectories to Backblaze B2 (`b2://ml-activations-store/trajectories/`)
- Fixed HDF5 file locking issues with `HDF5_USE_FILE_LOCKING=FALSE`
- Added `run_phase2_parallel.sh` for 4-GPU parallel collection
- Added `auto_complete_and_upload.sh` for automated monitoring
- Added `monitor_and_restart.sh` for crash recovery
- **Known issue**: `olmo3_base/gsm8k` corrupted (HDF5 gzip filter error) - needs recollection
- Total cost: ~$12 (8 hours × $1.43/hr)

**2026-01-14**:
- Reorganized project structure (new folders: docs/, scripts/{collection,analysis,storage,deployment}/)
- Added VASTAI_GUIDE.md with login, costs, GPU optimization
- Created B2 storage pipeline (b2_upload.py, b2_download.py, setup_b2_on_vastai.sh)
- Added run_phase2_pipeline.sh (integrated collect + upload)
- Added PHASE2_PIPELINE.md (complete pipeline guide)
- Added B2_SETUP.md and B2_QUICKSTART.md
- Moved all plans to docs/plans/
- Moved all guides to docs/guides/
- Updated claude.md with new structure

**2026-01-12**: Created detailed phase plans (PHASE1-5_DETAILED_PLAN.md), literature reviews (SHORT + LONG), updated RESEARCH_PLAN.md

---

## 🤖 Instructions for LLMs

### When You Create New Files

**IMPORTANT**: Update this document (`claude.md`) AND the appropriate section:

- Plans → `docs/plans/` + update Quick Navigation
- Guides → `docs/guides/` + update Quick Navigation
- Scripts → Appropriate `scripts/` subdirectory + update structure
- Results → `results/` + update Quick Navigation

### What NOT to Do

- ❌ Don't create files at project root (use subdirectories)
- ❌ Don't confuse geometric_compression_research_plan.md (background) with main focus
- ❌ Don't focus on RLVR vs SFT (Phase 1 already done)
- ❌ Don't forget to update this file
- ❌ Don't claim we detect "reasoning" — use "correct solution signatures" instead
- ❌ Don't claim true K-FAC curvature analysis — we use structural proxies

### What TO Do

- ✅ Focus on **correct vs incorrect solution geometry** (H1-H5)
- ✅ Use **interpolation-centric framing** (Allen-Zhu view)
- ✅ Implement **dynamical systems analyses** (vector field, Lyapunov, attractors)
- ✅ Implement **confound controls** (difficulty, length, format)
- ✅ Compare to **baselines** (model confidence, semantic entropy)
- ✅ Test **causal interventions** (H4 steering)
- ✅ Be honest about **limitations** (proxy analyses, what we can/can't claim)
- ✅ Document everything clearly in proper folders
