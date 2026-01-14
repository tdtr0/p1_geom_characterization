# ManiVer: Geometric Signatures of Reasoning in LLMs

**Project**: ManiVer (Manifold Verification)
**Core Question**: Can we learn the geometry of correct reasoning from verifiable domains and use it on non-verifiable domains?

## 🔄 Current Phase: Phase 2 - Trajectory Collection with Correctness Labels

**Status**: Ready to Launch
**Objective**: Collect activation trajectories with correctness labels to test H1 (distinguishable trajectories) and H2 (domain-invariant signatures)

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
│   │   ├── B2_SETUP.md           # Backblaze B2 + Cloudflare setup
│   │   └── B2_QUICKSTART.md      # Quick command reference
│   └── paper/                     # Research paper materials
│       ├── RESEARCH_PLAN.md      # Main hypotheses (H1-H5)
│       ├── LITERATURE_REVIEW_SHORT.md
│       ├── LITERATURE_REVIEW_LONG.md
│       ├── TRAJECTORY_GEOMETRY_CRITIQUE.md
│       └── geometric_compression_research_plan.md
│
├── scripts/                       # Executable scripts
│   ├── collection/                # Data collection scripts
│   │   ├── collect_trajectories_with_labels.py  # Main Phase 2 script
│   │   ├── collect_activations.py
│   │   ├── collect_single_model.py
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
│   ├── deployment/                # vast.ai management
│   │   ├── vast_launcher.py      # Search/launch/destroy instances
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

### For vast.ai Setup
- **vast.ai Guide**: [VASTAI_GUIDE.md](VASTAI_GUIDE.md) - **Login, costs, GPU selection**
- **GPU Optimization**: [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) - **Detailed pareto analysis**
- **GPU Finder**: [scripts/deployment/find_optimal_gpu.sh](scripts/deployment/find_optimal_gpu.sh) - Find best available GPU
- **Launcher**: [scripts/deployment/vast_launcher.py](scripts/deployment/vast_launcher.py)
- **B2 Setup**: [docs/guides/B2_SETUP.md](docs/guides/B2_SETUP.md)
- **B2 Quick Reference**: [docs/guides/B2_QUICKSTART.md](docs/guides/B2_QUICKSTART.md)

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

### Folder Structure Rules

When creating new files:

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

## Core Hypotheses (H1-H5)

**H1**: Correct vs incorrect reasoning have distinguishable trajectories (within-domain)
**H2**: The signature is domain-invariant (math → code → logic) - **CRITICAL TEST**
**H3**: Detector works on non-verifiable domains (validated by human judgment)
**H4**: Trajectories can be steered toward correct reasoning (causal intervention)
**H5**: Correct reasoning has lower curvature (exploratory)

### Key Concepts

- **Trajectory**: Activation path through layers (seq_len, n_layers, d_model)
- **Path signature**: Reparameterization-invariant trajectory features
- **Correctness label**: Boolean (model answer matches ground truth)
- **Verifiable domains**: Math (GSM8K), Code (HumanEval), Logic (LogiQA)
- **Non-verifiable domains**: Philosophy, ethics, strategy (no ground truth)

---

## Current Priorities

1. ✅ **Project reorganization** - Clean folder structure
2. ✅ **B2 storage setup** - Backblaze + Cloudflare CDN
3. ✅ **Pipeline scripts** - Automated collect + upload
4. 🔄 **Phase 2 collection** - Launch vast.ai, collect data
5. ⏳ **H1 testing** - Within-domain classification
6. ⏳ **H2 testing** - Cross-domain transfer (critical test)

---

## File Update Log

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

### What TO Do

- ✅ Focus on **correct vs incorrect reasoning geometry** (H1-H5)
- ✅ Implement **confound controls** (difficulty, length, format)
- ✅ Compare to **baselines** (model confidence, semantic entropy)
- ✅ Test **causal interventions** (H4 steering)
- ✅ Document everything clearly in proper folders
