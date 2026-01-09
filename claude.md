# Phase 1: Geometric Characterization of LLM Activations

## Project Overview

This project investigates whether RLVR (Reinforcement Learning with Verifiable Rewards) and SFT (Supervised Fine-Tuning) produce measurably different geometric structures in transformer activation space. This is Phase 1 of a larger research initiative - purely descriptive work to characterize activation manifolds before connecting to transfer performance.

## Core Research Question

**Do RLVR and SFT produce measurably different geometric structures in activation space?**

## Models Under Study

### Primary (OLMo 3 Family - Cleanest Controlled Comparison)
- `allenai/OLMo-3-7B` - Base (pretraining only)
- `allenai/OLMo-3-7B-Instruct` - Base + SFT
- `allenai/OLMo-3-7B-RL-Zero` - Base + RL (no SFT) - **Key model for isolating RL effect**

### Secondary (DeepSeek Distilled)
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- DeepSeek V3 variants

## Project Structure

```
geometric_transfer/
├── configs/
│   └── models.yaml
├── src/
│   ├── activation_collector.py    # Core activation extraction
│   ├── task_data.py               # GSM8K, HumanEval, LogiQA prep
│   ├── geometric_measures.py      # SVD, effective rank, preservation
│   ├── analyze_geometry.py        # Analysis pipeline
│   ├── statistical_tests.py       # Hypothesis testing
│   └── visualize.py               # Plotting functions
├── scripts/
│   ├── verify_pipeline.py
│   ├── collect_activations.py
│   └── run_analysis.py
├── data/
│   ├── activations/               # HDF5 files per model/task
│   └── trajectories/              # Full token trajectories (subset)
├── results/
│   └── figures/
└── notebooks/
```

## Key Geometric Measures

1. **Effective Rank** - Exp(entropy of normalized singular values)
2. **Spectral Decay** - Power-law exponent α of singular value decay
3. **Subspace Preservation** - Principal angles between base/fine-tuned top-k subspaces
4. **Local Curvature** - Perturbation response analysis

## Compute Environment

- **Primary Server**: eyecog (SSH: `ssh eyecog`)
- **GPU**: 2x RTX 3090 24GB (estimated 25-35 GPU hours for Phase 1)
- **Storage**: ~50-100GB total (activations + trajectories + models)

## Dependencies

- Python 3.10
- PyTorch 2.2.0 (CUDA 12.1)
- transformers 4.40.0
- TransformerLens 1.14.0
- h5py, scipy, scikit-learn
- signatory (path signatures)
- wandb (monitoring)

## Implementation Phases

### Weeks 1-2: Infrastructure
- Environment setup on eyecog
- Model download/verification
- Activation collection pipeline
- Pipeline verification on small samples

### Weeks 3-4: Data Collection
- OLMo family activations (500 samples × 3 tasks × 3 models)
- Full trajectories for subset (100 samples for path signatures)
- Quality checks

### Weeks 5-6: Static Geometric Analysis
- Compute geometric measures
- Statistical hypothesis testing (RLVR vs SFT preservation)
- Visualization

## Success Criteria

| Outcome | Signal | Action |
|---------|--------|--------|
| RLVR higher preservation, p<0.01, d>0.5 | Strong | Proceed to Phase 2 |
| RLVR higher preservation, p<0.05, d>0.3 | Moderate | Proceed cautiously |
| No significant difference | Weak | Pivot: analyze what differs |
| SFT higher preservation | Unexpected | Investigate |

## Commands Reference

```bash
# SSH to compute server
ssh eyecog

# Activate environment
conda activate geometric_transfer

# Run pipeline verification
python scripts/verify_pipeline.py

# Collect activations
python scripts/collect_activations.py --output-dir data/activations

# Run analysis
python scripts/run_analysis.py
```

## Development Workflow

### Auto-sync Setup
To keep local, eyecog, and GitHub in sync:

```bash
# On local machine - sync to eyecog
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.h5' --exclude='data/' . eyecog:~/p1_geom_characterization/

# On eyecog - commit and push to GitHub
ssh eyecog "cd ~/p1_geom_characterization && git add . && git commit -m 'Update: <description>' && git push"

# On local machine - pull from GitHub
git pull origin main
```

For continuous development:
1. Edit code locally
2. Sync to eyecog with rsync
3. Test on eyecog
4. Commit and push from eyecog
5. Pull to local

### File Sync Script
Create `sync.sh` for easy syncing:
```bash
#!/bin/bash
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.h5' --exclude='data/' --exclude='wandb/' . eyecog:~/p1_geom_characterization/
```

## Notes for Claude

- Always use TransformerLens for activation extraction (not manual hooks)
- Store activations in HDF5 with gzip compression
- Use float16 for storage efficiency
- Run quality checks after collection (no NaN/Inf, reasonable magnitudes)
- Aggregation: "last_token" for sequence-level representations
- Sync code changes to eyecog before running experiments
