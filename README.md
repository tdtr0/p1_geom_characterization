# Phase 1: Geometric Characterization of LLM Activations

## Overview

This project investigates whether RLVR (Reinforcement Learning with Verifiable Rewards) and SFT (Supervised Fine-Tuning) produce measurably different geometric structures in transformer activation space.

**Core Research Question**: Do RLVR and SFT produce measurably different geometric structures in activation space?

## Models Under Study

### Primary (OLMo 3 Family)
- `allenai/OLMo-3-7B` - Base model
- `allenai/OLMo-3-7B-Instruct` - Base + SFT
- `allenai/OLMo-3-7B-RL-Zero` - Base + RL (no SFT)

### Secondary (DeepSeek Distilled)
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- DeepSeek V3 variants

## Project Structure

```
p1_geom_characterization/
├── configs/           # Model and experiment configurations
├── src/              # Core analysis code
├── scripts/          # Executable scripts for data collection and analysis
├── data/             # Collected activations and trajectories
├── results/          # Analysis outputs and figures
└── notebooks/        # Exploratory notebooks
```

## Quick Start

### Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate geometric_transfer

# Or manually:
conda create -n geometric_transfer python=3.10
conda activate geometric_transfer
pip install -r requirements.txt
```

### Pipeline Verification

```bash
# Verify pipeline works on small sample
python scripts/verify_pipeline.py
```

### Data Collection

```bash
# Collect activations for all models and tasks
python scripts/collect_activations.py --output-dir data/activations
```

### Analysis

```bash
# Run geometric analysis
python scripts/run_analysis.py
```

## Key Geometric Measures

1. **Effective Rank** - Measures dimensionality usage
2. **Spectral Decay** - Power-law exponent of singular value decay
3. **Subspace Preservation** - Principal angles between base/fine-tuned subspaces
4. **Local Curvature** - Perturbation response analysis

## Compute Requirements

- GPU: 2x RTX 3090 (24GB each)
- Estimated compute: 25-35 GPU hours
- Storage: ~8GB

## Citation

```bibtex
@misc{p1_geom_characterization,
  title={Geometric Characterization of LLM Activations: RLVR vs SFT},
  author={Thanh Do},
  year={2026},
  url={https://github.com/tdtr0/p1_geom_characterization}
}
```

## License

MIT
