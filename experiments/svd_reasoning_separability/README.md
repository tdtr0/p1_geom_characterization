# SVD Reasoning Separability Experiment

**Question**: Does RLVR training change specific eigenvectors (separable reasoning) or all eigenvectors equally (entangled)?

## Hypothesis

If reasoning capabilities are **separable** from factual knowledge:
- Top eigenvectors should show HIGH delta (reasoning directions refined by RLVR)
- Tail eigenvectors should show LOW delta (knowledge unchanged)

If reasoning is **entangled** with knowledge:
- All eigenvectors change roughly equally
- Or: changes don't correlate with eigenvalue rank

## Method

1. Load activations from `olmo3_base` and `olmo3_rl_zero` (same prompts)
2. For each layer, compute SVD on (n_samples × seq_len, d_model) matrix
3. Compare eigenvectors using cosine similarity: `delta_k = 1 - |cos(U_base[:,k], U_rlvr[:,k])|`
4. Plot delta vs eigenvalue rank

## Data

Uses Phase 2 trajectory data from `/data/thanhdo/trajectories_0shot/`:
- `olmo3_base/gsm8k_trajectories.h5` (3.9GB, 500 samples)
- `olmo3_rl_zero/gsm8k_trajectories.h5` (3.9GB, 500 samples)

## Usage

```bash
# On eyecog
cd ~/p1_geom_characterization
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
python experiments/svd_reasoning_separability/analyze_svd_delta.py
```

## Results

See `results/` folder after running.
