# SVD Linear Separability Analysis: Motivating Negative Result

**Date**: 2026-01-19
**Experiment**: `experiments/svd_reasoning_separability/`

## Research Question

Does RLVR training create separable "reasoning subspaces" in the top eigenvectors of activation space?

**Prediction if reasoning is separable**:
- Top eigenvectors should show HIGH delta (reasoning directions refined by RLVR)
- Tail eigenvectors should show LOW delta (knowledge unchanged)
- Ratio top/tail > 1.5

**Prediction if entangled**:
- All eigenvectors change roughly equally
- Or: changes don't correlate with eigenvalue rank

## Method

1. Loaded trajectories from `olmo3_base` and `olmo3_rl_zero` (same prompts, seed=42)
2. For each of 16 layers, computed SVD on (n_samples × seq_len, d_model) matrix
3. Compared eigenvectors using cosine similarity: `delta_k = 1 - |cos(v_base_k, v_rlvr_k)|`
4. Analyzed top-10 vs tail-50 eigenvectors

**Data**: 50 samples × 256 seq_len from HumanEval and GSM8K (0-shot)

## Results

| Task | Top-10 Delta | Tail-50 Delta | Ratio (top/tail) | Interpretation |
|------|-------------|---------------|------------------|----------------|
| **HumanEval** | 0.0084 (0.8%) | 0.0725 (7.3%) | **0.12** | TAIL-HEAVY |
| **GSM8K** | 0.0221 (2.2%) | 0.0666 (6.7%) | **0.33** | TAIL-HEAVY |

### Layer-by-Layer Pattern (HumanEval)

| Layer | top10 | tail50 | Interpretation |
|-------|-------|--------|----------------|
| 0-4 | ~0.000 | ~0.001-0.004 | Near-zero change in early layers |
| 5-7 | ~0.001-0.013 | ~0.022-0.077 | Tail starts diverging |
| 8-11 | ~0.001-0.003 | ~0.074-0.117 | Large tail changes |
| 12-15 | ~0.006-0.087 | ~0.125-0.178 | Both increase, but tail dominates |

**Outlier**: Layer 12 showed top10=0.087 (highest), suggesting something special at that depth.

## Key Finding: OPPOSITE of Separable Reasoning Hypothesis

**Result**: Tail eigenvectors change 3-8x MORE than top eigenvectors.

This is the **opposite** of what "separable reasoning" would predict. Instead:

1. **Top eigenvectors (dominant variance) are preserved** - the core representational structure remains stable
2. **Tail eigenvectors (low variance) are refined** - RLVR adjusts fine-grained directions
3. **Changes are layer-dependent** - later layers show more modification

## Interpretation

### What This Means for the Research

**Linear methods don't capture reasoning differences.**

The SVD result supports the **interpolation view** (Allen-Zhu & Li, 2024):
- There is no separate "reasoning subspace" created by RLVR
- The main representational structure (top eigenvectors) is preserved
- RLVR refinement happens in a distributed manner across low-variance directions

### Why This Motivates Nonlinear/Dynamical Analysis

If reasoning isn't in separable linear directions, where is it?

**Hypothesis**: Reasoning is in the **flow** (trajectory dynamics), not the **space** (static subspace).

This motivates Phase 3's dynamical analysis:
- **Vector field decomposition**: How do correct vs incorrect solutions flow through the manifold?
- **Lyapunov exponents**: Are correct solutions more stable?
- **Attractor analysis**: Do they converge to different basins?
- **Path signatures**: Do they have different trajectory shapes?

## Connection to Prior Work

### Consistent With:
- **Allen-Zhu & Li (2024)**: Everything is interpolation; no "reasoning mode" vs "recall mode"
- **Merullo et al. (2025)**: Math uses memorization-like (low-curvature) circuits alongside general computation

### Sets Up:
- **Ren & Liu (2026)**: Attractor dynamics - correct solutions find right attractors
- **MARBLE analysis**: Vector field decomposition to characterize flow

## Figures

See `experiments/svd_reasoning_separability/results/`:
- `delta_vs_rank.png` - Delta vs eigenvalue rank (shows upward trend)
- `delta_heatmap.png` - Layer × eigenvector heatmap
- `top_vs_tail.png` - Bar chart comparing top-10 vs tail-50

## Conclusion

**Linear separability of reasoning: NOT SUPPORTED**

RLVR does not create distinct "reasoning eigenvectors." The main representational structure is preserved; changes are distributed in low-variance directions.

**Implication**: To find signatures of correct reasoning, we need to analyze **trajectory dynamics** (flow, stability, attractors), not static subspace structure.

This negative result motivates the dynamical systems approach in Phase 3.

---

## Appendix: Raw Results

```
HUMANEVAL:
  Mean delta (top-10 eigenvectors): 0.0084
  Mean delta (tail-50 eigenvectors): 0.0725
  Ratio (top/tail): 0.12
  --> SUGGESTS TAIL-HEAVY (tail changes more - unexpected)

GSM8K:
  Mean delta (top-10 eigenvectors): 0.0221
  Mean delta (tail-50 eigenvectors): 0.0666
  Ratio (top/tail): 0.33
  --> SUGGESTS TAIL-HEAVY (tail changes more - unexpected)

Total time: 733.4s (12.2m)
```

Full results JSON: `experiments/svd_reasoning_separability/results/svd_analysis_20260119_173629.json`
