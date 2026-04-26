# Phase 3 Critique and Findings

**Date**: 2025-01-21  
**Status**: Working document for iterative refinement

---

## Executive Summary

Phase 3 experiments revealed **two major positive findings**:
1. **H1 Confirmed**: RL-Zero maintains ~95% subspace preservation while SFT drops to ~45%
2. **Error-direction works**: Linear probe separates correct/incorrect with d=1-2

The geometric measures on raw activations (curvature, velocity) returned null results due to superposition—but this is a methodological insight, not a failure of the core hypothesis.

---

## Part 1: Positive Findings

### 1.1 H1: Subspace Preservation ✅ CONFIRMED

**Result**: Strong confirmation of the primary hypothesis.

![Subspace Preservation Plot](../figures/subspace_preservation.png)

| Model | Early Layers (0-5) | Middle Layers (15-20) | Late Layers (25-30) |
|-------|-------------------|----------------------|---------------------|
| RL-Zero | ~1.0 | ~0.98 | ~0.95 |
| SFT | ~0.85 → 0.60 | ~0.45 | ~0.45 |
| Think | ~0.85 → 0.55 | ~0.42 | ~0.45 |

**Key observations**:
1. **RL-Zero preserves geometry almost perfectly** across all layers
2. **SFT/Think show progressive drift** from base, stabilizing around 45% by layer 15
3. **The divergence happens in middle layers** (5-15), not early or late
4. **Think ≈ SFT** despite additional DPO+RLVR training — the SFT step dominates

**Interpretation**: This confirms Jin et al.'s findings at a more granular level. RLVR (when applied directly to base without SFT) preserves the representational structure. SFT fundamentally reshapes the geometry.

**Implication for transfer**: If transfer depends on preserving base model structure (the "elicitation" hypothesis), this explains why pure RL models might transfer better—they haven't disrupted the base geometry.

### 1.2 Error-Direction Analysis ✅ STRONG SIGNAL

**Result**: Simple linear probe (difference-in-means) separates correct/incorrect with Cohen's d = 1-2.

| Model | GSM8K d | HumanEval d | GSM8K Acc | HE Acc |
|-------|---------|-------------|-----------|--------|
| base | 2.17 | 1.06 | 88.0% | 70.7% |
| sft | 0.79 | 0.69 | 63.3% | 60.0% |
| rl_zero | 1.80 | 0.74 | 84.0% | 60.0% |
| think | 1.83 | 0.55 | 81.3% | 56.0% |

**Cross-domain transfer** (the interesting finding):

| Model | GSM8K→HE | HE→GSM8K | Pattern |
|-------|----------|----------|---------|
| SFT | 66.0% | 56.0% | **Bidirectional** |
| Base | 85.3% | 10.0% | Asymmetric |
| RL-Zero | 41.3% | 18.0% | **None** |
| Think | 22.7% | 44.7% | **None** |

**Interpretation**: 
- SFT learns a generic "task difficulty" feature that transfers
- RLVR learns domain-specific error detection
- This is *opposite* to "RLVR transfers better" narrative for error detection

### 1.3 SVD Insight (Reframed as Positive)
## We have the dât for this thread - we can actually do something here. 
**Result**: Tail eigenvectors change 3-8x more than top eigenvectors in RLVR.

| Task | Top-10 Delta | Tail-50 Delta | Ratio |
|------|-------------|---------------|-------|
| HumanEval | 0.8% | 7.3% | 0.12 |
| GSM8K | 2.2% | 6.7% | 0.33 |

**Reframing**: RLVR **preserves the signal subspace** (top eigenvectors) while **refining the noise subspace** (tail). This is consistent with:
- H1 (subspace preservation)
- "Elicitation not capability expansion" hypothesis
- The idea that RLVR cleans up without disrupting
we need to test this way more than needed. 

---

## Part 2: Null Results and Methodological Critique

### 2.1 Menger Curvature (Layers): ❌ NULL — Architectural

**Result**: r ≈ 1.0 for all comparisons (correct vs incorrect, cross-domain)

**Why it failed**: Layer trajectories capture fixed architecture:
- LayerNorm scaling patterns
- MLP expansion/contraction ratios
- Attention averaging effects

These are constant across inputs. Curvature on layer trajectories is **not a semantic measure**.

### 2.2 Sequence Flow (Velocity/Curvature): ❌ NULL — Architectural

**Result**: d ~ -0.3, r > 0.95

**Why it failed**: Per-token curvature in raw activation space captures:
- Token embedding distances
- Positional encoding effects
- Attention pattern regularity

**The fix**: Project onto semantic subspaces first, then measure geometry.

### 2.3 Lyapunov (Fast Method): ❌ INCONCLUSIVE — Method Too Crude

**Result**: No significant difference (d ~ -0.2 to -0.3)

**Method used**: Frobenius norm ratio as proxy:
```python
λ_approx = log(||x_{l+1}|| / ||x_l||)
```

**This is NOT a proper Lyapunov exponent.** See Section 3 for detailed analysis.

---

## Part 3: Frobenius vs Full Jacobian Analysis

### 3.1 What We Did (Frobenius Norm Ratio)

The "fast" Lyapunov method computes:
```python
λ_frobenius = log(||x_{l+1}||_F / ||x_l||_F)
```

This measures: **Average magnitude change across layers**

### 3.2 What Lyapunov Exponents Actually Measure

True Lyapunov exponents require the **Jacobian** of the layer transformation:
```python
J = ∂f_l(x) / ∂x  # Shape: (d_model, d_model)
```

The maximum Lyapunov exponent is:
```python
λ_max = log(σ_max(J))  # Largest singular value of Jacobian
```

The full Lyapunov spectrum is:
```python
λ_i = log(σ_i(J))  # All singular values
```

### 3.3 What Information Is Lost with Frobenius

| Property | Frobenius | Full Jacobian |
|----------|-----------|---------------|
| **Directional sensitivity** | ❌ Lost | ✅ Captured |
| **Expanding vs contracting directions** | ❌ Averaged out | ✅ Full spectrum |
| **Local sensitivity to perturbations** | ❌ Only global norm | ✅ Per-direction |
| **Chaos detection** | ❌ Cannot detect | ✅ λ_max > 0 = chaos |

**The key loss**: Frobenius averages across all directions. But semantic information lives in *specific directions* (like the error direction). A layer could:
- Preserve overall norm (Frobenius ≈ 0)
- While dramatically amplifying the error direction (λ_error >> 0)
- And suppressing irrelevant directions (λ_noise << 0)

Frobenius would show no signal; full Jacobian would show strong signal.

### 3.4 Why We Expect Different Results with Full Jacobian

**Evidence from other analyses**:

1. **Error-direction finds d=2.0** while **Frobenius finds d=-0.3**
   - This means the difference is *directional*, not magnitude-based
   - The error direction specifically separates correct/incorrect
   - Full Jacobian might show λ_error differs between correct/incorrect

2. **Subspace preservation shows RLVR ≈ 1.0, SFT ≈ 0.45**
   - RLVR preserves directions; SFT disrupts them
   - Full Jacobian would show *which* directions are preserved/disrupted
   - Frobenius can't see this — it's blind to directional structure

3. **SVD shows tail changes more than top**
   - The transformation is not isotropic
   - Different singular value regimes behave differently
   - Full Jacobian would capture this anisotropy

### 3.5 Specific Hypotheses for Full Jacobian

**H_jac1**: Maximum Lyapunov exponent differs between correct/incorrect
- Prediction: Incorrect samples show higher λ_max (more chaotic/sensitive)
- Rationale: Incorrect answers may result from amplified noise

**H_jac2**: Lyapunov exponent in error-direction subspace differs
- Method: Project Jacobian onto error direction, compute λ in that subspace
- Prediction: Correct samples show smaller λ_error (more stable in semantic subspace)

**H_jac3**: RLVR has flatter Lyapunov spectrum than SFT
- Method: Compare full spectrum distribution
- Prediction: RLVR shows more uniform spectrum (preserves all directions equally)
- Prediction: SFT shows peaked spectrum (amplifies some, suppresses others)

### 3.6 Computational Cost

| Method | Cost per sample | Feasibility |
|--------|-----------------|-------------|
| Frobenius | O(d) | Trivial |
| Full Jacobian (exact) | O(d³) | Infeasible for d=4096 |
| Jacobian (finite diff) | O(d²) forward passes | ~16M passes per sample |
| Jacobian (random proj) | O(k²) where k << d | Feasible if k ~ 100 |

**Practical approach**: 
1. Project to k=100 dimensions (random or PCA)
2. Estimate Jacobian in projected space
3. Compute Lyapunov spectrum in reduced space

Or: Compute Jacobian only in semantically meaningful subspaces (error direction ± neighbors).

---

## Part 4: Revised Research Status

### What's Confirmed

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: Subspace Preservation | ✅ **CONFIRMED** | RL-Zero ~0.95, SFT ~0.45 |
| H2: Error-direction exists | ✅ **CONFIRMED** | d = 1-2 within-domain |
| Transfer is model-dependent | ✅ **NEW FINDING** | SFT transfers, RLVR doesn't |

### What's Null (Methodological)

| Analysis | Result | Reason |
|----------|--------|--------|
| Menger curvature (layers) | r ≈ 1.0 | Captures architecture, not semantics |
| Sequence flow (raw) | d ~ -0.3 | Same — raw geometry is architectural |
| Lyapunov (Frobenius) | d ~ -0.3 | Method too crude, loses directional info |

### What's Untested

| Analysis | Why Pending |
|----------|-------------|
| Full Jacobian Lyapunov | Computationally expensive |
| Projected trajectory dynamics | Need to implement |
| Cross-model-family validation | Only OLMo tested |

---

## Part 5: Next Steps

### Priority 1: Validate Subspace Preservation Finding
- [ ] Reproduce on DeepSeek family
- [ ] Test on Llama/Qwen families
- [ ] Correlate preservation score with transfer performance

### Priority 2: Projected Trajectory Dynamics
- [ ] Compute trajectories in error-direction subspace
- [ ] Measure velocity/curvature in semantic space (not raw)
- [ ] Test if trajectory smoothness correlates with correctness

### Priority 3: Full Jacobian Analysis (if resources permit)
- [ ] Implement random projection approach
- [ ] Compute Lyapunov spectrum in reduced space
- [ ] Test H_jac1, H_jac2, H_jac3

---

## Appendix: Summary of Key Findings

| Finding | Implication |
|---------|-------------|
| RL-Zero preserves ~95% subspace structure | RLVR elicits, doesn't reshape |
| SFT drops to ~45% preservation | SFT fundamentally changes geometry |
| Think ≈ SFT despite extra training | SFT step dominates subsequent RL |
| Error-direction has d=1-2 | Models encode correctness linearly |
| SFT error-direction transfers | Generic "difficulty" feature |
| RLVR error-direction doesn't transfer | Domain-specific error detection |
| Tail eigenvectors change more than top | RLVR refines noise, preserves signal |

---

*This document is a working draft for iterative refinement.*
