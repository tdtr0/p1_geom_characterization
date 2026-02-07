# Dynamical Analysis Findings

**Date**: 2026-02-06
**Objective**: Characterize what distinguishes correct from incorrect solutions geometrically.

---

## Executive Summary

We systematically tested dynamical/geometric measures to distinguish correct from incorrect solutions. The key finding is that **static geometry works, dynamics don't** — with one important exception: **early token positions carry correctness signal**.

---

## 1. What Works

### Linear Probe on Mean Activations
- **Method**: Logistic regression on mean-pooled final layer activations
- **Result**: AUC 0.68-0.75
- **Status**: ✅ WORKS
- **Implication**: Correct vs incorrect occupy separable regions of representation space

### Per-Sample CKA (Token Similarity Preservation)
- **Method**: CKA between adjacent layers, per sample
- **Result**: d = -0.64 at L7 (p=0.029)
- **Status**: ✅ WORKS (but low resolution)
- **Finding**: Incorrect solutions have HIGHER CKA (preserve more token similarity)
- **Interpretation**: Correct solutions "work harder" — more token relationship restructuring

### Early Token Signal
- **Method**: Per-token activation norm, correct vs incorrect
- **Result**: Early tokens (0-33%) show d = -0.15, specific tokens up to d = -1.16
- **Status**: ✅ WORKS
- **Key tokens**: T11 (d=-1.16), T43 (d=-0.90), T39 (d=-0.71)
- **Interpretation**: Problem encoding tokens carry correctness signal

---

## 2. What Doesn't Work (NULL Results)

### Path-Level Dynamics (Architectural)

All these metrics are **identical for correct and incorrect** — they're architectural properties, not computation-dependent:

| Metric | Cohen's d | p-value | Status |
|--------|-----------|---------|--------|
| Step magnitude per layer | -0.14 | 0.63 | NULL |
| Jump count | 0.00 | NaN | NULL |
| Acceleration | -0.02 | 0.94 | NULL |
| Max jerk | 0.15 | 0.60 | NULL |
| Dimensionality trajectory | -0.07 | 0.81 | NULL |

**Script**: `layer_path_dynamics.py`

### Sensitivity-Based Measures (Orthogonality Bottleneck)

Token-level cos(X_l, X_{l+1}) ≈ 0.10 — consecutive layers are nearly orthogonal. This causes:

| Measure | Status | Why It Fails |
|---------|--------|--------------|
| Jacobian/Lyapunov | NULL | SVs ≈ 1 (rotation, not expansion) |
| Procrustes/SVCCA | NULL | Same transformation quality |
| Directional expansion | NULL | No sensitivity signal |

**Script**: `empirical_jacobian_lyapunov.py`, `alignment_aware_dynamics.py`

### Menger Curvature Profile (Architectural)

Curvature profiles are r ≈ 0.999 correlated across correct/incorrect and across tasks. This is an **architectural property of transformers**, not a reasoning signature.

---

## 3. Cross-Model Paired Analysis

### Base vs SFT (Same Question Comparison)

**Setup**: Same prompt (seed=42) processed by both models, compare layer-by-layer

| Layer | Cos Sim (base↔sft) | Distance |
|-------|-------------------|----------|
| L0 | 0.993 | 0.03 |
| L4-5 | 0.95 | 0.21-0.27 |
| L7 | 0.91 | 0.47 |
| L15 | 0.77 | 2.75 |

**Key findings**:
- Models diverge **gradually** starting at L4-5
- No sharp "phase transition" layer
- By L15, cos_sim drops to 0.77

### Discordant Pairs (Models Disagree on Correctness)

- 54/100 questions: models disagree
- 50 cases: SFT correct, base wrong
- 4 cases: base correct, SFT wrong
- **No special divergence pattern** for discordant vs concordant pairs

**Script**: `paired_trajectory_divergence.py`

### Base vs RL-Zero (Same Question Comparison)

**Setup**: Same prompt (seed=42) processed by both models, compare layer-by-layer

| Layer | Cos Sim (base↔rl_zero) | Distance |
|-------|------------------------|----------|
| L0 | 1.000 | 0.00 |
| L5 | 0.999 | 0.03 |
| L10 | 0.999 | 0.07 |
| L15 | **0.995** | 0.46 |

**Key findings**:
- RL-Zero stays **dramatically closer to base** than SFT
- Cos_sim **never drops below 0.99** (vs 0.77 for SFT)
- Models effectively share the same representation path

### Discordant Pairs: Base vs RL-Zero

- 11/100 questions: models disagree (vs 54 for SFT)
- 9 cases: RL-Zero correct, base wrong
- 2 cases: base correct, RL-Zero wrong
- **Agreement rate: 89%** (vs 46% for SFT)

**Critical finding**: When models DO disagree, divergence concentrates at L7-L9:

| Layer | Discordant Cos | Cohen's d | p-value |
|-------|----------------|-----------|---------|
| L7 | 0.9994 | -0.816 | 0.012* |
| L8 | 0.9991 | -0.759 | 0.020* |
| L9 | 0.9989 | -0.725 | 0.026* |

**Interpretation**: RL-Zero learns a *subtle refinement* to base, not a wholesale change. The "reasoning improvement" happens in a tiny subspace that barely affects overall trajectory similarity.

---

## 4. Token-Position Analysis

### Where Does Correctness Signal Live?

| Position Range | Cohen's d | Interpretation |
|----------------|-----------|----------------|
| **Early (0-33%)** | -0.152 | Signal here |
| Middle (33-66%) | -0.003 | No signal |
| Late (66-100%) | 0.000 | No signal |

### Specific High-Signal Tokens

| Token | Cohen's d | Position % |
|-------|-----------|------------|
| T11 | -1.16 | 2.1% |
| T43 | -0.90 | 8.4% |
| T39 | -0.71 | 7.6% |
| T33 | -0.61 | 6.4% |

**Interpretation**: Problem setup tokens (first ~50 tokens) carry the correctness signal. This explains why linear probes work — they capture how the model **encodes the problem**, not how it **reasons about it**.

---

## 5. Neuron Activation Patterns

### Sparsity Analysis (Null - Wrong Metric)

Initial sparsity analysis showed constant 0.1001 because we measured top-10% which is definitionally 10%. Need to measure:
- Activation magnitude variance
- Gini coefficient of activation distribution
- Effective rank per layer

**Status**: Needs revision

---

## 6. Key Interpretations

### Why Static Geometry Works, Dynamics Don't

1. **The answer is encoded early**: Correct vs incorrect is determined by how the problem is encoded in early tokens, not by the reasoning process
2. **Dynamics are architectural**: Step lengths, jumps, acceleration are transformer properties, not computation properties
3. **Orthogonality bottleneck**: Token-level layer transitions are rotations (cos ≈ 0.1), nullifying sensitivity-based measures

### What "Working Harder" Actually Means

CKA analysis showed correct solutions have lower CKA (more token relationship restructuring). But step magnitude analysis showed no difference. Resolution:

- **CKA measures token-token similarity structure** (Gram matrix)
- **Step magnitude measures overall displacement** (Frobenius norm)

Correct solutions don't take longer paths — they **reorganize token relationships differently** while traversing the same path length.

---

## 7. Cross-Validation with Prior Findings

| Prior Finding | Current Analysis | Alignment |
|---------------|-----------------|-----------|
| CKA peak at L7 | Step profile NULL | CKA captures different aspect |
| Error direction L24-28 | Divergence gradual | No sharp layer |
| Wynroe L16-18 spike | Not replicated | Architecture-dependent |
| Belief tracking | Early tokens | Consistent (problem encoding) |

---

## 8. Scripts Created

| Script | Purpose | Key Result |
|--------|---------|------------|
| `cka_deep_analysis.py` | Per-sample CKA | d=-0.64 at L7 |
| `layer_path_dynamics.py` | Path metrics | All NULL |
| `paired_trajectory_divergence.py` | Cross-model divergence | Gradual divergence, early token signal |

---

## Per-Token Cross-Model Divergence (Base↔RL-Zero)

### Token-Level Analysis at L15

Only 175/512 tokens are non-padding (short sequences).

| Token | Cos Sim | Distance | Position | Type |
|-------|---------|----------|----------|------|
| T10 | 0.9951 | **8.30** | 5.7% | Scale change |
| T7 | 0.9907 | **8.09** | 4.0% | Scale change |
| T5 | 0.9907 | **8.00** | 2.9% | Scale change |
| T174 | **0.9829** | 0.15 | 99.4% | Direction change |
| T173 | **0.9834** | 0.11 | 98.9% | Direction change |
| T112 | **0.9878** | 0.21 | 64.0% | Direction change |

### Two Types of Divergence

**1. Early tokens (T5-T10): Scale adjustment**
- High distance (7-8), same direction (cos_sim 0.99+)
- RL-Zero amplifies/dampens problem encoding
- Preserves direction, changes magnitude

**2. Middle/Late tokens: Directional refinement**
- Low distance (0.1-0.2), slightly lower cos_sim (0.98)
- RL-Zero makes subtle directional changes
- These are the "reasoning" tokens

### Position Quartile Analysis

| Quartile | Mean Cos Sim | Interpretation |
|----------|--------------|----------------|
| Q1 (0-25%) | 0.9951 | Early tokens, scale changes |
| Q2 (25-50%) | 0.9932 | Lowest — reasoning divergence |
| Q3 (50-75%) | 0.9927 | Lowest — reasoning divergence |
| Q4 (75-100%) | 0.9946 | Answer tokens, converge back |

**Key insight**: RL-Zero diverges most at Q2-Q3 (middle tokens = reasoning), then converges at Q4 (answer).

---

## 9. Next Steps

### Completed
- [x] **Base vs RL-Zero paired analysis** — ✅ RL-Zero stays MUCH closer to base (cos 0.995 vs 0.77)
- [x] **Per-token divergence** — ✅ Early tokens: scale; Middle tokens: direction
- [ ] Fix sparsity metric (Gini coefficient or magnitude variance)

### Comparison: Base→SFT vs Base→RL-Zero

| Metric | Base→SFT | Base→RL-Zero |
|--------|----------|--------------|
| Final layer cos_sim | 0.77 | **0.995** |
| Divergence onset (cos < 0.95) | L4-5 | **Never** |
| Discordant pairs | 54/100 | **11/100** |
| Agreement rate | 46% | **89%** |
| Early token signal (d) | -0.152 | -0.152 |
| Peak token | T11 (d=-1.16) | T11 (d=-1.16) |

### Open Questions (Answered)
1. ✅ **Does RL-Zero show different divergence pattern from SFT?**
   - **YES, dramatically different.** RL-Zero barely diverges from base (cos_sim > 0.99 throughout).
   - SFT makes large representation changes; RL-Zero makes subtle refinements.
   - This confirms previous finding: RL-Zero has 98.6% subspace preservation vs base.

2. Which specific neurons/dimensions carry the early-token signal?
3. Can we use early-token signal for real-time correctness prediction?

### Theoretical Implications

The **SFT vs RL-Zero divergence contrast** reveals:
- **SFT**: Learns new representations (large trajectory deviation)
- **RL-Zero**: Refines existing representations (minimal trajectory deviation)

Yet RL-Zero still improves correctness (17 vs 10 correct). This means:
- The "reasoning improvement" lives in a **tiny subspace**
- It's detectable at L7-L9 in discordant pairs (d ≈ -0.8)
- But it doesn't change overall trajectory geometry

---

## 10. Data References

- **Trajectories**: `data/trajectories_0shot/{model}/{task}_trajectories.h5`
- **Shape**: (n_samples, 512 tokens, 16 layers, 4096 dims)
- **Models**: olmo3_base, olmo3_sft, olmo3_rl_zero, olmo3_think
- **Tasks**: gsm8k, humaneval, logiqa
