# Phase 3 Analysis Summary

**Date**: 2026-01-23 (updated)
**Status**: Primary analyses complete, Full Lyapunov complete with positive result

---

## Data Available

### Trajectory Data (`/data/thanhdo/trajectories_0shot/`)

| Model | GSM8K | HumanEval | LogiQA |
|-------|-------|-----------|--------|
| olmo3_base | ✅ 500 (13% correct) | ✅ 500 (4% correct) | ✅ 500 (25% correct) |
| olmo3_sft | ✅ 500 (59% correct) | ✅ 500 (5% correct) | ❌ TRUNCATED |
| olmo3_rl_zero | ✅ 500 (14% correct) | ✅ 500 (13% correct) | ❌ TRUNCATED |
| olmo3_think | ✅ 500 (39% correct) | ✅ 500 (5% correct) | ❌ TRUNCATED |

**Trajectory shape**: `(n_samples, 512, 16, 4096)` — 512 tokens, 16 layers (even 0-30), 4096 dims

**Labels**: `is_correct` field in each HDF5 file

---

## Analyses Completed

### 1. Error-Detection Direction (Linear Probing)

> **Note**: This is NOT the Wynroe et al. methodology. Wynroe used **causal activation patching**; this is **correlational probing** (difference-in-means). See Section 6 for actual Wynroe replication.

**Script**: `scripts/analysis/phase3_dynamical_analysis.py`

**Method**:
1. Compute difference-in-means: `d = mean(correct) - mean(incorrect)` at each layer
2. Normalize direction
3. Project all samples onto direction
4. Classify using median threshold
5. Test cross-domain transfer

**Results**:

| Model | GSM8K Within | HumanEval Within | GSM8K→HE | HE→GSM8K |
|-------|--------------|------------------|----------|----------|
| base | 88.0% (d=2.17) | 70.7% (d=1.06) | **85.3%** | 10.0% |
| sft | 63.3% (d=0.79) | 60.0% (d=0.69) | **66.0%** | **56.0%** |
| rl_zero | 84.0% (d=1.80) | 60.0% (d=0.74) | 41.3% | 18.0% |
| think | 81.3% (d=1.83) | 56.0% (d=0.55) | 22.7% | 44.7% |

**Key findings**:
- ✅ Strong within-domain separation (d=0.5-2.0)
- ✅ SFT shows bidirectional transfer
- ❌ RL-Zero and Think show no transfer
- ⚠️ Base shows asymmetric transfer (math→code only)

This should be expected from the first order results - confirming the hypothesis that first order shouldnt transfer at all - at least not the direction/trjaectory
---

### 2. Menger Curvature on Layer Trajectories

**Script**: `scripts/analysis/phase3_dynamical_analysis.py`

**Method**:
1. Average trajectory across sequence: `(512, 16, 4096) → (16, 4096)`
2. Compute Menger curvature at each layer triplet: `κ = 4A / (|a||b||c|)`
3. Compare curvature profiles: correct vs incorrect

**Results**:

| Comparison | Correlation (r) |
|------------|-----------------|
| Correct ↔ Incorrect (same domain) | **0.9999** |
| Cross-domain (HumanEval ↔ LogiQA) | **0.996** |

**Key finding**: ❌ **NULL RESULT** — Curvature profile is purely architectural (r≈1.0 for all comparisons). Does not distinguish correct/incorrect or domains.

---

### 3. SVD Linear Separability

**Script**: `experiments/svd_reasoning_separability/analyze_svd_delta.py`

**Method**:
1. Compute SVD on activations: `(n_samples × seq_len, d_model)`
2. Compare eigenvectors between base and rl_zero: `delta = 1 - |cos(v_base, v_rlvr)|`
3. Compare top-10 vs tail-50 eigenvector changes

**Results**:

| Task | Top-10 Delta | Tail-50 Delta | Ratio |
|------|-------------|---------------|-------|
| HumanEval | 0.8% | 7.3% | **0.12** |
| GSM8K | 2.2% | 6.7% | **0.33** |

**Key finding**: ❌ **OPPOSITE of prediction** — Tail eigenvectors change 3-8x MORE than top. RLVR preserves top eigenvectors, refines tail. Linear separability not supported.

---

### 4. Sequence Flow at Last Layer (H_flow1, H_flow2, H_flow3)

**Script**: `scripts/analysis/sequence_flow_analysis.py`

**Method**:
1. Extract last layer activations: `(n_samples, 512, 4096)`
2. Compute velocity: `v_t = ||x_{t+1} - x_t||`
3. Compute Menger curvature on token sequence: `κ_t = Menger(x_{t-1}, x_t, x_{t+1})`
4. Compare correct vs incorrect distributions
5. Test cross-domain transfer using flow features

**Results**:

**H_flow1 (Velocity)**:
| Task | Correct | Incorrect | Cohen's d | p |
|------|---------|-----------|-----------|---|
| HumanEval | 15.8 ± 5.7 | 18.5 ± 8.6 | -0.32 | 0.29 |
| LogiQA | 28.0 ± 5.1 | 30.2 ± 7.8 | -0.30 | 0.23 |

❌ No significant difference

**H_flow2 (Sequence Curvature Profile)**:
| Task | Profile Correlation (r) |
|------|------------------------|
| HumanEval | 0.9769 |
| LogiQA | 0.9903 |

❌ Still architectural (r > 0.95)

**H_flow3 (Flow Feature Transfer)**:
| Metric | HumanEval | LogiQA | HE→LQ | LQ→HE |
|--------|-----------|--------|-------|-------|
| Within AUC | 0.560 | 0.662 | — | — |
| Cross AUC | — | — | 0.629 | 0.541 |

⚠️ Weak signal, opposite direction from error-direction transfer

---

### 5. Lyapunov Exponents (Fast Method)

**Script**: `scripts/analysis/phase3_dynamical_analysis.py`

**Method**:
1. Compute Frobenius norm ratio as proxy: `λ = log(||x_{l+1}|| / ||x_l||)`
2. Mean across layers

**Results**:
| Task | Correct | Incorrect | Cohen's d |
|------|---------|-----------|-----------|
| HumanEval | ~0.01 | ~0.01 | -0.3 |
| LogiQA | ~0.01 | ~0.01 | -0.2 |

❌ No significant difference (but fast method is crude approximation)

---

### 6. Wynroe Activation Patching (Causal) ⚠️ COMPLETE

**Script**: `experiments/aha_moment/replicate_wynroe_patching.py`

**Method** (matches Wynroe et al.):
1. Generate MATH solutions with think model
2. Inject arithmetic errors to create corrupted traces
3. For each layer: patch corrupted activations with clean
4. Measure logit-diff recovery for correction tokens ("Wait", "But", "Actually")
5. Filter: logit-diff > 3 (kept 47%, Wynroe kept 44%)

**Results** (think model, N=50 pairs, MATH/algebra):

| Layer | Recovery % |
|-------|------------|
| 0 | 100.0 ± 0.0 |
| 10 | 100.0 ± 0.0 |
| 20 | 100.0 ± 0.0 |
| 30 | 100.0 ± 0.0 |

**Comparison to Wynroe**:
| Aspect | Wynroe (DeepSeek-R1) | Us (OLMo-3-Think) |
|--------|----------------------|-------------------|
| Finding | Layer 20 spike (~70%) | **100% at ALL layers** |
| Interpretation | Localized circuit | **Distributed processing** |

**Key finding**: ⚠️ **DIFFERENT FROM WYNROE** — OLMo-3 has distributed error processing. Any single layer can fully recover clean behavior. No critical layer (no spike pattern). Only tested on think model; base/sft/rl_zero untested.

---

### 7. Pivot Velocity → Correction Success ✅ NULL RESULT

**Script**: `experiments/aha_moment/analyze_pivot_outcome_geometry.py`

**Method**:
1. Use 200 GSM8K samples from think model with pivot trajectories
2. Detect pivots ("Wait", "But") and measure velocity
3. Determine correctness (final answer matches ground truth)
4. Test if velocity predicts success

**Results**:

| Analysis | Finding | p-value |
|----------|---------|---------|
| Velocity → Correctness | **No relationship** | p=0.55 |
| Low vs High velocity | 66.7% vs 71.4% correct | p=0.93 |

**Self-Correction Analysis** (samples where model changed a number after pivot):
- Samples WITH self-corrections: **52.9% correct** (9/17)
- Samples WITHOUT: **71.0% correct** (130/183)

**Key finding**: ❌ **NULL RESULT** — Pivot velocity does NOT predict correction success. Self-corrections indicate trouble (harder problems). Pivot words are stylistic, not functional.

---

### 8. Full Lyapunov Spectrum Analysis ✅ POSITIVE RESULT

**Script**: `scripts/analysis/full_lyapunov_analysis.py`

**Date**: 2026-01-23

**Method**:
1. Compute SVD-based Lyapunov at each layer transition (not Frobenius norm)
2. Extract error-detection direction using difference-in-means
3. Compute directional Lyapunov: expansion rate in error-direction subspace
4. Test three hypotheses:
   - **H_jac1**: Max Lyapunov differs (incorrect = more chaotic?)
   - **H_jac2**: Directional Lyapunov in error-direction subspace differs
   - **H_jac3**: Spectrum width (anisotropy) differs

**Results** (n=50 samples per task, k=50 SVD components):

| Model/Task | H_jac1 (d, p) | **H_jac2 (d, p)** | H_jac3 (d, p) |
|------------|---------------|-------------------|---------------|
| olmo3_base/gsm8k | -0.12, 0.80 | **+1.68, 0.001** ✓ | +0.41, 0.38 |
| olmo3_base/humaneval | -0.16, 0.74 | **+1.42, 0.004** ✓ | -0.19, 0.69 |
| olmo3_base/logiqa | +0.61, 0.07 | **+1.41, 0.000** ✓ | -0.66, 0.05 |
| olmo3_sft/gsm8k | +0.10, 0.74 | **-1.12, 0.000** ✓ | +0.55, 0.06 |
| olmo3_sft/humaneval | -0.68, 0.13 | +0.81, 0.07 | +0.88, 0.05 |
| olmo3_rl_zero/gsm8k | -0.67, 0.06 | **+1.63, 0.000** ✓ | +0.06, 0.86 |
| olmo3_rl_zero/humaneval | -0.46, 0.12 | +0.39, 0.18 | +0.52, 0.07 |
| olmo3_think/gsm8k | -0.19, 0.52 | **-1.39, 0.000** ✓ | +0.15, 0.59 |
| olmo3_think/humaneval | -0.49, 0.23 | **+1.37, 0.002** ✓ | +0.74, 0.08 |

**Significance Count**: 8/27 (30%)
- H_jac1: 0/9 significant
- **H_jac2: 7/9 significant** (78%)
- H_jac3: 1/9 significant

---

### ⚠️ CRITICAL UPDATE: Data Leakage Detected (2026-01-23)

**The above effect sizes are INFLATED by ~10x due to circular analysis.**

The error direction is computed from the SAME data it's tested on:
```python
# Direction computed from ALL samples
error_dir = mean(incorrect) - mean(correct)
# Then tested on SAME samples → CIRCULAR
```

**Cross-Validation Results** (proper test):

| Test | Cohen's d | p-value | Notes |
|------|-----------|---------|-------|
| Original (same-data) | -1.52 | <0.001 | ❌ **LEAKY — inflated** |
| **Cross-validated** | **-0.11** ± 0.44 | 0.50 | Signal disappears |
| Cross-model (base→rl) | **+0.53** | 0.005 | ✅ REAL TRANSFER |
| Cross-task (HE→GSM8K) | **-0.37** | 0.048 | ✅ REAL TRANSFER |

**TRUE effect sizes**: d ~ 0.4-0.5 (medium), NOT d ~ 1.5 (large)

**What remains valid**:
- Cross-model transfer IS significant (d=0.53)
- Cross-task transfer IS significant (d=0.37)
- The error direction DOES capture a semantic signal

**What needs correction**:
- All within-dataset H_jac2 results in the table above are inflated
- The "7/9 significant" claim needs recomputation with proper CV

---

**Key findings** (CORRECTED):

1. ⚠️ **H_jac2 signal is REAL but WEAKER** — Within-dataset effect sizes are inflated by ~10x. True effect is d ~ 0.4-0.5, not d ~ 1.5

2. ✅ **Cross-model transfer works** — Direction from base model distinguishes correct/incorrect on rl_zero data (d=0.53, p=0.005)

3. ✅ **Cross-task transfer works** — Direction from HumanEval works on GSM8K (d=0.37, p=0.048)

4. ⚠️ **Sign reversal across transfer types** — Different directions capture different aspects of "error"

5. ❌ **H_jac1 and H_jac3 show no signal** — Only directional analysis works

**Interpretation** (REVISED):

The directional Lyapunov captures a REAL but MODEST semantic signal:
- Cross-model/cross-task transfer proves it's not purely task-specific
- Effect is medium (d ~ 0.4-0.5), not large (d ~ 1.5)
- The "universal reasoning signature" hypothesis is NOT supported — different sources give different directions

---

### Baseline Comparison: Linear Probe vs H_jac2 (2026-01-24)

**Question**: Does the "dynamical" H_jac2 measure add value beyond static geometry?

**Method**: Compare AUC for predicting correct/incorrect:
1. **Linear probe**: Logistic regression on mean activation at last layer (static)
2. **H_jac2**: Variance expansion in error-direction subspace (dynamical)

**Results** (5-fold CV on HumanEval):

| Model | Linear Probe AUC | H_jac2 AUC | Winner |
|-------|------------------|------------|--------|
| base | **0.679** ± 0.14 | 0.413 ± 0.09 | Probe (+0.27) |
| rl_zero | **0.752** ± 0.03 | 0.303 ± 0.05 | Probe (+0.45) |

**CRITICAL**: H_jac2 AUC < 0.5 means it's **WORSE than random chance**!

**Conclusion**:
- Static geometry (linear probe) captures correctness well (AUC 0.68-0.75)
- The "dynamical" H_jac2 measure adds **NO VALUE**
- The information is already in the static representation
- **Variance expansion does NOT capture reasoning dynamics beyond what a simple linear probe already gets**

---

**Comparison to Fast Lyapunov (Section 5)**:

| Method | Signal? | Why |
|--------|---------|-----|
| Fast (Frobenius) | ❌ NO | Averages over all directions, loses signal |
| Full (Directional) | ✅ YES | Isolates error-direction subspace |

This validates the critique that Frobenius norm is too crude — **directional information matters**.

---

## Summary of Findings

| Analysis | Signal? | Notes |
|----------|---------|-------|
| Error-Direction (linear probing) | ✅ YES | Strong within-domain, model-dependent transfer |
| Menger Curvature (layers) | ❌ NO | Purely architectural (r≈1.0) |
| SVD Linear Separability | ❌ NO | Tail changes MORE than top (opposite of prediction) |
| Sequence Flow Velocity | ❌ NO | d ~ -0.3, not significant |
| Sequence Flow Curvature | ❌ NO | Still architectural (r > 0.95) |
| Flow Feature Transfer | ⚠️ WEAK | AUC 0.54-0.66, opposite direction |
| Lyapunov (fast/Frobenius) | ❌ NO | Not significant (too crude) |
| **Lyapunov (full/directional)** | ❌ **NO VALUE** | Linear probe AUC=0.75 beats H_jac2 AUC=0.30 |
| **Wynroe Patching (causal)** | ⚠️ DIFFERENT | 100% recovery at all layers (distributed, not localized) |
| **Pivot Velocity → Success** | ❌ NO | p=0.55, pivot words are stylistic not functional |

**Wynroe Replication Note** — Three distinct patterns emerging across models:
1. **Mid-layer spike (DeepSeek-Distill)**: Sharp activation at L16-18 → localized "imitation circuit"
2. **Gradual ramp (OLMo RL-Zero)**: Distributed processing, no shortcuts
3. **Final spike (OLMo Think)**: Late-stage decision, possibly from RLVR "unlearning" the early circuit

---

## Key Takeaways

1. **Error-detection direction works** — BUT within-dataset effect sizes (d=1-2) are INFLATED by data leakage. True cross-validated effect is d ~ 0.1-0.4

2. **Cross-model/cross-task transfer IS real** (d=0.4-0.5):
   - Direction from base model works on rl_zero (d=0.53, p=0.005)
   - Direction from HumanEval works on GSM8K (d=0.37, p=0.048)
   - This proves the signal is NOT purely noise

3. **Geometric measures on raw activations fail** — Curvature/velocity capture architecture, not semantics (r>0.95)

4. **H_jac2 adds NO value over linear probe** — Linear probe AUC=0.68-0.75 beats H_jac2 AUC=0.30-0.41. The "dynamical" measure is actually WORSE than static geometry.

5. **"Universal reasoning signature" NOT supported**:
   - Different sources (models, tasks) give different error directions
   - Sign reversal across transfer types (cross-model: d > 0, cross-task: d < 0)
   - Static geometry (linear probe) captures the signal better than dynamics

6. **Critical methodological lessons**:
   - Always use cross-validation when computing directions (inflates effect sizes ~10x otherwise)
   - Always compare to baseline (linear probe) before claiming dynamical measures add value

7. **Linear methods insufficient for EXPLAINING, but sufficient for PREDICTING** — SVD analysis shows RLVR changes are distributed, but a simple linear probe predicts correctness well

---

## Analyses NOT Done (Secondary)

| Analysis | Reason Pending |
|----------|----------------|
| Attractor Analysis | Not yet implemented |
| ~~Full Lyapunov Spectrum~~ | ✅ **DONE** — See Section 8 |
| Vector Field Decomposition | Not yet implemented |
| Baseline Comparisons | Not yet done |
| Difficulty Stratification | Not yet done |
| SV Regime Projection | Skip (SVD analysis suggests null) |

---

## Code References

| Analysis | Script |
|----------|--------|
| Error-Direction (probing) | `scripts/analysis/phase3_dynamical_analysis.py` |
| Menger Curvature | `scripts/analysis/phase3_dynamical_analysis.py` |
| SVD Separability | `experiments/svd_reasoning_separability/analyze_svd_delta.py` |
| Sequence Flow | `scripts/analysis/sequence_flow_analysis.py` |
| Curvature Magnitude | `scripts/analysis/curvature_magnitude_test.py` |
| Lyapunov Test (fast) | `scripts/analysis/test_lyapunov.py` |
| **Lyapunov (full/directional)** | `scripts/analysis/full_lyapunov_analysis.py` |
| **Wynroe Patching** | `experiments/aha_moment/replicate_wynroe_patching.py` |
| **Pivot Velocity** | `experiments/aha_moment/analyze_pivot_outcome_geometry.py` |

---

## Related Documents

- [PHASE3_H1H2_FINDINGS.md](../../results/PHASE3_H1H2_FINDINGS.md) — Main results with discussion
- [FULL_LYAPUNOV_FINDINGS.md](./FULL_LYAPUNOV_FINDINGS.md) — **Full Lyapunov methodology and analysis**
- [MENGER_CURVATURE_FINDINGS.md](./MENGER_CURVATURE_FINDINGS.md) — Curvature analysis details
- [CURVATURE_HYPOTHESES.md](./CURVATURE_HYPOTHESES.md) — H_flow hypotheses
- [SVD_LINEAR_SEPARABILITY_FINDINGS.md](./SVD_LINEAR_SEPARABILITY_FINDINGS.md) — SVD analysis
- [DATA_COLLECTION_ISSUES.md](./DATA_COLLECTION_ISSUES.md) — Data status and issues
