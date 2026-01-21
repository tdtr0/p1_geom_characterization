# Phase 3 Analysis Summary

**Date**: 2026-01-21
**Status**: Primary analyses complete, secondary analyses pending

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

## Summary of Findings

| Analysis | Signal? | Notes |
|----------|---------|-------|
| Error-Direction (linear probing) | ✅ YES | Strong within-domain, model-dependent transfer |
| Menger Curvature (layers) | ❌ NO | Purely architectural (r≈1.0) |
| SVD Linear Separability | ❌ NO | Tail changes MORE than top (opposite of prediction) |
| Sequence Flow Velocity | ❌ NO | d ~ -0.3, not significant |
| Sequence Flow Curvature | ❌ NO | Still architectural (r > 0.95) |
| Flow Feature Transfer | ⚠️ WEAK | AUC 0.54-0.66, opposite direction |
| Lyapunov (fast) | ❌ NO | Not significant |
| **Wynroe Patching (causal)** | ⚠️ DIFFERENT | 100% recovery at all layers (distributed, not localized like DeepSeek-R1) |
| **Pivot Velocity → Success** | ❌ NO | p=0.55, pivot words are stylistic not functional |

---

## Key Takeaways

1. **Error-detection direction works** — Simple linear probe on activations distinguishes correct/incorrect with d=1-2

2. **Transfer is model-dependent**:
   - SFT: Bidirectional transfer (distillation hypothesis)
   - Base: Asymmetric (math→code only)
   - RL-Zero/Think: No transfer

3. **Geometric measures fail** — Curvature/velocity on raw activations capture architecture, not semantics

4. **Superposition persists** — Even at last layer, geometric measures show r>0.95 correlation

5. **Linear methods insufficient** — SVD analysis shows RLVR changes are distributed in low-variance directions

---

## Analyses NOT Done (Secondary)

| Analysis | Reason Pending |
|----------|----------------|
| Attractor Analysis | Not yet implemented |
| Full Lyapunov Spectrum | Only fast method done |
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
| Lyapunov Test | `scripts/analysis/test_lyapunov.py` |
| **Wynroe Patching** | `experiments/aha_moment/replicate_wynroe_patching.py` |
| **Pivot Velocity** | `experiments/aha_moment/analyze_pivot_outcome_geometry.py` |

---

## Related Documents

- [PHASE3_H1H2_FINDINGS.md](../../results/PHASE3_H1H2_FINDINGS.md) — Main results with discussion
- [MENGER_CURVATURE_FINDINGS.md](./MENGER_CURVATURE_FINDINGS.md) — Curvature analysis details
- [CURVATURE_HYPOTHESES.md](./CURVATURE_HYPOTHESES.md) — H_flow hypotheses
- [SVD_LINEAR_SEPARABILITY_FINDINGS.md](./SVD_LINEAR_SEPARABILITY_FINDINGS.md) — SVD analysis
- [DATA_COLLECTION_ISSUES.md](./DATA_COLLECTION_ISSUES.md) — Data status and issues
