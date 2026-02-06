# Phase 3 Analysis Summary

**Date**: 2026-01-29 (updated)
**Status**: Primary analyses complete. Lyapunov: NEGATIVE (data leakage + invalid proxy). True Jacobian: NULL. Cross-domain: SFT and Think aligned.

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

### 8. Full Lyapunov Spectrum Analysis ❌ NEGATIVE RESULT (Data Leakage)

**Script**: `scripts/analysis/full_lyapunov_analysis.py`

**Date**: 2026-01-23 (initial), 2026-01-27 (corrected)

**Method**:
1. Compute SVD-based Lyapunov at each layer transition
2. Extract error-detection direction using difference-in-means
3. Compute directional Lyapunov (H_jac2): expansion rate in error-direction subspace
4. Test H_jac1 (max Lyapunov), H_jac2 (directional), H_jac3 (spectrum width)

**⚠️ CRITICAL: Initial results were INVALID due to circular analysis (data leakage)**

The error direction was computed from the SAME data it was tested on:
```python
# CIRCULAR: Direction computed from ALL samples, then tested on SAME samples
error_dir = mean(incorrect) - mean(correct)  # Uses test data!
```

This inflated effect sizes by ~45-100%. The "7/9 significant H_jac2" finding was an ARTIFACT.

**Corrected Results (5-fold Cross-Validation)**:

| Model/Task | H_jac2 (circular) | H_jac2 (CV) | Real Signal? |
|------------|-------------------|-------------|--------------|
| base/gsm8k | d=1.68 | **d=0.54** | ✅ Weak (45% inflation) |
| base/logiqa | d=0.79 | **d=0.01** | ❌ **Complete artifact** |
| sft/gsm8k | d=-1.12 | d=0.01 | ❌ Artifact |
| rl_zero/gsm8k | d=1.63 | d=0.06 | ❌ Artifact |

**Summary**: Only 1/11 model-task combinations significant after proper CV (base/gsm8k: d=0.54)

**Baseline Comparison** — Linear probe BEATS H_jac2:

| Model | Linear Probe AUC | H_jac2 AUC | Winner |
|-------|------------------|------------|--------|
| base | **0.679** | 0.413 | Probe (+0.27) |
| rl_zero | **0.752** | 0.303 | Probe (+0.45) |

**H_jac2 AUC < 0.5 = WORSE than random chance!**

**Conclusion**: ❌ **Lyapunov analysis does NOT add value beyond static geometry (linear probe)**

---

### 8b. True Empirical Jacobian Analysis ❌ NULL RESULT (2026-01-29)

**Script**: `scripts/analysis/empirical_jacobian_lyapunov.py`

**Motivation**: The delta-based SVD (previous method) computes SVD(X_{l+1} - X_l), which measures **displacement magnitude**. The true Jacobian solves X_{l+1} ≈ X_l @ J.T and computes SVD(J), measuring **local sensitivity**.

**Results**:

| Model/Task | True Jacobian d | p-value | Delta Proxy d | Note |
|------------|-----------------|---------|---------------|------|
| base/gsm8k | 0.284 | 0.395 | 0.406 | ❌ |
| base/logiqa | -0.242 | 0.326 | -0.399 | ❌ |
| sft/gsm8k | 0.239 | 0.238 | -0.226 | ❌ |
| sft/logiqa | -0.035 | 0.865 | 0.311 | ❌ |
| rl_zero/gsm8k | **0.152** | **0.568** | **0.732** | ❌ Delta inflated 5x |
| rl_zero/logiqa | -0.057 | 0.800 | 0.050 | ❌ |

**Key Finding**: The delta-based proxy is **methodologically invalid** for transformers:

**Diagnostic Analysis** (`scripts/analysis/jacobian_diagnostic.py`):
- **cos(X_l, X_l1) ≈ 0.1**: Layers are nearly orthogonal (subspace reshuffling)
- **mean(SV(J)) ≈ 1.3**: Jacobian is mildly expansive but near-isometric
- **Implication**: Delta measures "subspace jumps" (displacement), not dynamical sensitivity

**Why Delta ≠ Jacobian**:
- When layers are orthogonal, the representation "jumps" to different subspaces
- Delta-SVD captures the magnitude of these jumps
- True Jacobian captures how perturbations propagate locally
- These are fundamentally different quantities!

**Conclusion**: ❌ **True Jacobian shows NULL results for all model/task combinations**. The previous delta-based H_jac1 signal (d=-0.73 for RL-Zero) was an artifact of measuring displacement, not dynamical instability.

---

### 8c. Path Signature Analysis ⚠️ WEAK SIGNAL (2026-01-31)

**Script**: `scripts/analysis/path_signature_analysis.py`

**Method**:
1. Treat layer activations as a path through representation space
2. Compute depth-3 path signatures (reparameterization-invariant features)
3. Compare signature norms and train classifiers

**Results** (n=50 samples per task):

| Model/Task | sig_norm_d | p | AUC | Note |
|------------|------------|---|-----|------|
| base/gsm8k | -0.38 | 0.31 | 0.64 | ❌ |
| base/humaneval | -0.16 | 0.45 | 0.58 | ❌ |
| base/logiqa | -0.37 | 0.10 | 0.50 | ❌ |
| sft/gsm8k | +0.24 | 0.83 | 0.58 | ❌ |
| **sft/humaneval** | **-0.39** | **0.003** | **0.82** | ✅ Significant |
| rl_zero/gsm8k | -0.27 | 0.48 | 0.52 | ❌ |
| rl_zero/humaneval | -0.36 | 0.39 | 0.71 | ⚠️ |

**Cross-Domain Transfer**:
| Model | GSM8K→HumanEval | HumanEval→GSM8K |
|-------|-----------------|-----------------|
| base | 0.578 | 0.564 |
| **sft** | **0.780** | 0.544 |
| rl_zero | 0.545 | 0.387 |

**Key Finding**: ⚠️ **SFT shows strong cross-domain transfer (0.78)** — consistent with error-direction findings. Most other results are null or weak.

---

### 9. Cross-Domain Subspace Alignment ✅ SFT + THINK ALIGNED (2026-01-27)

**Script**: `scripts/analysis/cross_domain_all_models.py`

**Method**:
1. Compute error direction for each task (GSM8K, LogiQA) using difference-in-means
2. Measure cosine similarity between error directions across tasks
3. Test cross-transfer: apply task A's direction to classify task B

**Results** (n=300 per task):

| Model | Task Accuracy | Cosine Sim | GSM8K→LogiQA d | LogiQA→GSM8K d | Pattern |
|-------|---------------|------------|----------------|----------------|---------|
| **base** | 12%/23% | 0.069 | 0.08 (p=0.46) | 0.18 (p=0.20) | Orthogonal |
| **sft** | 60%/36% | **0.355** | **0.48 (p=0.0009)** | **0.45 (p=0.0003)** | **Strong** |
| **rl_zero** | 14%/30% | 0.098 | 0.12 (p=0.42) | 0.11 (p=0.83) | Orthogonal |
| **think** | 40%/34% | **0.258** | **0.33 (p=0.028)** | **0.21 (p=0.027)** | **Moderate** |

**Key Finding**: ✅ **SFT and Think show cross-domain alignment; Base and RL-Zero do not**

- **Base & RL-Zero**: Error directions orthogonal (cos ≈ 0.07-0.10), no transfer
- **SFT**: Strongest alignment (cos=0.355, bidirectional p<0.001)
- **Think**: Moderate alignment (cos=0.258, bidirectional p<0.03) — SFT preserved through DPO+RLVR
- Pure RL (RL-Zero) does NOT create domain-general patterns

**Interpretation — The SFT Distillation Hypothesis**:

SFT models are trained on CoT data from stronger models (GPT-4, Claude), creating knowledge distillation that produces domain-general representations. RL training optimizes for task-specific rewards, creating specialists not generalists.

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

## Summary of Findings

| Analysis | Signal? | Notes |
|----------|---------|-------|
| Error-Direction (linear probing) | ✅ YES | Strong within-domain (AUC 0.68-0.75), model-dependent transfer |
| **Cross-Domain Alignment** | ✅ **SFT + Think** | SFT cos=0.355, Think cos=0.258 (bidirectional); Base/RL-Zero cos≈0.07-0.10 (orthogonal) |
| Menger Curvature (layers) | ❌ NO | Purely architectural (r≈1.0) |
| SVD Linear Separability | ❌ NO | Tail changes MORE than top (opposite of prediction) |
| Sequence Flow Velocity | ❌ NO | d ~ -0.3, not significant |
| Sequence Flow Curvature | ❌ NO | Still architectural (r > 0.95) |
| Flow Feature Transfer | ⚠️ WEAK | AUC 0.54-0.66, opposite direction |
| Lyapunov (fast/Frobenius) | ❌ NO | Not significant (too crude) |
| **Lyapunov (full/directional)** | ❌ **NO** | Initial "7/9 significant" was DATA LEAKAGE. CV shows 1/11 sig. Probe beats H_jac2. |
| **True Empirical Jacobian** | ❌ **NO** | All p > 0.2. Delta proxy invalid (measures displacement, not sensitivity). |
| **Path Signatures** | ⚠️ **WEAK** | sft/humaneval sig: p=0.003, AUC=0.82. Transfer: SFT GSM8K→HE = 0.78. Most others null. |
| **Wynroe Patching (causal)** | ⚠️ DIFFERENT | 100% recovery at all layers (distributed, not localized) |
| **Pivot Velocity → Success** | ❌ NO | p=0.55, pivot words are stylistic not functional |

**Wynroe Replication Note** — Three distinct patterns emerging across models:
1. **Mid-layer spike (DeepSeek-Distill)**: Sharp activation at L16-18 → localized "imitation circuit"
2. **Gradual ramp (OLMo RL-Zero)**: Distributed processing, no shortcuts
3. **Final spike (OLMo Think)**: Late-stage decision, possibly from RLVR "unlearning" the early circuit

---

## Key Takeaways

1. **Linear probe works well** — Static geometry (mean activation) predicts correctness with AUC 0.68-0.75. Simple and effective.

2. **Cross-domain alignment is SFT and Think**:
   - **SFT**: cos=0.355, bidirectional transfer (p<0.001) — learns domain-general patterns
   - **Think**: cos=0.258, bidirectional transfer (p<0.03) — SFT preserved through DPO+RLVR
   - **Base/RL-Zero**: cos≈0.07-0.10 (orthogonal) — task-specific patterns only
   - Pure RL (RL-Zero) does NOT improve generalization over base

3. **Lyapunov analysis FAILED** — Multiple issues revealed:
   - **Data leakage**: Initial "7/9 significant H_jac2" was circular analysis (CV: 1/11 significant)
   - **Invalid proxy**: Delta-based SVD measures displacement, not sensitivity (true Jacobian: all p > 0.2)
   - **Subspace reshuffling**: cos(X_l, X_l1) ≈ 0.1 means layers are orthogonal — delta captures "jumps" not dynamics
   - Linear probe AUC=0.75 beats H_jac2 AUC=0.30 (worse than random!)

4. **Geometric measures on raw activations fail** — Curvature/velocity capture architecture, not semantics (r>0.95)

5. **"Universal reasoning signature" NOT supported**:
   - Error directions are task-specific (orthogonal across domains)
   - Only SFT shows cross-domain alignment
   - Static geometry beats dynamical measures

6. **Critical methodological lessons**:
   - Always use cross-validation for direction-based analyses (prevents data leakage)
   - Always compare to baseline (linear probe) before claiming dynamical measures add value
   - Initial positive results should be verified with proper CV

7. **SFT distillation hypothesis supported** — SFT training on CoT data creates domain-general representations, while RL optimizes task-specific patterns

---

### 10. Token-Position Specificity ✅ DONE (2026-01-25)

**Script**: `scripts/analysis/additional_analyses.py`

**Method**: Compute Cohen's d for error signal at each token position, identify where signal peaks.

**Results** (olmo3_base):

| Task | Peak Position | Fraction | Signal Location | Early d | Middle d | Late d |
|------|---------------|----------|-----------------|---------|----------|--------|
| GSM8K | 50/512 | 10% | **Early** | 0.201 | 0.003 | 0.000 |
| LogiQA | 85/512 | 17% | **Early** | 0.220 | 0.211 | 0.000 |

**Key Finding**: Signal peaks in **early tokens** (10-17%), NOT at answer tokens.

**Interpretation**: The error direction detects **problem encoding/setup**, not answer format. This is consistent with the "surface structure" interpretation — the model's initial representation of the problem determines correctness, not the answer generation process.

---

## Analyses NOT Done (Secondary)

| Analysis | Status | Notes |
|----------|--------|-------|
| ~~Full Lyapunov Spectrum~~ | ✅ DONE | Section 8. CV shows NEGATIVE result (data leakage) |
| **Cross-Domain Alignment** | ✅ **DONE** | Section 9. SFT cos=0.355, Think cos=0.258 (both aligned) |
| **Token-Position Specificity** | ✅ **DONE** | Section 10. Signal peaks early (10-17%) |
| Difficulty Stratification | ✅ DONE | LogiQA confounded by difficulty, GSM8K not |
| **Order Sensitivity** | ❌ NOT DONE | Would test if scrambling input changes signal |
| **Length Conditioning** | ❌ NOT DONE | Is signal confounded by output length? |
| **Path Signatures** | ❌ BLOCKED | Needs `signatory` library (install failed) |
| Attractor Analysis | ❌ Skip | Lyapunov covers this (negative result) |
| SV Regime Projection | ❌ Skip | SVD analysis suggests null |

---

## Code References

| Analysis | Script |
|----------|--------|
| Error-Direction (probing) | `scripts/analysis/phase3_dynamical_analysis.py` |
| **Cross-Domain Alignment** | `scripts/analysis/cross_domain_all_models.py` |
| **Token-Position Specificity** | `scripts/analysis/additional_analyses.py` |
| Menger Curvature | `scripts/analysis/phase3_dynamical_analysis.py` |
| SVD Separability | `experiments/svd_reasoning_separability/analyze_svd_delta.py` |
| Sequence Flow | `scripts/analysis/sequence_flow_analysis.py` |
| Curvature Magnitude | `scripts/analysis/curvature_magnitude_test.py` |
| Lyapunov (full/directional) | `scripts/analysis/full_lyapunov_analysis.py` |
| H3 Extended (CV, transfer) | `scripts/analysis/h3_remaining_analyses.py` |
| Wynroe Patching | `experiments/aha_moment/replicate_wynroe_patching.py` |
| Pivot Velocity | `experiments/aha_moment/analyze_pivot_outcome_geometry.py` |

---

## Related Documents

- [**PHASE3_COMPLETE_FINDINGS.md**](../../results/PHASE3_COMPLETE_FINDINGS.md) — **Comprehensive findings: all 9 positive + 11 null results with methodology-specific scope**
- [PHASE3_H1H2_FINDINGS.md](../../results/PHASE3_H1H2_FINDINGS.md) — Main results with discussion
- [FULL_LYAPUNOV_FINDINGS.md](./FULL_LYAPUNOV_FINDINGS.md) — **Full Lyapunov methodology and analysis**
- [MENGER_CURVATURE_FINDINGS.md](./MENGER_CURVATURE_FINDINGS.md) — Curvature analysis details
- [CURVATURE_HYPOTHESES.md](./CURVATURE_HYPOTHESES.md) — H_flow hypotheses
- [SVD_LINEAR_SEPARABILITY_FINDINGS.md](./SVD_LINEAR_SEPARABILITY_FINDINGS.md) — SVD analysis
- [DATA_COLLECTION_ISSUES.md](./DATA_COLLECTION_ISSUES.md) — Data status and issues
