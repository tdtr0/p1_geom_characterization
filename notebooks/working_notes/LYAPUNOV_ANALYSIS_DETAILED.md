# Lyapunov Analysis: Detailed Methodology and Results

**Date**: 2026-01-24
**Status**: Discovered circular analysis bug; CV analysis running

---

## 1. Background and Motivation

### Why Lyapunov Exponents?

Lyapunov exponents measure the rate of divergence/convergence of nearby trajectories in a dynamical system. In the context of transformers:

- **Positive Lyapunov** (λ > 0): Trajectories diverge exponentially → "chaotic" / unstable
- **Negative Lyapunov** (λ < 0): Trajectories converge → "stable" / attractor behavior
- **Zero Lyapunov** (λ ≈ 0): Trajectories neither diverge nor converge

### Hypothesis (H_jac)

We hypothesized that **correct vs incorrect solutions would have different dynamical signatures**:

1. **H_jac1**: Max Lyapunov differs — incorrect solutions are more "chaotic"
2. **H_jac2**: Directional Lyapunov differs — expansion rate in the **error-detection direction** differs
3. **H_jac3**: Spectrum width differs — incorrect solutions have more anisotropic expansion

The key insight is that while overall chaos (H_jac1) might not differ, **chaos in a specific direction** (H_jac2) might discriminate.

---

## 2. Methodology

### 2.1 Fast Lyapunov (Frobenius Norm) — FAILED

**Script**: `scripts/analysis/phase3_dynamical_analysis.py`

```python
# Crude approximation: overall expansion rate
λ_fast = log(||x_{l+1}|| / ||x_l||)
```

**Result**: ❌ No significant difference (d ~ -0.2 to -0.3, p > 0.1)

**Why it failed**: Frobenius norm averages over ALL directions, masking any directional signal.

### 2.2 Full Lyapunov (SVD-based) — INITIAL SUCCESS

**Script**: `scripts/analysis/full_lyapunov_analysis.py`

**Method**:

1. For each layer transition l → l+1:
   - Compute velocity: `V_l = x_{l+1} - x_l` (shape: seq_len × d_model)
   - Compute SVD: `U, S, Vt = SVD(V_l, k=50)`
   - Lyapunov spectrum: `λ_i = log(s_i / ||x_l||)`

2. Extract error-detection direction:
   ```python
   error_dir = mean(incorrect) - mean(correct)  # Average over all samples
   error_dir = error_dir / ||error_dir||
   ```

3. Compute directional Lyapunov (H_jac2):
   ```python
   for each sample:
       for each layer l:
           proj_l = x_l @ error_dir  # Project onto error direction
           proj_l1 = x_{l+1} @ error_dir
           λ_dir = log(var(proj_l1) / var(proj_l)) / 2  # Variance expansion rate
   ```

**Initial Results** (n=50 samples, k=50 SVD components):

| Model/Task | H_jac1 d | **H_jac2 d** | H_jac3 d | H_jac2 p |
|------------|----------|--------------|----------|----------|
| base/gsm8k | -0.12 | **+1.68** | +0.41 | 0.001 ✓ |
| base/humaneval | -0.16 | **+1.42** | -0.19 | 0.004 ✓ |
| base/logiqa | +0.61 | **+1.41** | -0.66 | 0.000 ✓ |
| sft/gsm8k | +0.10 | **-1.12** | +0.55 | 0.000 ✓ |
| sft/humaneval | -0.68 | +0.81 | +0.88 | 0.07 |
| rl_zero/gsm8k | -0.67 | **+1.63** | +0.06 | 0.000 ✓ |
| rl_zero/humaneval | -0.46 | +0.39 | +0.52 | 0.18 |
| think/gsm8k | -0.19 | **-1.39** | +0.15 | 0.000 ✓ |
| think/humaneval | -0.49 | **+1.37** | +0.74 | 0.002 ✓ |

**Summary**: 7/9 significant for H_jac2, 0/9 for H_jac1, 1/9 for H_jac3

---

## 3. CRITICAL BUG: Circular Analysis in H_jac2

### The Problem

The error direction is computed from **the same data** that it's tested on:

```python
# CIRCULAR: error_dir computed from ALL samples
error_dir = mean(incorrect) - mean(correct)

# Then tested on SAME samples
for i in all_samples:
    λ_dir[i] = compute_directional_lyapunov(trajectory[i], error_dir)
```

This is **data leakage** — the error direction is **defined** to separate correct from incorrect. Testing on the same data inflates the effect size.

### Why the d values are suspiciously high

- d = 0.97 to 1.68 are **very large** effect sizes
- Compare to H_jac1 (non-circular): d = -0.67 to +0.61
- The circularity could inflate d by 2-3x

### The Fix: K-Fold Cross-Validation

Updated method in `h3_remaining_analyses.py`:

```python
for fold in range(5):
    # Split train/test
    train_idx, test_idx = kfold_split(fold)

    # Compute error direction from TRAIN only
    error_dir = mean(incorrect[train_idx]) - mean(correct[train_idx])

    # Test on HELD-OUT samples
    for i in test_idx:
        λ_dir_cv[i] = compute_directional_lyapunov(trajectory[i], error_dir)
```

Also added:
- **Random direction baseline**: What's d for a random direction? Should be ~0.
- **Permutation test**: Shuffle labels 100 times to get null distribution.

---

## 4. Results from H3 Analysis (Old Script — CIRCULAR)

**Date**: 2026-01-23
**Script**: `scripts/analysis/h3_remaining_analyses.py` (before CV fix)

### olmo3_base Results

| Task | n_samples | % Correct | H_jac1 d | H_jac2 d (circular) | H_jac3 d |
|------|-----------|-----------|----------|---------------------|----------|
| gsm8k | 300 | 11.0% | -0.25 | **0.97** | -0.05 |
| humaneval | 300 | 0.0% | — | — | — |
| logiqa | 300 | 23.0% | **0.41** | **0.79** | -0.30 |

**Observations**:

1. **HumanEval skipped** — 0% accuracy for base model (no correct samples)

2. **GSM8K**:
   - H_jac2 d=0.97, p<0.0001 — very large, likely inflated by circularity
   - H_jac1, H_jac3 not significant

3. **LogiQA** — MORE INTERESTING:
   - **H_jac1 significant!** (d=0.41, p=0.003) — This has NO circularity
   - H_jac2 d=0.79 — lower than GSM8K but still high (likely inflated)
   - H_jac3 marginally significant (d=-0.30, p=0.03)

4. **Difficulty confound**:
   - GSM8K: Within 60%, Cross 56% → NOT a confound
   - LogiQA: Within 67%, Cross 53% → **IS a confound** ⚠️

### Interpretation

The **H_jac1 significance for LogiQA** (d=0.41) is a real finding because:
- Max Lyapunov doesn't use the error direction at all
- It's computed purely from the SVD spectrum

This suggests incorrect solutions DO have different dynamical properties on LogiQA, but the effect is small/medium (d=0.4).

---

## 5. Expected True Effect Size (with CV)

Based on general statistical principles and the non-circular H_jac1 result:

| Metric | Old (circular) | Expected CV | Reasoning |
|--------|----------------|-------------|-----------|
| H_jac2 GSM8K | d=0.97 | d=0.2-0.4? | Circularity inflates ~2-3x |
| H_jac2 LogiQA | d=0.79 | d=0.3-0.5? | Similar inflation expected |
| Random baseline | — | d≈0 | Sanity check |

If CV results show d<0.2, the H_jac2 signal is likely **not real** (artifact of circularity).

If CV results show d>0.3, there IS a real signal, just smaller than originally reported.

---

## 6. Methodological Assessment

### Good Parts

1. **H_jac1 and H_jac3 are valid** — No circularity, just not significant
2. **Directional analysis is the right idea** — Frobenius norm is too crude
3. **SVD-based Lyapunov is principled** — Captures actual expansion rates
4. **Cross-domain testing planned** — Would detect overfitting

### Bad Parts

1. **H_jac2 circular analysis** — Major bug, inflates effect size
2. **Small sample sizes** — n=50 in original, n=200-300 in H3 rerun
3. **No baseline comparison** — Should compare to random directions
4. **No permutation test** — Should verify statistical significance empirically

### Lessons Learned

1. **Always use held-out data** for testing direction-based analyses
2. **Include random baselines** — What's the effect for a random direction?
3. **Permutation tests** — Parametric p-values can be misleading

---

## 7. Current Status

**Job 8401** running on tesla2 with fixes:
- 5-fold cross-validation for H_jac2
- Random direction baseline
- 100 permutation tests

**Expected completion**: ~8 hours (8 hours allocated)

Will update this document with CV results when available.

---

## 8. Updated Results (CV Analysis)

**Date**: 2026-01-24
**Job**: 8401 on tesla2

### olmo3_base/gsm8k (n=200, 12.5% correct)

| Metric | Old (circular) | CV | p | perm_p | Significant? |
|--------|----------------|-----|---|--------|--------------|
| H_jac1 (max λ) | d=-0.25 | d=-0.35 | 0.10 | — | ❌ No |
| **H_jac2 (dir λ)** | **d=0.97** | **d=0.54** | **0.01** | **0.01** | ✅ **YES** |
| H_jac2 (random) | — | d=0.03 | 0.90 | — | (baseline) |
| H_jac3 (width) | d=-0.05 | d=-0.09 | 0.67 | — | ❌ No |

**Key**: Circularity inflated d from 0.54 → 0.97 (~45% inflation). Effect is REAL but smaller.

### olmo3_base/logiqa (n=200, 22.5% correct)

| Metric | Old (circular) | CV | p | perm_p | Significant? |
|--------|----------------|-----|---|--------|--------------|
| **H_jac1 (max λ)** | d=0.41 | **d=0.47** | **0.007** | — | ✅ **YES** |
| **H_jac2 (dir λ)** | **d=0.79** | **d=0.01** | 0.93 | 0.89 | ❌ **NO!!** |
| H_jac2 (random) | — | d=-0.16 | 0.34 | — | (baseline) |
| **H_jac3 (width)** | d=-0.30 | **d=-0.42** | **0.015** | — | ✅ **YES** |

**Key**: Circular analysis was ESPECIALLY misleading here (d=0.79 → 0.01). H_jac2 is NOT significant for LogiQA.

### olmo3_base/humaneval

SKIPPED — 0% accuracy (no correct samples for base model on HumanEval).

---

## 9. Critical Finding: Task-Dependent Signatures

| Task | H_jac1 | H_jac2 | H_jac3 | Pattern |
|------|--------|--------|--------|---------|
| **GSM8K** | ❌ -0.35 | ✅ **0.54** | ❌ -0.09 | **Directional** |
| **LogiQA** | ✅ **0.47** | ❌ 0.01 | ✅ **-0.42** | **Global chaos** |

**Interpretation:**

1. **GSM8K (math)**: Correct vs incorrect differs in **expansion direction**
   - There IS a specific "error direction" along which incorrect solutions expand more
   - Overall chaos level (H_jac1) is similar
   - This supports the "error-detection direction" hypothesis

2. **LogiQA (logic)**: Correct vs incorrect differs in **overall dynamics**
   - Incorrect solutions are more chaotic (higher max Lyapunov)
   - Incorrect solutions have narrower expansion spectra (lower width)
   - But NO specific error direction (H_jac2 ≈ 0)
   - This suggests a different mechanism — incorrect logic solutions are unstable in ALL directions

**Why the difference?**

Possible explanation:
- **Math errors** are structured — they involve wrong operations or values, which create a specific error "direction" in activation space
- **Logic errors** are unstructured — they involve missing connections or wrong inferences, which create general instability

---

## 10. Lessons Learned

1. **Circular analysis can inflate effect sizes by 45-100%**
   - GSM8K H_jac2: 0.97 → 0.54 (45% inflation)
   - LogiQA H_jac2: 0.79 → 0.01 (complete artifact!)

2. **Cross-validation is ESSENTIAL for directional analyses**
   - Error direction must be computed on held-out data
   - Random baseline confirms the direction is meaningful

3. **Different tasks have different signatures**
   - One-size-fits-all analysis misses the pattern
   - Need to test multiple hypotheses (H_jac1, H_jac2, H_jac3)

4. **The 7/9 "significant" findings in Phase 3 need reassessment**
   - Some may be artifacts of circular analysis
   - Need to rerun all with proper CV

**Job crashed due to numpy bool JSON serialization bug** — Fixed and can resubmit for other models.

---

## 11. Final Assessment

### Is the H_jac2 (Directional Lyapunov) Signal Real?

**Answer: YES for GSM8K, NO for LogiQA**

| Task | Old (circular) d | CV d | Real Signal? |
|------|-----------------|------|--------------|
| GSM8K | 0.97 | **0.54** | ✅ YES (medium effect) |
| LogiQA | 0.79 | **0.01** | ❌ NO (complete artifact) |

### Summary

1. **GSM8K**: The directional Lyapunov signal IS REAL
   - Effect size is medium (d=0.54), not large as originally reported
   - Incorrect solutions expand more along the error-detection direction
   - This is a genuine dynamical signature of incorrect math reasoning

2. **LogiQA**: The directional Lyapunov signal was ENTIRELY AN ARTIFACT
   - With proper CV, d=0.01 (essentially zero)
   - The original d=0.79 was entirely due to circular analysis
   - However, H_jac1 (global chaos) and H_jac3 (spectrum width) ARE significant
   - Logic errors have different signatures than math errors

3. **Methodology lesson**: Cross-validation is ESSENTIAL for directional analyses
   - Random baseline and permutation tests confirm validity
   - Circular analysis can inflate effect sizes by 45-100%+

### Recommendations

1. The original Phase 3 claim of "7/9 significant H_jac2" should be revised
2. Need to rerun all Phase 3 analyses with proper CV
3. Different tasks (math vs logic) have fundamentally different error signatures
4. H_jac2 works for structured errors (math), not unstructured errors (logic)

---

## 12. Critical Interpretation: Surface Structure vs Reasoning Capability

### The Key Observation

H_jac2 shows a significant signal (d=0.54) on **olmo3_base** for GSM8K, but the base model only achieves **12.5% accuracy** on GSM8K. This raises a fundamental question:

**If the base model can barely solve math problems, what is H_jac2 actually detecting?**

### Connection to SVD/Eigenvector Analysis

Our earlier SVD analysis (see [SVD_LINEAR_SEPARABILITY_FINDINGS.md](SVD_LINEAR_SEPARABILITY_FINDINGS.md)) showed:
- **Top eigenvectors** don't change with RLVR training (structural preservation)
- **Tail eigenvectors** change 3-8x more than top eigenvectors
- This means the "error direction" likely exists in **pre-training structure**, not learned reasoning

### Interpretation: Surface-Level Format Detection

The H_jac2 signal likely detects **surface-level format structure** rather than reasoning capability:

| What H_jac2 Might Detect | Evidence |
|--------------------------|----------|
| **Mathematical notation patterns** | Base model has seen math in training |
| **Answer format structure** | `#### <number>` pattern recognition |
| **Numeric token sequences** | Structural patterns in number generation |
| **Template compliance** | Whether output follows expected format |

### Why This Matters

1. **Not Detecting Reasoning**: The base model (12.5% accuracy) doesn't "reason" about math — it produces formatted outputs that sometimes happen to be correct
2. **Pre-Existing Structure**: The error direction exists in the base model's representation space before any task-specific training
3. **Format vs Content**: H_jac2 may distinguish "well-formatted wrong answers" from "malformed outputs" rather than "correct reasoning" from "incorrect reasoning"

### Implications for H2 (Cross-Domain Transfer)

If H_jac2 detects **format/surface structure**:
- ❌ Cross-domain transfer would fail (math format ≠ code format ≠ logic format)
- ❌ Not useful for detecting "reasoning quality" in general
- ✅ Might still be useful for format-specific verification within domains

### Alternative Analyses Needed

To separate surface structure from reasoning capability:

| Analysis | What It Would Show |
|----------|-------------------|
| **Path Signatures** | Higher-order trajectory structure (not just mean difference) |
| **Conditional Analysis** | Split by difficulty/length to control confounds |
| **Token-Position Specificity** | Where in sequence the signal emerges |
| **Cross-Domain Subspace Alignment** | Whether "error directions" align across domains |

### Conclusion

**H_jac2 (d=0.54) is a REAL signal, but likely reflects surface-level format structure rather than reasoning capability.** This is consistent with:
1. The signal existing in base model (before task-specific training)
2. Top SVD components being preserved across training
3. The fundamentally different signatures for math (H_jac2) vs logic (H_jac1/H_jac3)

This finding reframes our interpretation: we're detecting **structural differences** in how the model generates formatted outputs, not **capability differences** in reasoning.

---

## 13. Status of Additional Analyses

The following analyses address the surface structure vs reasoning question:

| Analysis | Status | Implementation | Notes |
|----------|--------|----------------|-------|
| **Path Signatures** | **NOT IMPLEMENTED** | Would need `signatory` library | Captures higher-order trajectory structure (iterated integrals) - would detect non-linear patterns missed by mean differences |
| **Conditional Analysis** | **PARTIAL** | `difficulty_stratification()` in `h3_remaining_analyses.py` | Splits by sequence length tertiles (easy/medium/hard), tests cross-stratum transfer. Results show difficulty IS a confound for LogiQA |
| **Token-Position Specificity** | **NOT IMPLEMENTED** | Would need per-token Jacobian norms | Find "decision tokens" where error signal peaks - could reveal if signal is at answer vs reasoning tokens |
| **Cross-Domain Subspace Alignment** | **PARTIAL** | `test_direction_transfer()` in `phase3_dynamical_analysis.py` | Tests if error direction from one task predicts on another. Initial results: math→logic transfer ~50% (chance) |

### Difficulty Stratification Results (from h3_remaining_analyses.py)

The `difficulty_stratification` analysis was run as part of H3:

| Task | Within-Stratum Acc | Cross-Stratum Acc | Difficulty Confound? |
|------|-------------------|-------------------|---------------------|
| GSM8K | 60% | 56% | ❌ No (4% drop) |
| LogiQA | 67% | 53% | ✅ **YES** (14% drop) |

**Interpretation**:
- GSM8K: Error direction generalizes across difficulty levels → NOT just detecting easy/hard
- LogiQA: Error direction fails to transfer across difficulty → IS confounded by difficulty

### Cross-Domain Transfer Results (from phase3_dynamical_analysis.py)

Initial cross-domain testing (before CV fix):

| Train → Test | Accuracy | Interpretation |
|--------------|----------|----------------|
| GSM8K → LogiQA | ~52% | Chance level - no transfer |
| LogiQA → GSM8K | ~48% | Chance level - no transfer |
| GSM8K → HumanEval | ~51% | Chance level - no transfer |

**Interpretation**: Error directions are task-specific, NOT domain-general. This is consistent with the "surface structure" hypothesis — each task has its own format/structure patterns.

### Analyses Still Needed

1. **Path Signatures**: Would require implementing with `signatory` library. High value for detecting non-linear trajectory patterns.

2. **Token-Position Specificity**:
   - Compute Jacobian norm ||∂x_{l+1}/∂x_l|| per token position
   - Find where signal peaks (answer tokens vs reasoning tokens)
   - Would definitively show if we're detecting answer format vs reasoning process

3. **Post-CV Cross-Domain Transfer**: The cross-domain results above used the circular H_jac2. Need to rerun with CV-corrected error directions.

---

## 14. Related Files

- Original Lyapunov: `scripts/analysis/full_lyapunov_analysis.py`
- H3 Analysis (fixed): `scripts/analysis/h3_remaining_analyses.py`
- SLURM job: `scripts/deployment/run_h3_analyses.sbatch`
- Results: `results/h3_remaining/`
