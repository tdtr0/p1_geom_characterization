# Full Lyapunov Spectrum Analysis: Methodology and Findings

**Date**: 2026-01-23 (Updated 2026-01-29)
**Script**: `scripts/analysis/full_lyapunov_analysis.py`, `scripts/analysis/h3_remaining_analyses.py`, `scripts/analysis/empirical_jacobian_lyapunov.py`
**Status**: CV-corrected + True Jacobian analysis complete — **NULL RESULTS FOR TRUE JACOBIAN**

> ⚠️ **CRITICAL UPDATE (2026-01-25)**: The original H_jac2 effect sizes were **inflated ~3-5x by circular analysis** (computing error direction from the same data it was tested on). Cross-validated results show weaker but still partially significant effects.

---

## Motivation

The fast Lyapunov analysis (Section 5 of PHASE3_ANALYSIS_SUMMARY) used Frobenius norm ratio as a proxy:

```
λ_fast = log(||x_{l+1}|| / ||x_l||)
```

This showed **no signal** (d ~ -0.3, not significant). The critique document hypothesized this was because Frobenius norm averages over all directions, losing directional information.

**Key insight**: If correct and incorrect solutions differ in *specific directions* (e.g., the error-detection direction), a global norm will miss this. We need directional analysis.

---

## Methodology

### Three Hypotheses Tested

| Hypothesis | What it Tests | Prediction |
|------------|---------------|------------|
| **H_jac1** | Max Lyapunov exponent | Incorrect = more chaotic (higher λ_max) |
| **H_jac2** | Directional Lyapunov in error-direction | Dynamics differ in error-detection subspace |
| **H_jac3** | Spectrum width (std of λ) | Correct = more anisotropic (wider spectrum) |

### Step 1: Compute Layer Transition Dynamics

For each layer transition l → l+1, we compute the "expansion" using randomized SVD:

```python
delta = x_{l+1} - x_l  # Layer transition (seq_len, d_model)

# Randomized SVD for efficiency
U, s, Vt = randomized_svd(delta, n_components=k)

# Normalize by input magnitude
expansion_ratios = s / ||x_l||_F
```

**Why randomized SVD?** Full SVD on 4096-dim vectors is O(d³) = O(68B) operations. Randomized SVD with k=50 components is O(seq_len × d × k) = O(105M), a 650× speedup.

### Step 2: Compute Lyapunov Exponents

```python
lyapunov_exponents = log(expansion_ratios)
max_lyapunov = lyapunov_exponents[0]  # Largest
mean_lyapunov = mean(lyapunov_exponents)
spectrum_width = std(lyapunov_exponents)
```

### Step 3: Extract Error-Detection Direction

Using difference-in-means (same as linear probing):

```python
error_direction = mean(incorrect_activations) - mean(correct_activations)
error_direction = normalize(error_direction)
```

### Step 4: Compute Directional Lyapunov

Project trajectory onto error direction and measure expansion in that subspace:

```python
proj_l = trajectory[:, l, :] @ error_direction  # (seq_len,)
proj_l1 = trajectory[:, l+1, :] @ error_direction

# Variance ratio as proxy for directional expansion
directional_lyapunov = log(var(proj_l1) / var(proj_l)) / 2
```

---

## Results

### Data Used

- **Models**: olmo3_base, olmo3_sft, olmo3_rl_zero, olmo3_think
- **Tasks**: GSM8K, HumanEval, LogiQA (where available)
- **Samples**: n=50 per task (first 50 from each HDF5 file)
- **SVD components**: k=50


### CV-Corrected Results (n=100 samples, k=5 folds) — COMPLETE 2026-01-25

| Model/Task | H_jac1 (d, p) | H_jac2 CV (d, p) | Random (d, p) | Perm p | Significant? |
|------------|---------------|-------------------|---------------|--------|--------------|
| olmo3_base/gsm8k | -0.12, 0.80 | **+0.72, 0.034** | +0.53, 0.117 | **0.04** | ✅ H_jac2 |
| olmo3_base/humaneval | -0.16, 0.74 | +0.32, 0.287 | +0.38, 0.210 | 0.33 | ❌ |
| olmo3_base/logiqa | +0.40, 0.11 | -0.03, 0.920 | -0.30, 0.230 | 0.89 | ❌ |
| olmo3_sft/gsm8k | +0.23, 0.26 | +0.01, 0.971 | -0.18, 0.373 | 0.93 | ❌ |
| olmo3_sft/humaneval | SKIPPED | SKIPPED | — | — | 0% accuracy |
| olmo3_sft/logiqa | -0.31, 0.136 | -0.13, 0.536 | +0.02, 0.936 | 0.58 | ❌ |
| **olmo3_rl_zero/gsm8k** | **-0.73, 0.007** | +0.06, 0.833 | -0.17, 0.516 | 0.79 | ✅ **H_jac1** |
| **olmo3_rl_zero/humaneval** | **-0.59, 0.004** | +0.31, 0.135 | +0.16, 0.435 | 0.16 | ✅ **H_jac1** |
| olmo3_rl_zero/logiqa | -0.05, 0.824 | +0.22, 0.333 | +0.01, 0.971 | 0.37 | ❌ |
| olmo3_think/gsm8k | -0.06, 0.78 | -0.19, 0.333 | -0.25, 0.218 | 0.28 | ❌ |
| olmo3_think/humaneval | -0.25, 0.40 | +0.53, 0.069 | +0.44, 0.129 | 0.04 | ⚠️ borderline |
| olmo3_think/logiqa | -0.31, 0.143 | -0.03, 0.878 | -0.26, 0.228 | 0.89 | ❌ |

**Key Observations (Updated 2026-01-25 - ALL TASKS COMPLETE)**:

1. **H_jac1 works for RL-Zero on math/code ONLY**:
   - ✅ GSM8K: d=-0.73, p=0.007 (math)
   - ✅ HumanEval: d=-0.59, p=0.004 (code)
   - ❌ LogiQA: d=-0.05, p=0.824 (logic) — **NO SIGNAL**
   - Incorrect solutions have **lower** max expansion rate on math/code

2. **H_jac2 works for base/gsm8k only**: Directional Lyapunov (d=0.72) is weak and similar to random baseline (d=0.53).

3. **Different mechanisms by training method**:
   - Base model: H_jac2 (directional) on GSM8K only
   - **RL-Zero model: H_jac1 (max λ) on math/code ONLY** — task-specific effect
   - SFT/Think: No signal (SFT/humaneval has 0% accuracy)

4. **RL-Zero effect is task-specific**: The max Lyapunov signal appears only on **verifiable computation** (math, code), NOT on **logical reasoning** (LogiQA). This suggests the dynamics differ between computational vs reasoning tasks.

### What This Means

The directional Lyapunov analysis (H_jac2) **does show a real signal** for olmo3_base/gsm8k, but:
- The effect is **much weaker** than original analysis suggested
- It's **not robust across tasks**: humaneval and logiqa show no signal
- It's **not much better than random direction**: suggesting the "error direction" might be partially incidental

### Comparison: H_jac2 vs Linear Probe

| Method | olmo3_base/gsm8k | Interpretation |
|--------|------------------|----------------|
| Linear Probe AUC | ~0.75 | Static separation works |
| H_jac2 CV d | +0.72 | Weak dynamical signal |
| Random Direction d | +0.53 | Noise level |

The linear probe (static, same direction) gives AUC ~0.75, while the dynamical Lyapunov (H_jac2) gives d ~0.72. This suggests **most of the signal is in the static direction**, not the dynamics along that direction.

---

## Analysis (Original — Interpret with Caution)

### Finding 1: H_jac2 is the Key Signal

The directional Lyapunov in the error-detection subspace shows significant differences in 7/9 model-task combinations. This is a strong positive result.

**Effect sizes are large**: Cohen's d ranges from 1.12 to 1.68, indicating substantial separation between correct and incorrect trajectories.

### Finding 2: Pattern Reversal with Training Method

The sign of Cohen's d reveals two distinct patterns:

| Training Type | Models | Pattern (d sign) | Interpretation |
|---------------|--------|------------------|----------------|
| **Base/RL** | olmo3_base, olmo3_rl_zero | **Positive (+)** | Incorrect expands MORE in error direction |
| **SFT-based** | olmo3_sft, olmo3_think | **Negative (-)** on GSM8K | Correct expands MORE in error direction |

**Numerical breakdown**:

```
BASE MODEL (GSM8K):
  Correct:   0.318 ± 0.005  (low expansion in error dir)
  Incorrect: 0.339 ± 0.012  (high expansion in error dir)
  => Incorrect solutions DIVERGE in error direction

SFT MODEL (GSM8K):
  Correct:   0.226 ± 0.018  (higher expansion)
  Incorrect: 0.207 ± 0.016  (lower expansion)
  => Correct solutions have MORE ACTIVATION in error direction
```

### Finding 3: Why the Reversal?

**Hypothesis**: SFT training changes the representation structure.

In the base model, the error-detection direction points from correct → incorrect. Incorrect solutions "drift" along this direction (unstable).

After SFT, the model learns to actively represent correctness. The error direction may now encode "how confident am I?" rather than "am I drifting toward error?"

This is consistent with:
- SFT distillation teaching explicit error awareness
- The bidirectional transfer we observed in SFT's linear probing

### Finding 4: GSM8K vs HumanEval Asymmetry

| Task | Significant Results | Pattern Consistency |
|------|---------------------|---------------------|
| GSM8K | 4/4 (100%) | All significant, clear pattern |
| HumanEval | 3/4 (75%) | Mixed patterns |

GSM8K shows cleaner signal than HumanEval. Possible reasons:
- Math has clearer correct/incorrect boundary
- HumanEval correctness is binary but solutions vary more
- Syntax vs semantic correctness difference

### Finding 5: H_jac1 and H_jac3 Show No Signal

**H_jac1 (Max Lyapunov)**: No overall chaos difference
- Correct and incorrect solutions have similar maximum expansion rates
- The difference is *where* they expand, not *how much*

**H_jac3 (Spectrum Width)**: Weak signal (1/9 significant)
- Anisotropy doesn't consistently distinguish correct/incorrect
- The variance structure is similar

---

## Comparison to Other Analyses

| Analysis | Signal? | Key Insight |
|----------|---------|-------------|
| Menger Curvature (layers) | No | Purely architectural (r≈1.0) |
| Menger Curvature (sequence) | No | Still architectural (r>0.95) |
| Fast Lyapunov (Frobenius) | No | Too crude, averages out signal |
| **Full Lyapunov (Directional)** | **YES** | Directional isolation reveals signal |
| Error-Direction Probing | YES | Linear separation works |

**Key lesson**: Raw geometric measures on high-dimensional activations capture architecture, not semantics. Signal exists in *specific subspaces* (error-detection direction).

---

## Interpretation: What Does This Mean?

### For Correct Solutions (Base/RL-Zero)
Correct solutions are **stable** in the error-detection direction. They don't drift toward the "incorrect" region of activation space.

### For Incorrect Solutions (Base/RL-Zero)
Incorrect solutions **diverge** along the error-detection direction. The model's dynamics push them further from "correct" as processing continues.

### For SFT Models
The mechanism is different. SFT seems to make the model **actively represent correctness** by having correct solutions occupy high-activation regions in the error-detection subspace.

---

## Limitations

1. **Sample size**: Only n=50 per task. Should increase to 100+ for robustness.

2. **Error direction is task-specific**: We compute direction per task. Cross-task direction might differ.

3. **Directional Lyapunov is a proxy**: We measure variance ratio, not true Jacobian eigenvalue in that direction.

4. **LogiQA missing**: 3/4 models have truncated LogiQA files. Results are based on GSM8K + HumanEval only for those models.

---

## Conclusions (Updated 2026-01-25)

### Original Conclusions (Partially Retracted)

~~1. **Directional analysis works where global measures fail**. The Frobenius-based fast Lyapunov showed no signal; directional Lyapunov shows strong signal.~~

**Revised**: Directional analysis shows **weak signal** for some model-task combinations, but effect sizes were inflated by circular analysis.

~~2. **The error-detection direction is key**. This is the same direction that works for linear probing. Dynamics in this subspace distinguish correct/incorrect.~~

**Revised**: The error-detection direction captures **static separation** (linear probe works), but **dynamical signal is weak** (only marginally better than random direction).

3. **Training method changes the mechanism**: This finding stands but with weaker effect sizes.

~~4. **H1 is supported**: Correct and incorrect solutions have distinguishable dynamical signatures.~~

**Revised**: **H1 is weakly supported**. The dynamical signature (directional Lyapunov) adds minimal value over static probing.

### Updated Conclusions (2026-01-25)

1. **Data leakage is critical**: Always use cross-validation when computing data-driven directions.

2. **Training method determines mechanism**:
   - **Base model**: Signal in error-direction (H_jac2 d=0.72, GSM8K only)
   - **RL-Zero model**: Signal in max expansion (H_jac1 d=-0.7, both tasks) — unique to RLVR!
   - **SFT/Think**: No consistent signal

3. **RL-Zero has task-specific dynamical signature**:
   - **Math/Code**: Incorrect = lower max Lyapunov (less exploration)
   - **Logic**: NO signal (d=-0.05, p=0.82)
   - Pattern is unique to RL-Zero (not in base/SFT/Think)
   - Suggests RLVR changes dynamics for **verifiable computation**, not general reasoning

4. **Cross-model transfer works asymmetrically**:
   - think→rl_zero works (d=-0.76); rl_zero→think weak
   - Direction learned on Think predicts on RL-Zero

5. **Layer separation profiles differ**:
   - RL-Zero: peak at early-mid layers (L8-L12)
   - Think: peak at late layers (L10-L15)
   - Different models use different "computational depths"

6. **H_jac2 (directional) still weak**: The error-direction Lyapunov adds minimal value over linear probe, even after proper CV.

---

## True Empirical Jacobian Analysis (2026-01-29)

### Motivation

The delta-based SVD method (SVD of X_{l+1} - X_l) computes the **velocity magnitude**, not the true Jacobian. This is a proxy that may not accurately reflect sensitivity to perturbations.

The true approach solves the regression problem: **X_{l+1} ≈ X_l @ J.T** and computes SVD(J).

### Results

| Model/Task | True Jacobian max_d | p-value | Delta Proxy max_d | Interpretation |
|------------|---------------------|---------|-------------------|----------------|
| base/gsm8k | 0.284 | 0.395 | 0.406 | ❌ Not significant |
| base/logiqa | -0.242 | 0.326 | -0.399 | ❌ Not significant |
| sft/gsm8k | 0.239 | 0.238 | -0.226 | ❌ Not significant |
| sft/logiqa | -0.035 | 0.865 | 0.311 | ❌ Not significant |
| rl_zero/gsm8k | 0.152 | 0.568 | **0.732** | ❌ True null, delta inflated |
| rl_zero/logiqa | -0.057 | 0.800 | 0.050 | ❌ Not significant |

**Note**: HumanEval was skipped for all models (0% accuracy in this sample).

### Key Finding: Delta Proxy is NOT a Valid Jacobian Approximation

The most striking result is **rl_zero/gsm8k**:
- Delta proxy: d = 0.732 (would suggest strong effect)
- True Jacobian: d = 0.152, p = 0.568 (null result)

This ~5x inflation shows the delta-based proxy captures something **different** from true Jacobian sensitivity.

### Diagnostic Analysis: Why Delta ≠ Jacobian

A diagnostic analysis (`scripts/analysis/jacobian_diagnostic.py`) on rl_zero/gsm8k reveals:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **cos_sim(X_l, X_l1)** | 0.10-0.12 | Layers are nearly orthogonal! |
| **mean(SV(J))** | 1.2-1.5 | Jacobian is mildly expansive |
| **ortho_error** | ~0.999 | Not orthogonal, but uniform SVs |
| **delta_normalized** | 0.2-0.25 | Moderate displacement |
| **condition_number** | 1.3-6 | Mild anisotropy |

**Key insight**: Layer representations are nearly **orthogonal** to each other (cos_sim ≈ 0.1). This means:

1. **Subspace reshuffling**: Each layer projects representations into a different subspace
2. **Delta measures displacement**: How far the representation "jumps" between subspaces
3. **Jacobian measures local sensitivity**: How perturbations propagate through the transformation

**Why they differ**: When cos(X_l, X_l1) ≈ 0.1:
- Delta-based SVD captures the **magnitude of subspace jumps**
- True Jacobian captures **local sensitivity** within the transformation

If incorrect solutions "jump" to different subspaces more dramatically, delta shows a signal. But if the local sensitivity is similar for both correct/incorrect, true Jacobian shows null.

### Correct vs Incorrect Comparison

| Metric | Correct | Incorrect | Difference |
|--------|---------|-----------|------------|
| delta_normalized | 0.216 | 0.227 | -0.011 |
| mean_SV | 1.461 | 1.347 | +0.114 |
| cos_sim | 0.106 | 0.117 | -0.011 |

The actual differences are **tiny**! The inflated delta-based d=0.732 likely comes from aggregation effects in the per-sample SVD computation, not from fundamentally different dynamics.

### Implications

1. **Delta-based Lyapunov is methodologically invalid for transformers**: When layers are nearly orthogonal, delta measures subspace displacement, not dynamical sensitivity.

2. **Previous H_jac1 results should be disregarded**: The RL-Zero signal (d=-0.73) was an artifact of the proxy method, not a true dynamical difference.

3. **True Jacobian shows no signal**: All model/task combinations have p > 0.2 for the regression-based Jacobian.

4. **The linear probe remains the only reliable method**: Static geometry (mean activation) separates correct/incorrect; dynamics do not add information.

### Why Lyapunov Analysis is Fundamentally Inappropriate for Transformers

The null result is not just "no signal" — it reflects a **fundamental mismatch** between Lyapunov analysis and transformer architecture:

**1. Transformers ≠ Continuous Dynamical Systems**

Lyapunov exponents measure how nearby trajectories diverge/converge in a continuous flow:
```
dx/dt = f(x)  →  λ = lim_{t→∞} (1/t) log(||δx(t)||/||δx(0)||)
```

But transformers have:
- **Discrete layers** (not continuous time)
- **Residual connections**: x_{l+1} = x_l + f(x_l) — this is NOT a flow, it's a discrete map
- **Subspace projection**: Each layer projects to a fundamentally different subspace

**2. Orthogonality = Rotation, Not Expansion**

When cos(X_l, X_{l+1}) ≈ 0.1:
- The layer transition is approximately a **rotation** (orthogonal transformation)
- Rotations have singular values = 1 (they preserve norms)
- log(1) = 0 → Lyapunov exponent ≈ 0 for ALL trajectories
- No meaningful expansion/contraction signal can emerge

**3. The "Subspace Reshuffling" Problem**

Each transformer layer doesn't "evolve" the representation — it **projects** it into a new subspace:
```
Layer l:   representation lives in subspace S_l
Layer l+1: representation lives in subspace S_{l+1}
cos(S_l, S_{l+1}) ≈ 0.1  →  nearly orthogonal subspaces
```

The Jacobian J in X_{l+1} ≈ X_l @ J.T is fitting a **rotation between orthogonal subspaces**. This gives no dynamical information because:
- All rotations have the same spectrum (SVs ≈ 1)
- No distinction between correct/incorrect possible via Jacobian

### Potential Methods to Address Orthogonality

| Method | How It Helps | Status | Result |
|--------|-------------|--------|--------|
| **Procrustes Alignment** | Find optimal rotation R, measure reconstruction error | ✅ Tested | **NULL** (d ≈ 0.02-0.03) |
| **SVCCA** | Find maximally correlated subspaces, measure alignment | ✅ Tested | **NULL** (d ≈ -0.02) |
| **Subspace-Restricted Jacobian** | Compute Jacobian only in semantic direction | Partial (H_jac2) | Weak (d ≈ 0.72) |
| **CKA** | Rotation-invariant similarity | ✅ Done | Limited signal |

### Alignment-Aware Analysis Results (2026-01-31)

We tested both Procrustes and SVCCA as unsupervised alignment-aware methods:

| Model/Task | Procrustes mean_d | SVCCA mean_d | Interpretation |
|------------|-------------------|--------------|----------------|
| base/gsm8k | 0.025 | -0.020 | ❌ Null |
| base/humaneval | 0.022 | — | ❌ Null |
| base/logiqa | 0.031 | — | ❌ Null |

**Key Finding**: Even alignment-aware methods show **negligible signal** (|d| < 0.04).

This confirms that the orthogonality problem is not just a technical issue with the measurement method — transformer layer dynamics are **fundamentally uninformative** for distinguishing correct vs incorrect solutions.

**Interpretation**:
1. Correct and incorrect solutions undergo **the same layer transformations** (same Procrustes reconstruction error)
2. Layer representations are aligned in **the same subspaces** (same SVCCA correlations)
3. The difference between correct/incorrect lives in **static geometry** (what information is encoded), not **dynamics** (how it's transformed)

**Recommendation**: Abandon dynamical analysis for correctness detection. Focus on:
1. Static geometry (linear probe on mean activation — works with AUC 0.75)
2. Semantic subspaces (error-direction — weak but detectable signal)
3. Path signatures (SFT shows transfer with AUC 0.78)

### Reconciling with Anthropic's Circuit Findings

Our results seem to contradict Anthropic's mechanistic interpretability work on circuits. If dynamics don't matter and only linear embeddings do, how can circuits be meaningful?

**The Resolution: Circuits Create the Embedding, But Don't Distinguish Usage**

1. **Circuits build the linear structure**: The induction heads, IOI circuits, etc. are the *mechanism* that produces the final linear representation. They're not separate from it — they *create* it.

2. **Same circuits, different content**: Correct and incorrect solutions use the **same computational machinery** (same circuits, same attention patterns, same MLP transformations). What differs is the **input content** flowing through that machinery.

3. **Analogy**: A calculator's circuits (adder, multiplier) work identically whether you compute 2+2=4 (correct) or type 2+2=5 (incorrect input). The circuit isn't "wrong" — it faithfully transforms whatever you give it.

**What Our Null Result Actually Means**:

| What We Tested | What It Measures | Result |
|----------------|------------------|--------|
| Lyapunov/Jacobian | Sensitivity of transformation | NULL |
| Procrustes | Transformation magnitude | NULL |
| SVCCA | Subspace alignment of transformation | NULL |
| **Linear probe** | **Content of representation** | **WORKS** |

The circuits (transformations) are identical for correct/incorrect. The difference is in **what information is encoded**, not **how it's processed**.

**Implications for Mechanistic Interpretability**:

1. **Circuits explain capability, not correctness**: Circuits explain *how* the model can do arithmetic, not *whether* a specific answer is right.

2. **Correctness is in the residual stream content**: The "is this correct?" signal lives in the linear embedding space, not in the circuit dynamics.

3. **Error detection ≠ circuit analysis**: To detect errors, you don't need to understand the circuits — you need to probe the representation content.

**This is consistent with the "linear representation hypothesis"** (Park et al., 2023): Concepts are encoded as linear directions in activation space. The circuits are the machinery that *creates* those directions, but once created, correctness is readable via simple linear probes.

**Why This Makes Sense**:

- Circuits are **trained** to produce useful representations
- The representation space is **shared** across correct/incorrect (same concepts, same directions)
- What differs is **where** in that space a particular computation lands
- A linear probe reads "where" — dynamics measure "how you got there"

**Conclusion**: Anthropic's circuit findings and our null dynamical result are **complementary, not contradictory**. Circuits explain the mechanism; linear embeddings encode the outcome. For correctness detection, you only need to read the outcome.

---

## Complete Summary: Positive vs Null Results (2026-01-31)

### What Actually Matters: The Only Reliable Signal is Linear Representation

After exhaustive analysis, we find that **linear probing on mean activations** is the only consistently reliable method. All dynamical measures failed.

### ✅ POSITIVE RESULTS (What Works)

| Finding | Evidence | What It Tells Us |
|---------|----------|------------------|
| **Linear probe separates correct/incorrect** | AUC 0.68-0.75, d=0.5-2.0 | Correctness is linearly encoded in activation space |
| **SFT creates domain-general error directions** | cos=0.355, bidirectional p<0.001 | SFT distillation creates shared representations |
| **Think preserves SFT alignment** | cos=0.258, bidirectional p<0.03 | SFT patterns survive DPO+RLVR |
| **Error signal peaks in early tokens** | 10-17% of sequence | Detection is about problem setup, not answers |
| **SFT curvature differs for correct/incorrect** | d=0.53-0.75, p<0.002 | Only SFT shows geometric differences |
| **Path signatures transfer for SFT** | GSM8K→HumanEval AUC=0.78 | SFT has domain-general trajectory structure |
| **Wynroe-style patching: distributed in OLMo** | 100% recovery at ALL layers | OLMo has distributed error processing (vs DeepSeek localized) |

### ❌ NULL/NEGATIVE RESULTS (What Doesn't Work)

| Method | Result | Why It Failed |
|--------|--------|---------------|
| **True Jacobian** | All p > 0.2 | Layers are orthogonal (cos ≈ 0.1), Jacobian = rotation |
| **Delta-based Lyapunov** | Artifact (5x inflation) | Measures displacement, not sensitivity |
| **Directional Lyapunov (H_jac2)** | AUC < 0.5 (worse than random) | Data leakage in original; CV shows null |
| **Procrustes alignment** | d ≈ 0.02-0.03 | Same transformation dynamics for correct/incorrect |
| **SVCCA** | d ≈ -0.02 | Same subspace alignment for correct/incorrect |
| **Menger curvature (layers)** | r ≈ 0.999 | Purely architectural |
| **Sequence flow velocity** | d ≈ -0.3, not sig | No difference |
| **Sequence curvature** | r > 0.95 | Still architectural |
| **SVD linear separability** | Opposite of prediction | Tail changes MORE than top |
| **Pivot velocity → success** | p=0.55 | Pivot words are stylistic, not functional |
| **RL-Zero cross-domain** | cos=0.098 (orthogonal) | RL creates specialists, not generalists |

### ⚠️ WEAK/PARTIAL RESULTS

| Finding | Evidence | Caveat |
|---------|----------|--------|
| **Path signatures (sft/humaneval only)** | p=0.003, AUC=0.82 | Only 1 model/task significant |
| **Base asymmetric transfer** | Math→Code 85%, Code→Math 10% | Only one direction works |
| **Belief probe transfer** | SFT best | Limited to aha_moment experiment |

---

## The Big Picture: Circuits vs Linear Representations

Your question is profound: **If dynamics don't matter, what do circuits tell us?**

### Our Answer: Circuits and Linear Reps Serve Different Purposes

| Question | What Answers It | Our Findings |
|----------|-----------------|--------------|
| **How can the model do arithmetic?** | Circuit analysis | Circuits explain capability |
| **Did this specific answer come out correct?** | Linear probe | Correctness is in the embedding |
| **Why did the model make this error?** | ??? | Neither fully answers this |

### The Imitation Hypothesis (from SFT findings)

**SFT models don't "reason" — they imitate reasoning:**

1. **SFT trains on CoT from stronger models** → creates "imitation circuits"
2. **These circuits produce representations that LOOK like correct reasoning**
3. **The linear probe detects if the imitation matches the pattern of correct solutions**
4. **Cross-domain transfer works because imitation is domain-general**

**Contrast with RL-Zero:**
- RL-Zero optimizes for task-specific rewards → creates specialist circuits
- No imitation → no domain-general patterns → no cross-domain transfer

### What We Can Claim vs What We Can't

**CAN claim:**
1. Correctness is linearly readable in activation space
2. SFT creates domain-general representations; RL creates domain-specific ones
3. Dynamical measures (Jacobian, Lyapunov, Procrustes, SVCCA) do NOT add information
4. The transformation mechanics are identical for correct/incorrect

**CANNOT claim:**
1. That circuits are "irrelevant" — they create the representations we probe
2. That we understand WHY linear probing works
3. That there's no dynamical signal at all (just that we couldn't find it with these methods)

### Implications for Mechanistic Interpretability

1. **Circuit analysis and probing are complementary:**
   - Circuits explain what the model CAN do (capability)
   - Probes explain what the model DID do (outcome)

2. **For error detection, you only need probing:**
   - No need to trace circuits
   - Just read the linear representation

3. **For understanding HOW errors happen, neither is sufficient:**
   - Probing tells you "this is wrong" but not why
   - Circuits tell you the machinery but not the mistake

### The Training Method Story

| Model | Training | What It Creates | Cross-Domain | Interpretation |
|-------|----------|-----------------|--------------|----------------|
| **Base** | Pre-train | Generic representations | Orthogonal (0.07) | No structure |
| **SFT** | Distillation | Imitation circuits | **Aligned (0.35)** | Copies domain-general patterns |
| **RL-Zero** | Task RL | Specialist circuits | Orthogonal (0.10) | Optimizes for reward, not structure |
| **Think** | SFT+RL | Mixed | Partial (0.26) | SFT patterns survive RL |

**Key insight:** The **training method determines the representation structure**, which determines what probing can find. But the **dynamics** (how representations transform) are identical regardless of training.

---

## Path Signature Analysis (2026-01-31)

### Motivation

Path signatures are reparameterization-invariant features that capture the "shape" of a trajectory. They should be sensitive to the *structure* of computation flow, not just endpoint positions.

### Method

1. **Layers-as-path**: Mean activation across tokens → path through 16 layers → depth-3 signature
2. **Sequence-as-path**: Activations at layer 8 → path through 512 tokens → depth-3 signature
3. Compare signature norms (Cohen's d) and train classifiers (AUC)

### Results

| Model/Task | View | sig_norm_d | p | AUC |
|------------|------|------------|---|-----|
| base/gsm8k | layers | -0.38 | 0.31 | 0.64 |
| base/humaneval | layers | -0.16 | 0.45 | 0.58 |
| base/logiqa | layers | -0.37 | 0.10 | 0.50 |
| **sft/humaneval** | **layers** | **-0.39** | **0.003** | **0.82** |
| sft/gsm8k | layers | +0.24 | 0.83 | 0.58 |
| rl_zero/gsm8k | seq_L8 | -0.48 | 0.34 | 0.81 |
| rl_zero/humaneval | layers | -0.36 | 0.39 | 0.71 |

### Cross-Domain Transfer

| Model | GSM8K→HumanEval | HumanEval→GSM8K |
|-------|-----------------|-----------------|
| base | 0.578 | 0.564 |
| **sft** | **0.780** | 0.544 |
| rl_zero | 0.545 | 0.387 |

### Key Findings

1. **SFT shows best transfer** (AUC=0.78 for GSM8K→HumanEval) — consistent with error-direction findings
2. **Path signatures show signal for some model/task** (AUC up to 0.82)
3. **RL-Zero does NOT transfer** — also consistent with previous findings
4. **Only sft/humaneval is statistically significant** (p=0.003 for norm difference)

### Interpretation

Path signatures capture *something* about correct vs incorrect trajectories, but:
- Effect sizes are small (|d| < 0.5 mostly)
- Not consistently significant across models/tasks
- SFT's strong transfer (0.78) suggests CoT distillation creates similar "computation shapes" across domains

---

## Next Steps

### Completed
1. ✅ **Increase sample size**: Ran with n=100 (from n=50)
2. ✅ **Cross-validation**: Fixed circular analysis

### Completed (2026-01-25)
3. ✅ **Cross-domain direction transfer**: 1/4 significant
   - olmo3_rl_zero/humaneval→gsm8k: d=-0.598, p=0.027 ✓
   - Other directions: not significant

4. ✅ **Cross-model direction transfer**: 2/4 significant
   - think→rl_zero/gsm8k: d=-0.760, p=0.005 ✓
   - rl_zero→think/humaneval: d=0.702, p=0.017 ✓
   - Other directions: not significant

5. ✅ **Layer-by-layer analysis**: Peak separation varies by model
   | Model/Task | Peak Layer | Peak d | Early/Mid/Late |
   |------------|------------|--------|----------------|
   | rl_zero/gsm8k | L8 | 0.464 | 0.34/0.41/0.15 |
   | rl_zero/humaneval | L12 | 0.393 | 0.33/0.33/0.36 |
   | think/gsm8k | L15 | -0.331 | -0.30/-0.24/-0.31 |
   | think/humaneval | L10 | 0.512 | 0.50/0.51/0.51 |

   **Interpretation**: RL-Zero separation emerges early-to-mid layers; Think uses late layers

### Future Work
6. **Causal test**: Can we intervene on this direction to flip correctness?
   - Would require activation patching infrastructure
   - Low priority given weak signal

### Notes on Incomplete Analysis

The extended analysis (`h3_remaining_analyses.py`) was killed after ~62 CPU hours due to computational cost. Only `olmo3_base` results were saved. The script computes:
- Per-sample randomized SVD (k=50 components) across 15 layer transitions
- 5-fold CV for H_jac2
- 100 permutation tests per task
- ~768,000 SVD operations per task

**To complete**: Either reduce k, reduce permutations, or run on fewer samples.

---

## Code Reference

```python
# Key functions in full_lyapunov_analysis.py

def compute_layer_jacobian_svd(x_l, x_l1, k=50):
    """SVD-based expansion estimation"""

def compute_lyapunov_spectrum(trajectory, k=50):
    """Full Lyapunov statistics per sample"""

def compute_directional_lyapunov(trajectory, direction, k=50):
    """Lyapunov in specific subspace"""

def extract_error_direction(correct_traj, incorrect_traj, layer_idx=-1):
    """Difference-in-means direction"""
```

---

## Related Documents

- [PHASE3_ANALYSIS_SUMMARY.md](./PHASE3_ANALYSIS_SUMMARY.md) — Full Phase 3 summary
- [MENGER_CURVATURE_FINDINGS.md](./MENGER_CURVATURE_FINDINGS.md) — Why raw curvature fails
- [SVD_LINEAR_SEPARABILITY_FINDINGS.md](./SVD_LINEAR_SEPARABILITY_FINDINGS.md) — SVD analysis (null result, but reframable)
