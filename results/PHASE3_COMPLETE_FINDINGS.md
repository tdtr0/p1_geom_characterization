# Phase 3 Complete Findings: What We Tested, What We Found, and What It Means

**Date**: 2026-02-03
**Project**: ManiVer (Manifold Verification)

---

## Preamble: Data and Methods

All analyses in this document are performed on pre-collected activation trajectories from the OLMo-3 7B model family.

**Models**:
| Key | Model | Training Method | Size |
|-----|-------|-----------------|------|
| base | allenai/OLMo-3-1025-7B | Pre-training | 7B |
| sft | allenai/OLMo-3-7B-Think-SFT | Supervised fine-tuning on CoT | 7B |
| rl_zero | allenai/OLMo-3-7B-RL-Zero-General | RL from scratch (no SFT) | 7B |
| think | allenai/OLMo-3-7B-Think | SFT + DPO + RLVR | 7B |

**Tasks**:
| Task | Domain | Samples | Correctness Criterion |
|------|--------|---------|----------------------|
| GSM8K | Math | 500 | Exact numerical match (#### format) |
| HumanEval | Code | 164-500 | Syntax check (not execution) |
| LogiQA | Logic | 500 | Exact letter match (A/B/C/D) |

**Trajectory format**: `(n_samples, 512 tokens, 16 layers, 4096 dims)` — 16 even layers (0, 2, 4, ..., 30), sequence length capped at 512 tokens, all prompts generated with seed=42.

**Task accuracy by model**:
| Model | GSM8K | HumanEval | LogiQA |
|-------|-------|-----------|--------|
| base | 12.6% | 3.8% | 25% |
| sft | 59.4% | 4.8% | 36% |
| rl_zero | 14.0% | 13.4% | 30% |
| think | 39.4% | 5.0% | 34% |

**Note on sample imbalance**: For base/GSM8K, only ~13% of samples are correct (63 out of 500). For base/HumanEval, only ~4% (19 out of 500). Several analyses use subsets of 50-200 samples, which can reduce the number of correct samples to as few as 5. We note sample sizes for each finding.

---

## Positive Findings

### P1. Linear Probing Separates Correct from Incorrect Activations

**What we measured**: Whether a logistic regression classifier on mean-pooled last-layer activations can distinguish correct from incorrect model outputs.

**Method**: Difference-in-means direction at each layer, project activations onto direction, threshold at median. (`scripts/analysis/phase3_dynamical_analysis.py`)

**What we found** (n=150 per task per model):

| Model | GSM8K Acc | GSM8K d | HumanEval Acc | HumanEval d |
|-------|-----------|---------|---------------|-------------|
| base | 88.0% | 2.17 | 70.7% | 1.06 |
| sft | 63.3% | 0.79 | 60.0% | 0.69 |
| rl_zero | 84.0% | 1.80 | 60.0% | 0.74 |
| think | 81.3% | 1.83 | 56.0% | 0.55 |

**Scope of claim**: On mean-pooled OLMo-3 7B activations at the last collected layer (layer 30), correct and incorrect outputs occupy distinguishable regions of activation space, detectable by a linear classifier with Cohen's d between 0.55 and 2.17.

**Note**: The base model shows the strongest separation (d=2.17) despite the lowest task accuracy (12.6%). This may reflect that when a low-accuracy model gets something right, those samples are more geometrically distinct from the incorrect majority.

---

### P2. SFT Training Creates Domain-General Error Directions

**What we measured**: Whether the direction separating correct from incorrect activations (the "error direction") is aligned across different tasks within the same model.

**Method**: Compute difference-in-means direction per task, measure cosine similarity. Cross-validated to avoid circular analysis. (`scripts/analysis/cross_domain_all_models.py`)

**What we found** (n=300 per task per model, GSM8K vs LogiQA):

| Model | Cosine Similarity | GSM8K→LogiQA d (p) | LogiQA→GSM8K d (p) |
|-------|-------------------|---------------------|---------------------|
| base | 0.069 | 0.08 (0.46) | 0.18 (0.20) |
| sft | **0.355** | **0.48 (0.0009)** | **0.45 (0.0003)** |
| rl_zero | 0.098 | 0.12 (0.42) | 0.11 (0.83) |
| think | **0.258** | **0.33 (0.028)** | **0.21 (0.027)** |

**Scope of claim**: In OLMo-3 SFT and Think models, the direction separating correct from incorrect activations shows moderate alignment across math and logic domains (cos=0.36 and 0.26 respectively), with statistically significant bidirectional transfer (all p < 0.03). In base and RL-Zero models, error directions are near-orthogonal across tasks (cos < 0.10).

**Note**: SFT was trained on chain-of-thought data from stronger models. The cross-domain alignment may reflect patterns inherited from the training data rather than a property of the model architecture.

---

### P3. Error Signal Peaks in Early Tokens

**What we measured**: At which token positions does the correctness-discriminating signal peak?

**Method**: Compute Cohen's d at each token position along the error direction. (`scripts/analysis/additional_analyses.py`)

**What we found** (base model):

| Task | Peak Position | Fraction | Early d | Middle d | Late d |
|------|---------------|----------|---------|----------|--------|
| GSM8K | 50/512 | 10% | 0.201 | 0.003 | 0.000 |
| LogiQA | 85/512 | 17% | 0.220 | 0.211 | 0.000 |

**Scope of claim**: In the base model, the activation difference between correct and incorrect outputs is concentrated in the first 10-17% of the token sequence, corresponding to the problem statement and initial setup, not the answer tokens.

**Note**: This was measured on the base model only. SFT/RL-Zero/Think may show different positional profiles. The observation is consistent with the model's initial encoding of the problem determining the trajectory toward a correct or incorrect solution.

---

### P4. SFT Shows Significant Menger Curvature Differences

**What we measured**: Whether the curvature of the mean layer trajectory (averaged over tokens) differs between correct and incorrect outputs.

**Method**: Menger curvature κ = 4A/(|a||b||c|) on consecutive layer triplets. (`scripts/analysis/phase3_dynamical_analysis.py`)

**What we found** (n=150 per task):

| Model | GSM8K d | GSM8K p | HumanEval d | HumanEval p |
|-------|---------|---------|-------------|-------------|
| base | 0.145 | 0.595 | 0.200 | 0.428 |
| **sft** | **0.531** | **0.002** | **0.751** | **0.002** |
| rl_zero | 0.162 | 0.448 | 0.272 | 0.103 |
| think | 0.045 | 0.784 | 0.326 | 0.159 |

**Scope of claim**: Only in the SFT model do correct and incorrect outputs show significantly different layer-trajectory curvature (d=0.53-0.75, p=0.002 for both tasks). All other models show no significant difference.

**Caveat**: When comparing curvature *profiles* (shape across layers, not magnitude), correct and incorrect are near-identical (r ≈ 0.9999). The SFT effect is in the *magnitude* of curvature, not its shape.

---

### P5. Path Signatures Transfer Cross-Domain for SFT

**What we measured**: Whether reparameterization-invariant path signatures of layer trajectories distinguish correct from incorrect, and whether this transfers across tasks.

**Method**: PCA to 32 dims, compute depth-3 path signatures using `signatory`, train logistic regression. (`scripts/analysis/path_signature_analysis.py`)

**What we found** (n=50 per task):

| Model/Task | sig_norm d | p | AUC |
|------------|------------|---|-----|
| sft/humaneval | -0.39 | **0.003** | **0.82** |
| rl_zero/humaneval | -0.36 | 0.39 | 0.71 |
| base/gsm8k | -0.38 | 0.31 | 0.64 |
| (most others) | — | >0.10 | <0.65 |

**Cross-domain transfer**:
| Model | GSM8K→HumanEval | HumanEval→GSM8K |
|-------|-----------------|-----------------|
| sft | **0.780** | 0.544 |
| base | 0.578 | 0.564 |
| rl_zero | 0.545 | 0.387 |

**Scope of claim**: For the SFT model on HumanEval, path signatures distinguish correct from incorrect (p=0.003, AUC=0.82), and a classifier trained on GSM8K signatures transfers to HumanEval (AUC=0.78). This finding is specific to SFT; other models show weaker or null results.

**Caveat**: Sample size is small (n=50). The SFT/HumanEval result should be replicated with more samples.

---

### P6. OLMo-3 Has Distributed Error Processing (Wynroe Replication)

**What we measured**: Whether error-correction in OLMo-3-Think is localized to specific layers (as in DeepSeek-R1) or distributed.

**Method**: Activation patching following Wynroe et al. methodology. Inject arithmetic errors into MATH traces, patch each layer with clean activations, measure logit-diff recovery for correction tokens. (`experiments/aha_moment/replicate_wynroe_patching.py`)

**What we found** (n=50 pairs, think model, MATH/algebra):

| Layer | Recovery % |
|-------|------------|
| 0 | 100.0 ± 0.0 |
| 10 | 100.0 ± 0.0 |
| 20 | 100.0 ± 0.0 |
| 30 | 100.0 ± 0.0 |

**Comparison**: Wynroe et al. found a localized spike at layer 16-18 (~70% recovery) in DeepSeek-R1-Distill-Llama-8B.

**Scope of claim**: In OLMo-3-Think, patching any single layer with clean activations fully recovers error-correction behavior. This suggests distributed error processing, in contrast to DeepSeek-R1's localized critical layer. This was tested only on the think model with MATH/algebra problems. Base, SFT, and RL-Zero were not tested.

**Note**: 100% recovery at all layers is surprising. It may indicate that the patching setup (clean vs corrupted traces) is providing enough information at every layer to recover, or that OLMo-3's error-correction redundancy is genuinely higher than DeepSeek-R1's.

---

### P7. Belief Smoothness Reverses by Training Method

**What we measured**: Whether the smoothness of the model's internal "belief" (probe-estimated P(correct)) across reasoning clauses differs between correct and incorrect outputs, and whether this pattern depends on training method.

**Method**: Train cross-validated logistic regression probe on final-layer mean activations, apply to clause boundary positions, compute smoothness as 1/total_variation. Clause boundaries detected via regex (sentence terminators, reasoning markers, newlines; min 20 chars). (`experiments/belief_tracking/analyze_belief_dynamics.py`)

**What we found** (n=200 per model, GSM8K):

| Model | n_correct | Probe AUC | Probe Smoothness d | Permutation p | Direction |
|-------|-----------|-----------|-------------------|---------------|-----------|
| base | 25 | 0.70 | **+0.54** | 0.013 | Correct = SMOOTH |
| rl_zero | 36 | 0.66 | **+1.72** | <0.001 | Correct = SMOOTH |
| sft | 119 | 0.57 | **-0.90** | <0.001 | Correct = JUMPY |
| think | 88 | 0.62 | **-1.05** | <0.001 | Correct = JUMPY |

**Probe-free validation** (clause-to-clause cosine distance of activations at final layer — no probe needed):

| Model | Act Cosine Dist d | p | Act Total Var d | p | Act Max Jump d | p |
|-------|-------------------|---|-----------------|---|----------------|---|
| base | -0.008 | 0.971 | -0.108 | 0.614 | 0.323 | 0.132 |
| rl_zero | -0.043 | 0.816 | -0.068 | 0.710 | 0.243 | 0.188 |
| sft | 0.163 | 0.259 | -0.445 | 0.002 | 0.125 | 0.385 |
| think | **0.612** | **<0.001** | 0.018 | 0.899 | **0.331** | **0.021** |

**Interpretation of probe-free validation**:
- **Think**: Both probe and activations agree — correct solutions have jumpier representations (activation cosine d=+0.61, probe smoothness d=-1.05). This is real signal.
- **RL-Zero**: Probe shows strong smoothness (d=+1.72) but activations show no difference (d=-0.04). This suggests the probe outputs near-constant values for RL-Zero, which inflates the smoothness metric. The probe-based smoothness may be an artifact of probe weakness (AUC only 0.66), not genuine representational smoothness.
- **SFT**: Probe shows jumpiness (d=-0.90) but activation cosine shows no signal (d=0.16). However, SFT activation total variation IS significant (d=-0.45, p=0.002) — incorrect SFT outputs have higher total activation variation.
- **Base**: Probe shows mild smoothness (d=+0.54) with no activation signal. Base has only 25 correct samples, limiting power.

**Critical control — Base model comparison**:
Base model (no task-specific training) shows mild correct=smooth (d=+0.54), similar direction to RL-Zero but weaker. This suggests smoothness may be closer to a default state, and SFT introduces jumpiness via CoT training. RL-Zero preserves or amplifies the base pattern.

**Cross-model transfer** (AUC when training probe on one model, testing on another):
| | rl_zero | think | sft |
|---|---------|-------|-----|
| rl_zero probe | — | 0.61 | 0.62 |
| think probe | 0.57 | — | 0.68 |
| sft probe | 0.66 | 0.68 | — |

**Revised scope of claim**: The probe-based smoothness effect is strongest for RL-Zero (d=+1.72) but is NOT confirmed by probe-free activation analysis. For Think, BOTH probe and activation measures agree (d=-1.05 and d=+0.61 respectively), providing converging evidence that Think's correct solutions involve larger representational changes between clauses. SFT shows partial activation support (total variation, d=-0.45). The most robust finding is the **direction reversal**: base/RL-Zero correct=smooth vs SFT/Think correct=jumpy, consistent across probe measures and partially validated by activations for Think.

**Note**: Think and SFT are trained on CoT data that includes backtracking and self-correction phrases. The "jumpy correct" pattern may reflect performative verification behavior (phrases like "Wait, let me check...") that causes genuine representational shifts. RL-Zero, lacking this training signal, shows a pattern closer to the untrained base model.

**Additional observation**: Think and SFT correct outputs use fewer clauses (d=-0.98, -0.61). Incorrect outputs ramble with more clauses.

**Future work**: Multi-pass generation (multiple completions per prompt with different seeds) would test whether belief smoothness is stable across generations. This avoids the bias of few-shot prompting while providing variance estimates.

---

### P8. RL-Zero Preserves Base Representation; SFT Reshapes It

**What we measured**: How much each training method changes the representation space relative to the base model, using same-prompt activations.

**Method**: CKA, eigenvector correspondence (SVD k=50), Procrustes alignment, error direction cosine, and probe transfer AUC. (`scripts/analysis/cross_model_alignment.py`)

**What we found** (n=200 per model/task):

| Pair | Task | CKA | Eigenvec Diag Dom | Error Dir Cos | Probe Transfer AUC |
|------|------|-----|-------------------|---------------|---------------------|
| base→rl_zero | GSM8K | **0.995** | **0.90** | **0.67** | **0.83** |
| base→sft | GSM8K | 0.810 | 0.19 | 0.34 | 0.57 |
| base→think | GSM8K | 0.812 | 0.18 | 0.36 | 0.57 |
| base→rl_zero | HumanEval | **1.000** | **0.96** | **0.96** | **0.89** |
| base→sft | HumanEval | 0.991 | 0.39 | 0.71 | 0.84 |
| base→think | HumanEval | 0.991 | 0.38 | 0.69 | 0.84 |

**Scope of claim**: RL-Zero preserves the base model's representation to a high degree (CKA ≈ 1.0, eigenvector diagonal dominance 0.90-0.96). SFT and Think substantially reshape the representation (eigenvector dominance 0.18-0.39). Probes trained on the base model transfer well to RL-Zero (AUC 0.83-0.89) but poorly to SFT/Think on GSM8K (AUC 0.57).

---

### P9. SFT Aligns Error Directions Across Tasks; RL-Zero Makes Them More Orthogonal

**What we measured**: How similar are error directions between GSM8K and HumanEval within each model?

**Method**: Cross-task CKA and error direction cosine similarity. (`scripts/analysis/cross_task_within_model_cka.py`)

**What we found** (n=200 per model/task):

| Model | Cross-Task CKA | Error Dir Cos (GSM8K↔HE) |
|-------|----------------|--------------------------|
| base | 0.022 | 0.337 |
| sft | 0.022 | **0.553** |
| rl_zero | 0.025 | **0.235** |
| think | 0.022 | 0.430 |

**Scope of claim**: Overall cross-task CKA is near-identical across models (~0.02). The difference is in error-direction alignment: SFT increases it (0.553 vs base's 0.337), while RL-Zero decreases it (0.235). RL-Zero makes "what counts as an error" more task-specific; SFT makes it more domain-general.

**Note**: The overall representation space (CKA) is not changed by training — only the error-relevant direction within that space is rotated.

---

## Null and Negative Findings

### N1. True Empirical Jacobian Shows No Correctness Signal

**What we tested**: Whether the regression-based Jacobian SVD spectrum (solving X_{l+1} ≈ X_l @ J.T) differs between correct and incorrect outputs at each layer transition.

**Method**: Randomized SVD on centered mean-pooled activations, regularized pseudo-inverse, compute SVD of J. Cohen's d on max/mean singular values. (`scripts/analysis/empirical_jacobian_lyapunov.py`)

**What we observed** (n=50-100 per model/task):

| Model/Task | True Jacobian max d | p | Delta Proxy d |
|------------|---------------------|---|---------------|
| base/gsm8k | 0.284 | 0.395 | 0.406 |
| base/logiqa | -0.242 | 0.326 | -0.399 |
| sft/gsm8k | 0.239 | 0.238 | -0.226 |
| sft/logiqa | -0.035 | 0.865 | 0.311 |
| rl_zero/gsm8k | 0.152 | 0.568 | 0.732 |
| rl_zero/logiqa | -0.057 | 0.800 | 0.050 |

**What this rules out**: Under this method (regression-based Jacobian on mean-pooled activations, n=50-100 samples), there is no detectable difference in layer-transition sensitivity between correct and incorrect outputs.

**What this does NOT rule out**: (1) Token-level Jacobian differences (we used mean-pooled activations). (2) Autograd-based Jacobian computation (we used regression). (3) Differences in other model families or at larger sample sizes. (4) Differences measurable with different regularization or dimensionality.

**Methodological note**: We also observed that the previously reported delta-based proxy (SVD of X_{l+1} - X_l) inflates effect sizes relative to the true Jacobian by 1.4-8.9x. The delta measures displacement magnitude, not local sensitivity.

---

### N2. Delta-Based Lyapunov Exponents Are a Methodological Artifact

**What we tested**: Whether SVD(X_{l+1} - X_l) is a valid proxy for the Jacobian of the layer transition.

**Method**: Diagnostic comparison (`scripts/analysis/jacobian_diagnostic.py`): compute cos(X_l, X_{l+1}), SVD of the true regression Jacobian, and delta-based SVD side by side.

**What we observed**: cos(X_l, X_{l+1}) ≈ 0.10-0.12 across all layers and models. Mean SVD of the Jacobian ≈ 1.2-1.5 (near-isometric). Delta-based d=0.732 (rl_zero/gsm8k) vs true Jacobian d=0.152.

**What this rules out**: The delta-based SVD method is not a valid proxy for Jacobian sensitivity in transformers. The previously reported RL-Zero dynamical signal (d=-0.73) was an artifact of this invalid proxy.

**What this does NOT rule out**: The delta (displacement) itself may carry information about subspace transitions, but it should not be interpreted as Lyapunov-type dynamical sensitivity.

**Methodological note**: When consecutive layer representations have cosine similarity ≈ 0.1, the layer transition is approximately a rotation into a new subspace. Rotations have singular values ≈ 1, producing log(1) = 0 for any Lyapunov-type measure. This is a property of the OLMo-3 architecture, not necessarily of all transformers.

---

### N3. Directional Lyapunov (H_jac2) Was Inflated by Data Leakage

**What we tested**: Whether the expansion rate along the error-detection direction differs between correct and incorrect.

**Method**: Compute error direction (difference-in-means), project Jacobian SVD onto this direction, measure expansion. Initially computed without cross-validation (circular analysis), then corrected with 5-fold CV. (`scripts/analysis/full_lyapunov_analysis.py`)

**What we observed**:

| Model/Task | Without CV (d) | With 5-fold CV (d) | Inflation |
|------------|----------------|---------------------|-----------|
| base/gsm8k | 1.68 | 0.72 | 133% |
| base/logiqa | 0.79 | -0.03 | complete artifact |
| sft/gsm8k | -1.12 | 0.01 | complete artifact |
| rl_zero/gsm8k | 1.63 | 0.06 | complete artifact |

**Baseline comparison** (5-fold CV on HumanEval):
| Model | Linear Probe AUC | H_jac2 AUC |
|-------|------------------|------------|
| base | 0.679 | 0.413 |
| rl_zero | 0.752 | 0.303 |

H_jac2 AUC < 0.5 means it performs worse than random chance.

**What this rules out**: The directional Lyapunov measure does not add predictive value beyond a simple linear probe, under proper cross-validation.

**What this does NOT rule out**: Other directional measures on different subspaces (e.g., PCA-derived, or learned). The specific error-direction may not be the right subspace.

**Methodological lesson**: When computing data-driven directions and testing them on the same data, cross-validation is essential. Without it, effect sizes can be inflated by 45-100%.

---

### N4. Procrustes Alignment Shows Same Transformation for Correct and Incorrect

**What we tested**: Whether, after optimally aligning consecutive layers via Procrustes rotation, the reconstruction error differs between correct and incorrect tokens.

**Method**: PCA to 256 dims, fit Procrustes (unsupervised — no labels used), compute per-token normalized distance. (`scripts/analysis/alignment_aware_dynamics.py`)

**What we observed** (n=50 per model, preliminary — only base model completed):

| Model/Task | Procrustes mean d |
|------------|-------------------|
| base/gsm8k | 0.025 |
| base/humaneval | 0.022 |
| base/logiqa | 0.031 |

**What this rules out**: Under Procrustes alignment on PCA-reduced OLMo-3 base activations (n=50), there is no measurable difference in layer transformation quality between correct and incorrect tokens (d < 0.04).

**What this does NOT rule out**: (1) Differences in SFT or RL-Zero models (not yet completed). (2) Differences at the full d_model=4096 dimensionality. (3) Non-linear alignment methods. (4) Token-specific (not pooled) analysis.

---

### N5. SVCCA Shows Same Subspace Alignment for Correct and Incorrect

**What we tested**: Whether the canonical correlations between consecutive layer subspaces differ for correct vs incorrect tokens.

**Method**: SVD to 256 dims, CCA with 20 components (unsupervised), compute per-token alignment score. (`scripts/analysis/alignment_aware_dynamics.py`)

**What we observed** (n=50, base model):

| Model/Task | SVCCA mean d |
|------------|--------------|
| base/gsm8k | -0.020 |

**What this rules out**: Under SVCCA on PCA-reduced base activations (n=50), correct and incorrect tokens show the same canonical correlation structure between consecutive layers.

**What this does NOT rule out**: Same caveats as N4. Additionally, the CCA convergence warnings suggest the method may not have converged fully at 256 dimensions.

---

### N6. Menger Curvature on Layer Trajectories Is Architectural

**What we tested**: Whether the curvature *profile* (shape across layers) differs between correct and incorrect, or between tasks.

**Method**: Average activations across sequence at each layer, compute Menger curvature at each layer triplet, correlate profiles. (`scripts/analysis/phase3_dynamical_analysis.py`)

**What we observed**:

| Comparison | Correlation (r) |
|------------|-----------------|
| Correct ↔ Incorrect (same domain) | 0.9999 |
| Cross-domain (HumanEval ↔ LogiQA) | 0.996 |

**What this rules out**: The layer-to-layer curvature profile is determined by the transformer architecture, not by correctness or task content.

**What this does NOT rule out**: (1) Curvature *magnitude* differences (SFT does show significant magnitude differences — see P4). (2) Token-level curvature (not mean-pooled). (3) Higher-order geometric features.

**Note**: This is distinct from P4 (SFT curvature magnitude). The profile shape is the same for everyone; only SFT shows different magnitudes.

---

### N7. Sequence Flow Velocity Shows No Signal

**What we tested**: Whether token-to-token velocity (‖x_{t+1} - x_t‖) at the last layer differs between correct and incorrect outputs.

**Method**: Compute velocity at each token position, compare distributions. (`scripts/analysis/sequence_flow_analysis.py`)

**What we observed** (base model):

| Task | Correct | Incorrect | d | p |
|------|---------|-----------|---|---|
| HumanEval | 15.8 ± 5.7 | 18.5 ± 8.6 | -0.32 | 0.29 |
| LogiQA | 28.0 ± 5.1 | 30.2 ± 7.8 | -0.30 | 0.23 |

**What this rules out**: Under this method, token-to-token velocity at the last layer does not significantly distinguish correct from incorrect outputs (d ≈ -0.3, p > 0.2).

**What this does NOT rule out**: Velocity differences at earlier layers, in other models, or with larger sample sizes. The trend (d ≈ -0.3, correct slightly slower) might reach significance with more data.

---

### N8. Sequence Flow Curvature Is Architectural

**What we tested**: Whether Menger curvature along the token sequence (at the last layer) differs between correct and incorrect.

**Method**: Menger curvature at each token triplet, correlate profiles. (`scripts/analysis/sequence_flow_analysis.py`)

**What we observed**:

| Task | Profile Correlation (r) |
|------|-------------------------|
| HumanEval | 0.9769 |
| LogiQA | 0.9903 |

**What this rules out**: Token-sequence curvature profiles are highly similar between correct and incorrect (r > 0.97), suggesting architectural rather than semantic determination.

---

### N9. SVD Linear Separability Shows Opposite of Prediction

**What we tested**: Whether RLVR training changes top eigenvectors more than tail eigenvectors (which would indicate separable reasoning structure).

**Method**: SVD on base and RL-Zero activations at each layer, compute delta = 1 - |cos(v_base, v_rlzero)| for each eigenvector. (`experiments/svd_reasoning_separability/analyze_svd_delta.py`)

**What we observed**:

| Task | Top-10 Delta | Tail-50 Delta | Ratio |
|------|-------------|---------------|-------|
| HumanEval | 0.8% | 7.3% | 0.12 |
| GSM8K | 2.2% | 6.7% | 0.33 |

**What this rules out**: The prediction that RLVR creates separable reasoning structure in the top eigenvectors. Instead, tail eigenvectors change 3-8x more.

**What this does NOT rule out**: That RLVR changes computation in ways not captured by eigenvector analysis. The interpretation is that RLVR preserves the core representation structure (top eigenvectors) while refining fine-grained patterns (tail).

---

### N10. Pivot Velocity Does Not Predict Correction Success

**What we tested**: Whether the activation velocity at self-correction pivot tokens ("Wait", "But", "Actually") predicts whether the model's final answer is correct.

**Method**: Detect pivot tokens, measure velocity, correlate with final correctness. (`experiments/aha_moment/analyze_pivot_outcome_geometry.py`)

**What we observed** (n=200, think model, GSM8K):

| Analysis | Finding | p |
|----------|---------|---|
| Velocity → Correctness | No relationship | 0.55 |
| Low vs High velocity | 66.7% vs 71.4% correct | 0.93 |

Samples with self-corrections: 52.9% correct (9/17). Without: 71.0% correct (130/183).

**What this rules out**: Under this method, the velocity of activations at pivot tokens does not predict whether the model produces a correct final answer. Self-corrections are associated with lower accuracy, not higher.

**What this does NOT rule out**: That pivot tokens have a different function than velocity captures. The presence of self-corrections may indicate harder problems (confounded by difficulty).

---

### N11. RL-Zero Shows No Cross-Domain Transfer

**What we tested**: Whether RL-Zero's error direction transfers across tasks (same analysis as P2, but focusing on the negative result for RL-Zero specifically).

**What we observed**: RL-Zero error direction cosine similarity across tasks: 0.098 (vs base 0.069, SFT 0.355). Transfer effect sizes: d=0.12 (p=0.42) and d=0.11 (p=0.83).

**What this rules out**: RL-Zero training does not create domain-general error patterns. In fact, its error directions are as orthogonal as the base model's (cos ≈ 0.10 vs 0.07).

**What this does NOT rule out**: That RL-Zero creates other forms of cross-domain structure not captured by the error direction. Or that different RL training objectives (not zero-shot RL) might produce alignment.

---

## The Orthogonality Problem: Why Dynamical Measures Fail on These Activations

We observed that consecutive layer representations in OLMo-3 have cosine similarity ≈ 0.10. This has specific implications:

**Observation**: On mean-pooled OLMo-3 7B activations, cos(X_l, X_{l+1}) ≈ 0.10-0.12 across all layers and models (`scripts/analysis/jacobian_diagnostic.py`, n=50 per model/task).

**Consequence for Jacobian analysis**: When X_l and X_{l+1} occupy nearly orthogonal subspaces, the regression X_{l+1} ≈ X_l @ J.T fits a transformation whose singular values are near 1 (approximately a rotation). log(1) = 0, producing near-zero Lyapunov exponents for all trajectories regardless of correctness.

**Consequence for Procrustes/SVCCA**: Both methods confirmed that the transformation *quality* is the same for correct and incorrect — the same rotation, the same reconstruction error, the same canonical correlations.

**What this tells us**: Under these methods, on these activations, the layer-to-layer transformation mechanics do not carry correctness information. The information that distinguishes correct from incorrect is in the *content* of the representations (readable by linear probes), not in *how* the representations are transformed between layers.

**Scope**: This observation is specific to mean-pooled activations from OLMo-3 7B. Token-level analysis, gradient-based Jacobians (via autograd), other model architectures, or different pooling strategies may yield different results.

---

## The Training Method Story

Across all analyses, a coherent pattern emerges about how training method shapes representation geometry:

### SFT: The Domain-General Pattern Creator

| Analysis | SFT Finding |
|----------|-------------|
| Cross-domain error direction (P2) | cos=0.355 (strongest alignment) |
| Cross-task error direction (P9) | cos=0.553 (highest) |
| Menger curvature magnitude (P4) | Significant (d=0.53-0.75, p=0.002) |
| Path signature transfer (P5) | GSM8K→HumanEval AUC=0.78 |
| Belief smoothness (P7) | Correct=jumpy (performative patterns) |
| Cross-model probe transfer (P7) | Best transfer (AUC 0.66-0.68) |
| Representation change from base (P8) | Reshaped (eigenvec dom=0.19) |

SFT was trained on chain-of-thought data from stronger models. Across multiple independent analysis methods, it consistently shows the most domain-general patterns. This may reflect that CoT training data encodes shared reasoning patterns across domains.

### RL-Zero: The Domain-Specific Specialist

| Analysis | RL-Zero Finding |
|----------|-----------------|
| Cross-domain error direction (P2) | cos=0.098 (orthogonal) |
| Cross-task error direction (P9) | cos=0.235 (lowest) |
| Belief smoothness (P7) | Correct=smooth ("honest" uncertainty) |
| Representation change from base (P8) | Preserved (CKA=0.995, eigenvec dom=0.90) |

RL-Zero optimizes for task-specific rewards without intermediate supervision. It preserves the base model's representation space but rotates error directions to be more task-specific. Its belief dynamics are "honest" — smooth when correct, jumpy when uncertain.

### Think: The SFT Preserver

| Analysis | Think Finding |
|----------|--------------|
| Cross-domain error direction (P2) | cos=0.258 (moderate) |
| Cross-task error direction (P9) | cos=0.430 (between SFT and base) |
| Belief smoothness (P7) | Correct=jumpy (like SFT) |
| Representation change from base (P8) | Reshaped (like SFT) |

Think (SFT + DPO + RLVR) partially preserves SFT's domain-general patterns through additional RL training.

### Base: The Unstructured Starting Point

| Analysis | Base Finding |
|----------|-------------|
| Cross-domain error direction (P2) | cos=0.069 (no structure) |
| Cross-task error direction (P9) | cos=0.337 (moderate) |
| Linear probe (P1) | Strongest separation (d=2.17) |

Base has the strongest within-domain separation but no cross-domain structure. The high within-domain d may reflect that rare correct samples (12.6% accuracy) are geometrically very distinct.

---

## On Circuits and Linear Representations

Our analyses examined activations, not circuits directly. We can make the following observations:

**What we observed**: Under linear probing methods, correctness is readable from activation content (AUC 0.68-0.75). Under dynamical measures (Jacobian, Procrustes, SVCCA, Lyapunov), correctness is not detected in the transformation mechanics.

**What we can say**: On these activations and with these methods, the *content* of representations carries correctness information, but the *transformation* between layers does not. A linear probe on mean activations is sufficient for correctness detection; no dynamical analysis improves on it.

**What we cannot say**: That circuits are irrelevant. Circuits create the representations that we probe. Our null dynamical results mean that the layer-to-layer transformation — as measured by these specific methods on mean-pooled activations — is the same for correct and incorrect. This does not mean the computational pathway is identical; it means these particular summaries of it do not differ.

**What remains open**: Whether token-level dynamics, gradient-based analysis, or circuit-level investigation would reveal transformation differences that our activation-based methods missed.

---

## Limitations

1. **Model family**: All results are from OLMo-3 7B. Generalization to other architectures (Llama, GPT, Mistral) is untested.

2. **Prompting**: All trajectories use 0-shot prompting with seed=42. 8-shot data exists but was not analyzed in Phase 3. Different prompt formats may change results.

3. **Correctness labeling**: HumanEval uses syntax checking (does the code parse?), not execution against test cases. Some "correct" samples may not be functionally correct.

4. **Sample imbalance**: For low-accuracy models (base on GSM8K: 12.6%, base on HumanEval: 3.8%), the number of correct samples in subsets of 50-200 can be as low as 5-10. This limits statistical power and may inflate effect sizes.

5. **Mean pooling**: Most analyses use mean-pooled activations over 512 tokens. This discards positional information and may mask token-level dynamics.

6. **Single seed**: All prompts generated with seed=42. Results may be sensitive to prompt ordering or specific sample composition.

7. **Clause detection**: Belief tracking uses regex-based clause detection, which is heuristic and may misidentify clause boundaries.

8. **LogiQA data**: 0-shot LogiQA trajectories for SFT, RL-Zero, and Think were truncated/corrupted. LogiQA results are only available for the base model in most analyses, and for all models in the cross-domain alignment analysis (which used a separate data collection).

9. **Alignment-aware analysis incomplete**: Procrustes and SVCCA analyses were only completed for the base model (SFT and RL-Zero analyses were interrupted by server connectivity issues).

---

## Scripts Reference

| Analysis | Script | Output |
|----------|--------|--------|
| Error direction, curvature | `scripts/analysis/phase3_dynamical_analysis.py` | — |
| Cross-domain alignment | `scripts/analysis/cross_domain_all_models.py` | — |
| Token-position specificity | `scripts/analysis/additional_analyses.py` | — |
| Path signatures | `scripts/analysis/path_signature_analysis.py` | `results/h2_path_signatures.csv` |
| True Jacobian | `scripts/analysis/empirical_jacobian_lyapunov.py` | `results/h2_true_jacobian.csv` |
| Jacobian diagnostic | `scripts/analysis/jacobian_diagnostic.py` | — |
| Full Lyapunov | `scripts/analysis/full_lyapunov_analysis.py` | — |
| Procrustes + SVCCA | `scripts/analysis/alignment_aware_dynamics.py` | (partial) |
| Sequence flow | `scripts/analysis/sequence_flow_analysis.py` | — |
| SVD separability | `experiments/svd_reasoning_separability/analyze_svd_delta.py` | — |
| Wynroe patching | `experiments/aha_moment/replicate_wynroe_patching.py` | — |
| Pivot velocity | `experiments/aha_moment/analyze_pivot_outcome_geometry.py` | — |
| Belief tracking | `experiments/belief_tracking/analyze_belief_dynamics.py` | `experiments/belief_tracking/results/belief_dynamics_gsm8k.json` |
| Cross-model alignment | `scripts/analysis/cross_model_alignment.py` | (on eyecog) |
| Cross-task CKA | `scripts/analysis/cross_task_within_model_cka.py` | — |

---

## Related Documents

- [PHASE3_H1H2_FINDINGS.md](./PHASE3_H1H2_FINDINGS.md) — H1/H2 results with belief tracking and cross-model alignment
- [FULL_LYAPUNOV_FINDINGS.md](../notebooks/working_notes/FULL_LYAPUNOV_FINDINGS.md) — Detailed Lyapunov methodology, true Jacobian, alignment-aware analysis
- [PHASE3_ANALYSIS_SUMMARY.md](../notebooks/working_notes/PHASE3_ANALYSIS_SUMMARY.md) — Running analysis log
- [SVD_LINEAR_SEPARABILITY_FINDINGS.md](../notebooks/working_notes/SVD_LINEAR_SEPARABILITY_FINDINGS.md) — SVD analysis
- [MENGER_CURVATURE_FINDINGS.md](../notebooks/working_notes/MENGER_CURVATURE_FINDINGS.md) — Curvature analysis
