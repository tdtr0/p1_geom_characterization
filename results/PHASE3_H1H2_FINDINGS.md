
# Phase 3 Findings: H1/H2 Hypothesis Testing

**Date**: 2026-01-28 (updated with belief tracking)
**Models**: olmo3_base, olmo3_sft, olmo3_rl_zero, olmo3_think
**Tasks**: GSM8K (math), HumanEval (code)
**Samples**: 150 per task per model
**Data**: `/data/thanhdo/trajectories_0shot/`

---

## Methodology

### Error-Detection Direction Analysis
- **Method**: Compute difference-in-means between correct and incorrect activations at each layer
- **Direction**: `d = mean(correct) - mean(incorrect)`, normalized
- **Classifier**: Project activations onto direction, threshold at median
- **Transfer test**: Train classifier on task A, test on task B

### Menger Curvature
- **Method**: Compute curvature at each layer transition using 3 consecutive points
- **Formula**: `κ = 4A / (|a||b||c|)` where A = triangle area, a,b,c = side lengths

### Statistical Tests
- Cohen's d for effect size
- t-test for significance (p < 0.05)

---

## Task Accuracy by Model

| Model | Training | GSM8K | HumanEval |
|-------|----------|-------|-----------|
| base | Pre-train | 12.6% | 3.8% |
| sft | SFT | **59.4%** | 4.8% |
| rl_zero | RL-Zero | 14.0% | **13.4%** |
| think | SFT+DPO+RLVR | 39.4% | 5.0% |

---

## H1: Within-Domain Classification

| Model | GSM8K Layer | GSM8K Acc | GSM8K d | HE Layer | HE Acc | HE d |
|-------|-------------|-----------|---------|----------|--------|------|
| base | 28 | **88.0%** | 2.17 | 24 | **70.7%** | 1.06 |
| sft | 26 | 63.3% | 0.79 | 26 | 60.0% | 0.69 |
| rl_zero | 26 | 84.0% | 1.80 | 24 | 60.0% | 0.74 |
| think | **2** | 81.3% | 1.83 | 26 | 56.0% | 0.55 |

**Observations**:
- Base has strongest separation despite lowest task accuracy
- Think localizes GSM8K at layer 2 (very early)
- SFT has weakest separation despite highest task accuracy

---

## H2: Cross-Domain Transfer

### Math ↔ Code (GSM8K ↔ HumanEval)

| Model | GSM8K → HE | HE → GSM8K | Pattern |
|-------|------------|------------|---------|
| base | **85.3%** | 10.0% | Asymmetric |
| sft | **66.0%** | **56.0%** | **Bidirectional** |
| rl_zero | 41.3% | 18.0% | None |
| think | 22.7% | 44.7% | None |

### Cross-Domain Subspace Alignment (2026-01-27)

**Method**: Compute error direction for each task, measure cosine similarity and cross-transfer effect.

**Results by Model (n=300 per task):**

| Model | Task Accuracy | Cosine Sim | GSM8K→LogiQA d | LogiQA→GSM8K d | Pattern |
|-------|---------------|------------|----------------|----------------|---------|
| **base** | 12%/23% | 0.069 | 0.08 (p=0.46) | 0.18 (p=0.20) | Orthogonal |
| **sft** | 60%/36% | **0.355** | **0.48 (p=0.0009)** | **0.45 (p=0.0003)** | **Strong** |
| **rl_zero** | 14%/30% | 0.098 | 0.12 (p=0.42) | 0.11 (p=0.83) | Orthogonal |
| **think** | 40%/34% | **0.258** | **0.33 (p=0.028)** | **0.21 (p=0.027)** | **Moderate** |

**Key Finding**: **SFT and Think show cross-domain alignment**, Base and RL-Zero are orthogonal.

**Alignment order**: SFT (0.355) > Think (0.258) > RL-Zero (0.098) ≈ Base (0.069)

**Interpretation**:
- **Base model**: Task-specific error directions (orthogonal, no transfer)
- **SFT model**: Learns generalizable error patterns that transfer across domains (strongest)
- **RL-Zero model**: Despite RL training, shows orthogonal patterns like base
- **Think model**: Shows moderate alignment (SFT component preserved through DPO+RLVR)
- **SFT distillation hypothesis supported**: The SFT component creates domain-general representations

### Token-Position Specificity (2026-01-25)

**Question**: Where in the sequence does the error signal peak?

| Task | Peak Position | Fraction | Signal Location | Early d | Middle d | Late d |
|------|---------------|----------|-----------------|---------|----------|--------|
| GSM8K | 50/512 | 10% | **Early** | 0.201 | 0.003 | 0.000 |
| LogiQA | 85/512 | 17% | **Early** | 0.220 | 0.211 | 0.000 |

**Key Finding**: Signal peaks in **early tokens** (10-17%), NOT in answer tokens.

**Interpretation**:
- If signal was in late tokens → detecting answer format
- Signal is in **early** tokens → detecting initial problem encoding/setup
- This suggests the error direction captures how the model *starts* processing, not how it *finishes*

### Implications for H2

1. **No Universal Reasoning Signature (for base/RL-Zero)**: Error directions are task-specific (cos≈0.07-0.10)
2. **SFT Creates Domain-General Patterns**: SFT training produces aligned error directions (cos=0.355)
3. **SFT Survives RL**: Think model (SFT+DPO+RLVR) preserves alignment (cos=0.258)
4. **Pure RL Fails to Generalize**: RL-Zero (cos=0.098) is as orthogonal as base (cos=0.069)
5. **Early Token Signal**: The discriminative signal is in problem setup, not answer format

---

## Menger Curvature

| Model | GSM8K d | GSM8K p | HE d | HE p |
|-------|---------|---------|------|------|
| base | 0.145 | 0.595 | 0.200 | 0.428 |
| **sft** | **0.531** | **0.002** | **0.751** | **0.002** |
| rl_zero | 0.162 | 0.448 | 0.272 | 0.103 |
| think | 0.045 | 0.784 | 0.326 | 0.159 |

**Only SFT shows significant curvature differences** (p < 0.01 for both tasks).

---

## Summary Table

| Property | Base | SFT | RL-Zero | Think |
|----------|------|-----|---------|-------|
| Task accuracy | Low | High | Medium | High |
| Within-domain separation | **Best** | Weak | Strong | Strong |
| Cross-domain transfer | None | **Strong** | None | **Moderate** |
| Curvature significant | No | **Yes** | No | No |
| Belief smoothness (correct) | — | Jumpy | **Smooth** | Jumpy |
| Belief probe transfer | — | **Best** | Moderate | Moderate |

### Cross-Domain Alignment Summary

| Model | Cosine Sim | Transfer Pattern | Interpretation |
|-------|------------|------------------|----------------|
| **base** | 0.069 | None | Task-specific (orthogonal) |
| **sft** | **0.355** | **Strong bidirectional** | Domain-general (strongest) |
| **rl_zero** | 0.098 | None | Task-specific (orthogonal) |
| **think** | **0.258** | **Moderate bidirectional** | Partial domain-general |

**Key Conclusion**: **SFT and Think show cross-domain alignment; Base and RL-Zero do not.**
- **Base & RL-Zero**: Orthogonal patterns (cos ≈ 0.07-0.10), no transfer
- **SFT**: Strongest alignment (cos=0.355, p<0.001)
- **Think**: Moderate alignment (cos=0.258, p<0.03) — SFT component preserved through DPO+RLVR
- Pure RL (RL-Zero) does NOT create domain-general patterns

---

## Discussion: Why Does SFT Transfer?
Validates h1 - nothing new. 
### The SFT Distillation Hypothesis

SFT models are typically trained on chain-of-thought data generated by larger reasoning models (e.g., GPT-4, Claude). This creates a form of **knowledge distillation** where:

1. **Higher-rank representations**: The SFT training signal comes from a more capable model that has richer, more general representations across domains

2. **Shared reasoning patterns**: CoT data from strong models encodes domain-general reasoning patterns (decomposition, verification, correction) rather than task-specific shortcuts

3. **Geometric implication**: SFT learns to traverse similar manifold regions regardless of task domain, producing bidirectional transfer

### RL-Zero: Domain Specialization

RL-Zero optimizes directly for task-specific rewards without intermediate supervision:

1. **Reward hacking**: The model finds domain-specific shortcuts that maximize reward without generalizable reasoning

2. **Low-rank task circuits**: Creates efficient but narrow pathways for each task type

3. **No transfer**: High within-domain accuracy (84%) but representations don't generalize

### Base Model: Emergent Asymmetry

The pretrained base model shows asymmetric transfer (math→code works, code→math fails):

1. **Math subsumes code**: Mathematical reasoning patterns may be more general, encompassing code-like logical patterns

2. **Code is specialized**: Code generation requires specific syntax/API knowledge not captured by math

### Think Model: Preserved SFT Alignment

Despite DPO+RLVR training on top of SFT, think model preserves **moderate cross-domain alignment**:

1. **Cosine similarity**: cos=0.258 (between SFT's 0.355 and base's 0.069)

2. **Bidirectional transfer**: Significant in both directions (p<0.03)

3. **Interpretation**: The SFT component's domain-general patterns survive additional RL training

4. **Contrast with RL-Zero**: Pure RL (cos=0.098) does NOT create alignment — the alignment in Think comes from SFT, not from RLVR 
---

## Implications

1. **For verification**: SFT models are better candidates for cross-domain verifiers
2. **For training**: RL-Zero creates specialists; SFT creates generalists
3. **For interpretability**: The geometric signature of "correct" depends on training method, not just correctness

---

## Belief State Tracking Analysis (2026-01-25)

### Motivation

Previous analyses used **static geometry** (mean activations). This analysis tracks **belief state evolution** per-clause within model generations.

**Key insight**: Token probability P(next_token | context) ≠ belief state P(task_success | understanding). We use activation-based correctness probes to track task-level belief.

### Methodology

1. **Clause detection**: Parse model outputs into clauses using sentence boundaries + reasoning markers ("So", "Therefore", "First", "Wait")
2. **Belief probe**: Train logistic regression on mean final-layer activations to predict correctness (cross-validated)
3. **Belief curve**: Apply probe retroactively to each clause boundary position
4. **Metrics**: Smoothness (1/total_variation), accumulation rate (mean delta), final belief

### Results: Opposite Patterns by Training Method

| Model | Probe AUC | Smoothness d | Direction | p-value |
|-------|-----------|--------------|-----------|---------|
| **rl_zero** | 0.66 | **+1.72** | Correct = SMOOTH | <1e-17 |
| think | 0.62 | **-1.05** | Correct = JUMPY | <1e-11 |
| sft | 0.57 | **-0.90** | Correct = JUMPY | <1e-9 |

**Key Finding**: RL-Zero shows expected pattern (smooth belief = correct), but Think/SFT show the **opposite** — correct samples have more belief fluctuations.

### Number of Clauses: Correct Answers are Shorter (Think/SFT)

| Model | n_clauses d | Correct mean | Incorrect mean | p-value |
|-------|-------------|--------------|----------------|---------|
| rl_zero | -0.04 | 30.2 | 30.3 | 0.83 |
| **think** | **-0.98** | 28.2 | 34.3 | <1e-10 |
| **sft** | **-0.61** | 25.8 | 30.6 | <1e-4 |

Think/SFT: Correct answers are more concise. Incorrect answers ramble with more clauses.

### Cross-Model Belief Transfer

```
Transfer Matrix (AUC):
              | rl_zero | think | sft   |
rl_zero_train |  1.00   | 0.61  | 0.62  |
think_train   |  0.57   | 1.00  | 0.68  |
sft_train     |  0.66   | 0.68  | 1.00  | ← Best transfer
```

**SFT probe transfers best** (0.66-0.68 AUC). This beats H2 static baseline (d~0.4 ≈ AUC 0.57).

### Interpretation

The opposite smoothness patterns reveal different belief dynamics by training paradigm:

| Training | Correct Samples | Incorrect Samples | Interpretation |
|----------|-----------------|-------------------|----------------|
| **RL-Zero** | Smooth, confident | Jumpy, uncertain | "Honest" uncertainty |
| **Think/SFT** | Jumpy (self-correction) | Smooth (confidently wrong) | "Performative" patterns |

**Possible explanations:**

1. **Think/SFT learn performative patterns**: Backtracking, verification phrases, self-correction behaviors that cause belief fluctuations even when correct

2. **Incorrect samples in Think/SFT are "confidently wrong"**: Smooth belief evolution because the model doesn't doubt itself — no self-correction attempts

3. **RL-Zero is more "honest"**: Smooth when actually correct, fluctuating when uncertain

### Connection to Previous Findings

1. **Wynroe L16-18 spike in SFT**: May be where "performative patterns" activate — explains belief jumps in correct samples

2. **RL-Zero distributed processing**: Explains smooth belief evolution (no localized circuit activation)

3. **Cross-model transfer works** (AUC 0.61-0.68): Better than H2 static geometry transfer (d~0.4), suggesting belief dynamics capture something additional

4. **SFT creates generalists** (from H2): Confirmed here — SFT probe transfers best across models

### Implications

1. **H_belief is TRUE but nuanced**: Training paradigm determines the *direction* of the smoothness effect

2. **H_style partially supported**: Think/SFT jumps may be style (performative patterns) rather than substance (reasoning progress)

3. **For verification**: Can't use smoothness naively — must account for training paradigm

4. **For interpretability**: "Correct = smooth" only applies to RL-Zero; Think/SFT show the opposite

---

## Cross-Model Alignment Analysis (2026-02-02)

### Motivation

Previous analyses measured cross-domain transfer **within** a single model (GSM8K error direction applied to LogiQA). But the key mechanistic question is: **What changes between base and RL-Zero that destroys transfer?** This requires comparing representations **across models** on the same inputs.

### Methodology

**Cross-model alignment** (same task, different models):
- Load activations for same prompts (seed=42) from base, SFT, RL-Zero, Think
- For each pair (base→SFT, base→RL-Zero, base→Think):
  - **CKA**: Rotation-invariant representation similarity
  - **Eigenvector diagonal dominance**: Do SVD eigenvectors map 1-to-1?
  - **Error direction cosine**: Does base's error direction align with target's?
  - **Probe transfer AUC**: Does base's correctness probe work on target?

**Cross-task within-model CKA** (different tasks, same model):
- Compare GSM8K vs HumanEval activations within each model
- Measure CKA (overall space similarity) and error direction cosine (correctness-specific similarity)

**Scripts**: `scripts/analysis/cross_model_alignment.py`, `scripts/analysis/cross_task_within_model_cka.py`

### Results: Cross-Model Alignment (Same Task, Different Models)

| Pair | Task | CKA | Error Dir Cos | Eigenvec Diag Dom | Probe Transfer AUC |
|------|------|-----|---------------|-------------------|--------------------|
| base→**rl_zero** | GSM8K | **0.995** | **0.67** | **0.90** | **0.83** |
| base→sft | GSM8K | 0.810 | 0.34 | 0.19 | 0.57 |
| base→think | GSM8K | 0.812 | 0.36 | 0.18 | 0.57 |
| base→**rl_zero** | HumanEval | **1.000** | **0.96** | **0.96** | **0.89** |
| base→sft | HumanEval | 0.991 | 0.71 | 0.39 | 0.84 |
| base→think | HumanEval | 0.991 | 0.69 | 0.38 | 0.84 |

**Key Finding**: RL-Zero is nearly **identical** to base (CKA ≈ 1.0, eigenvectors preserved 90-96%). SFT and Think **reshape** the space (eigenvector preservation only 18-39%).

### Results: Cross-Task Within-Model CKA (Different Tasks, Same Model)

| Model | Cross-Task CKA | Error Dir Cos (GSM8K↔HE) |
|-------|----------------|--------------------------|
| base | 0.022 | 0.337 |
| **sft** | 0.022 | **0.553** (most aligned) |
| **rl_zero** | 0.025 | **0.235** (most orthogonal) |
| think | 0.022 | 0.430 |

**Key Finding**: Overall cross-task CKA is identical (~0.02) for all models. But error direction alignment differs dramatically.

---

## Per-Sample CKA Analysis (2026-02-06)

### Motivation

Previous analyses focused on **static geometry** (mean activations) or **global alignments** (Procrustes on all samples). This analysis tracks **per-sample representational similarity** using CKA at each layer transition.

**Key insight**: CKA measures how well the token-token similarity pattern (Gram matrix) is preserved across layers. If correct solutions "work harder", they should have LOWER CKA (more transformation per layer).

### Methodology

1. **Per-sample CKA**: For each sample, compute CKA(X_l, X_{l+1}) using the Gram matrices K_l = X_l @ X_l.T
2. **Layer profile**: Compare CKA at each layer transition between correct/incorrect
3. **Gram matrix structure**: Analyze eigenspectrum and change magnitude at most discriminative layer

**Script**: `scripts/analysis/cka_deep_analysis.py`

### Results: Correct Solutions Have LOWER CKA

| Layer | Correct CKA | Incorrect CKA | Cohen's d | p-value | Sig |
|-------|-------------|---------------|-----------|---------|-----|
| L0 | 0.9628 | 0.9655 | -0.513 | 0.078 | . |
| L6 | 0.9885 | 0.9925 | -0.541 | 0.063 | . |
| **L7** | 0.9885 | 0.9912 | **-0.639** | **0.029** | * |
| L9 | 0.9930 | 0.9937 | -0.531 | 0.068 | . |

**Most discriminative layer**: L7 (d=-0.639, p=0.029)

**Effect direction**: Incorrect solutions have **HIGHER** CKA (more preserved token-token similarity)

### CKA Trajectory Shape

| Metric | Cohen's d | p-value | Interpretation |
|--------|-----------|---------|----------------|
| **Mean CKA** | -0.584 | **0.045** | Incorrect has higher mean CKA |
| CKA variance | 0.327 | 0.26 | Not significant |
| CKA slope | 0.078 | 0.79 | Not significant |
| Early-Late diff | -0.222 | 0.44 | Not significant |
| Min CKA | -0.264 | 0.36 | Not significant |

### Gram Matrix Structure at L7

| Metric | Correct | Incorrect | Interpretation |
|--------|---------|-----------|----------------|
| Effective rank | 39.3 | 43.2 | Incorrect uses more dimensions |
| Gram change magnitude | **0.137** | 0.122 | Correct changes 13% MORE |

### Interpretation

**Correct solutions are "working harder":**
- Lower CKA = token-token similarity patterns change MORE between layers
- Higher Gram change = representations undergo more transformation
- Correct solutions do ~13% more representational work per layer

**Incorrect solutions are "coasting":**
- Higher CKA = token relationships stay similar across layers
- Lower Gram change = minimal transformation
- Possibly stuck in pattern-matching mode without active computation

**Connection to Linear Probe Success**:
This explains why mean-pooled linear probes work (AUC 0.68-0.75): correct solutions have different representational **dynamics** that leave signatures in the mean activation. The CKA finding reveals the mechanism: correct solutions transform token relationships more aggressively.

### CKA as Classifier

| Feature | AUC | Notes |
|---------|-----|-------|
| Best single layer (L12) | 0.66 | Modest signal |
| All layers combined | 0.52 | Combining doesn't help |

**Conclusion**: CKA is informative but weaker than direct linear probes. The value is in understanding the **mechanism**, not as a standalone classifier.

---

### Mechanistic Interpretation

#### The Paradox Resolved

| What We Expected | What We Found |
|------------------|---------------|
| RL-Zero destroys base's features | RL-Zero **preserves** base's features (CKA ≈ 1.0) |
| SFT preserves base's features | SFT **reshapes** the space (eigenvec diag dom = 0.19) |

The cross-domain transfer difference is NOT about the overall representation space (CKA ~0.02 for all models). It is specifically about **where "error" points within that space**.

3. **Lyapunov Exponent Analysis**
   - Test stability hypothesis: correct solutions have more stable dynamics

4. **Attractor Analysis**
   - Cluster final layer states
   - Test if correct/incorrect occupy different attractor basins

5. **Feature Decomposition**
   - Which specific features (path signature vs curvature vs length) transfer best?
   - May find that *some* features transfer even if overall classifier doesn't

---

## Data Quality Notes

### Issues Encountered

1. **Class Imbalance**: HumanEval has only 6.3% correct (19/300 samples)
2. **Limited Data**: 0-shot LogiQA data only available for olmo3_base (other models' files were corrupted)
3. **Missing GSM8K**: olmo3_base/gsm8k was corrupted — cannot complete full 3-domain transfer matrix

### Data Integrity

- Correctness labels were fixed for HumanEval (extracted Python code from markdown blocks)
- All trajectories verified: shape (300, 512, 16, 4096)
- HDF5 files read successfully with `HDF5_USE_FILE_LOCKING=FALSE`

---

## Statistical Details

### Cross-Validation

All metrics computed with 5-fold stratified cross-validation with standard deviation reported.

### Success Criteria (from PHASE3_DETAILED_PLAN.md)

| Level | Criterion | Our Result |
|-------|-----------|------------|
| Strong (supports H2) | Mean accuracy > 60% across transfers | ✗ 51.9% |
| Weak (challenges H2) | Mean accuracy 52-58% | ✗ 51.9% |
| No transfer (falsifies H2) | Mean accuracy ≤52% | ✓ 51.9% |

**Conclusion**: Results fall into "No transfer" category (mean cross-domain accuracy = 51.9%).

---

---

## Phase 3 Dynamical Systems Analysis (COMPLETE Results)

**Date**: 2026-01-20 (Updated)
**Analysis**: `phase3_dynamical_analysis.py`
**Samples**: 100 per task
**Model**: olmo3_base (0-shot)

### 1. Error-Detection Direction Analysis (Wynroe-style)

**Method**: Extract linear direction separating correct/incorrect via difference-in-means, test across layers.

| Task | Best Layer | Effect Size (d) | p-value | Classification Accuracy |
|------|-----------|-----------------|---------|------------------------|
| HumanEval | 26 | **1.066** | 0.0006 | **68.0%** |
| LogiQA | 28 | **1.054** | <0.0001 | **71.0%** |

**Key Finding**: A single linear direction can distinguish correct from incorrect with good accuracy. The error-detection direction emerges in late layers (26-28 out of 32).

**Interpretation**: This replicates findings from Wynroe et al. — there exists a linear "error-detection direction" in the residual stream that separates correct from incorrect solutions. The effect is similar across both domains (d≈1.0).

### 2. Menger Curvature Analysis (Zhou et al., 2025)

**Method**: Compute Menger curvature at each layer transition, compare profiles between correct/incorrect.

| Task | Correct Mean | Incorrect Mean | Effect Size | p-value |
|------|--------------|----------------|-------------|---------|
| HumanEval | 2.383 | 2.111 | 0.315 | 0.291 |
| LogiQA | 1.266 | 1.196 | 0.244 | 0.322 |

**Within-Domain**: Not significant (p > 0.2). Correct solutions have slightly higher curvature, but the difference is not statistically reliable with N=100.

**Cross-Domain Correlation**:
| Comparison | Pearson r | p-value |
|------------|-----------|---------|
| HumanEval ↔ LogiQA | **0.996** | **<0.0001** |

**Cross-Domain Correlation**: r=0.996 - but this is a **NULL RESULT** (see below)!

**CRITICAL CORRECTION** (from follow-up analysis):

We tested curvature profiles conditioned on correctness:
| Comparison | r |
|------------|---|
| HumanEval Correct ↔ Incorrect | 0.9999 |
| LogiQA Correct ↔ Incorrect | 0.9998 |
| Cross-domain (any) | 0.996 |

**All correlations are r ≈ 1.0!** This means curvature profile is:
- Identical whether correct or incorrect
- Identical across domains
- **An architectural property, NOT related to reasoning**

The r=0.996 cross-domain finding is a **red herring** - it tells us about transformer architecture, not about reasoning transfer.

### 3. Lyapunov Exponent Analysis

**Method**: Compute Frobenius norm ratio at each layer transition as proxy for trajectory expansion/contraction.

| Task | Correct Mean | Incorrect Mean | Effect Size | p-value |
|------|--------------|----------------|-------------|---------|
| HumanEval | 0.186 | 0.189 | **-0.303** | 0.311 |
| LogiQA | 0.190 | 0.191 | **-0.267** | 0.280 |

**Key Finding**: The effect direction is as hypothesized — correct solutions have LOWER expansion (more stable trajectories). However, the effect is not statistically significant (p > 0.3).

**Interpretation**: Weak support for H5 (stability hypothesis). The trend is in the expected direction but more samples or more sensitive methods may be needed for significance.

### 4. Attractor Analysis

**Method**: Cluster final layer states using K-means (k=8), analyze cluster composition.

| Task | Mean Purity | Correct-Dominated Clusters | Incorrect-Dominated Clusters |
|------|-------------|---------------------------|------------------------------|
| HumanEval | 92.7% | 0 | 8 |
| LogiQA | 84.8% | 0 | 8 |

**Key Finding**: Clusters have high purity (solutions within clusters tend to be all-correct or all-incorrect), but no correct-dominated clusters exist. This is due to severe class imbalance — with only 13% correct samples, even random assignment would produce few correct-dominated clusters.

### 5. Error Direction Transfer (H2 Detailed Test)

**Method**: Train linear classifier on error-detection direction from one domain, test on another.

| Train → Test | Train Accuracy | Test Accuracy | Status |
|--------------|----------------|---------------|--------|
| HumanEval → LogiQA | 52.0% | **75.0%** | ✓ Transfer |
| LogiQA → HumanEval | 67.0% | **19.0%** | ✗ No transfer |

**Critical Finding**: Direction transfer is **ASYMMETRIC**!
- Code → Logic: **Works** (75% accuracy)
- Logic → Code: **Fails** (19% accuracy, worse than chance)

**Interpretation**: The error-detection direction learned from code generation (HumanEval) generalizes to logic reasoning (LogiQA), but NOT vice versa. This suggests:
1. Code generation may require more structured reasoning that encompasses logic-like patterns
2. Logic reasoning may use more task-specific representations that don't generalize to code
3. There may be a **hierarchy** of reasoning complexity: code ⊃ logic

### 6. Summary of Dynamical Findings

| Analysis | Within-Domain | Cross-Domain |
|----------|---------------|--------------|
| Error Direction | ✓ Strong (d>1.0, p<0.001) | **Asymmetric**: code→logic works, logic→code fails |
| Menger Curvature | ✗ Weak (d~0.3, p>0.2) | ✗ NULL RESULT (architectural, r≈1.0 for everything) |
| Lyapunov | ~ Trend in expected direction | N/A |
| Attractor | High purity (85-93%) | Class imbalance confound |

### 7. Revised Interpretation of H2

The original H2 hypothesis ("dynamical signatures share structure across domains") is **mostly FALSE** with one interesting exception:

**What transfers**:
1. ~~Geometric structure (curvature profile)~~ - **CORRECTED**: This is a null result (architectural property)
2. **Error direction from code to logic** — 75% transfer accuracy (ASYMMETRIC)

**What doesn't transfer**:
1. **Error direction from logic to code** — 19% accuracy (worse than chance)
2. **Generic linear classifiers** — ~52% cross-domain accuracy
3. **Curvature profile** — Not useful (identical for correct/incorrect)

**The one interesting finding**: Asymmetric direction transfer

```
Code generation (HumanEval) ──75%──> Logic reasoning (LogiQA)
Logic reasoning (LogiQA) ──19%──> Code generation (HumanEval)
```

**Proposed hierarchy**:
```
Code generation ⊃ Logic reasoning
```

This suggests code generation requires reasoning patterns that **encompass** logic-like patterns, but logic reasoning uses **domain-specific** patterns.

**What we learned about Menger curvature**: It's an architectural property of transformers, not a signal for correctness or reasoning. All trajectories through OLMo-3 have nearly identical curvature profiles (r≈1.0) regardless of task or correctness.

---

## Files Generated

- `results/phase3_0shot_olmo3_base.json`: Raw results from `h1_h2_classifier.py`
- `results/PHASE3_H1H2_FINDINGS.md`: This document

---

## Appendix: Raw Output

```
error_dir = mean(incorrect_activations) - mean(correct_activations)
```

This direction separates correct from incorrect in activation space. Each task (GSM8K, HumanEval) has its own error direction. The question is: do these directions align across tasks?

| Training | Effect | Error Dir Cos | Transfer |
|----------|--------|---------------|----------|
| **RL-Zero** | Specializes error direction per task | 0.235 (↓ from base) | Worse |
| **Base** | Weakly shared error concept | 0.337 | Asymmetric |
| **Think** | Partially generalizes | 0.430 (↑ from base) | Moderate |
| **SFT** | Generalizes error direction across tasks | 0.553 (↑↑ from base) | Bidirectional |

**RL-Zero makes error directions MORE orthogonal** (0.235 vs base's 0.337). It "sharpens" what counts as an error into task-specific patterns: what makes a math answer wrong becomes different from what makes a code answer wrong.

**SFT makes error directions MORE aligned** (0.553 vs base's 0.337). It creates a shared "wrongness" concept: similar patterns indicate errors regardless of domain.

#### Implications

1. **The mechanism is direction rotation, not feature destruction**: The overall space is preserved, but the error-relevant direction within that space is rotated
2. **RL-Zero is too faithful to base**: It preserves base's task-specific patterns rather than creating new cross-domain structure
3. **SFT adds cross-domain structure**: The CoT training signal creates domain-general error patterns that didn't exist in base
4. **Potential regularization**: During RL training, constraining error direction alignment (`L = 1 - cos(e_task1, e_task2)`) could preserve cross-domain transfer while still improving task performance

### Scripts and Data

**Analysis scripts**:
- `scripts/analysis/cross_model_alignment.py` — Cross-model Procrustes, CKA, eigenvector correspondence, error direction rotation test, and cross-model probe transfer. Compares base→SFT, base→RL-Zero, base→Think on same inputs.
- `scripts/analysis/cross_task_within_model_cka.py` — Cross-task CKA and error direction cosine within each model (GSM8K vs HumanEval).

**Output files**:
- `results/cross_model_alignment/cross_model_alignment_summary.csv` — Full metrics for all model pairs and tasks
- `results/cross_model_alignment/eigenvector_correspondence.npz` — Eigenvector correspondence matrices (50×50) for visualization

**Data used**:
- `/data/thanhdo/trajectories_0shot/` — 0-shot trajectories, 200 samples per model/task
- All models used identical prompts (seed=42), enabling matched cross-model comparison
- Last layer activations, mean-pooled over sequence (512 tokens → 4096-dim vector per sample)

**Methods applied**:
1. **CKA** (Kornblith et al., 2019): Linear centered kernel alignment, rotation-invariant similarity
2. **Procrustes alignment** (scipy.linalg.orthogonal_procrustes): Optimal rotation R and scale to align source→target, plus Frobenius residual
3. **Eigenvector correspondence**: Randomized SVD (k=50), correspondence matrix C = |V_source @ V_target.T|, diagonal dominance and row entropy
4. **Error direction**: Difference-in-means (incorrect − correct), normalized, cosine similarity across tasks/models
5. **Probe transfer**: Logistic regression trained on source, evaluated on target (with and without Procrustes rotation)
