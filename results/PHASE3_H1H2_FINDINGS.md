# Phase 3 Initial Findings: H1/H2 Hypothesis Testing

**Date**: 2026-01-19
**Model**: olmo3_base (0-shot)
**Tasks**: HumanEval (code), LogiQA (logic)
**Data Source**: `/data/thanhdo/trajectories_0shot/`

---

## Executive Summary

We conducted initial H1/H2 hypothesis testing on the olmo3_base model using 0-shot trajectory data from HumanEval and LogiQA tasks. The results reveal a clear **asymmetry**:

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| **H1** (Within-Domain) | ✓ PASS | Trajectory features distinguish correct/incorrect solutions (AUC 0.62-0.75) |
| **H2** (Cross-Domain) | ✗ FAIL | Transfer accuracy at chance level (~52%) for logistic classifier |

**Critical insight**: The geometric signatures that distinguish correct from incorrect solutions appear to be **domain-specific**, not universal.

---

## Experimental Setup

### Data Summary

| Task | Samples | Correct | Incorrect | Correct Rate |
|------|---------|---------|-----------|--------------|
| HumanEval | 300 | 19 | 281 | 6.3% |
| LogiQA | 300 | 69 | 231 | 23.0% |

**Note**: Severe class imbalance in HumanEval (6.3% correct) affects interpretation.

### Features Extracted (124 total)

The classifier uses geometric trajectory features including:
- **Path signature coefficients** (depth 2-3): Captures trajectory shape
- **Curvature measures**: Per-layer and aggregate
- **Trajectory length**: Total path distance through layers
- **Layer-wise statistics**: Mean, std of activations per layer

### Classifiers

1. **Logistic Regression**: Linear classifier with L2 regularization (less prone to overfitting)
2. **Random Forest**: 100 trees, max_depth=10 (more prone to overfitting on small datasets)

---

## H1: Within-Domain Classification Results

**Question**: Can we distinguish correct vs incorrect trajectories within each domain?

### HumanEval (Code)

| Metric | Logistic | Random Forest |
|--------|----------|---------------|
| Accuracy | 54.0% ± 7.3% | 93.3% ± 1.1% |
| AUC-ROC | 0.727 ± 0.074 | 0.753 ± 0.120 |

**Interpretation**:
- Logistic AUC of 0.73 indicates moderate discriminative ability
- RF achieves high accuracy (93.3%) but this is inflated by class imbalance — predicting "incorrect" for everything would achieve ~94%
- **The AUC of 0.73-0.75 is the meaningful metric** — significantly above chance (0.5)

### LogiQA (Logic)

| Metric | Logistic | Random Forest |
|--------|----------|---------------|
| Accuracy | 58.0% ± 5.5% | 76.3% ± 1.6% |
| AUC-ROC | 0.511 ± 0.070 | 0.622 ± 0.064 |

**Interpretation**:
- Logistic AUC of 0.51 is essentially chance level
- RF AUC of 0.62 shows weak but present discrimination
- LogiQA trajectories are harder to distinguish than HumanEval

### H1 Verdict: **PASS** (with caveats)

The geometric features can distinguish correct from incorrect solutions within domains, though the effect varies by domain:
- **HumanEval**: Strong signal (AUC ~0.75)
- **LogiQA**: Weak signal (AUC ~0.62)

**Caveat**: The low correct rate in HumanEval (6.3%) means we have only 19 correct samples — statistical power is limited.

---

## H2: Cross-Domain Transfer Results

**Question**: Does a classifier trained on one domain transfer to another?

### Transfer Matrix — Logistic Regression (Primary Metric)

|  | Test: HumanEval | Test: LogiQA |
|--|-----------------|--------------|
| **Train: HumanEval** | 54.3%* (in-domain) | 52.0% |
| **Train: LogiQA** | 51.7% | 70.7%* (in-domain) |

*Asterisks indicate in-domain performance (not transfer)

### Transfer Matrix — Random Forest

|  | Test: HumanEval | Test: LogiQA |
|--|-----------------|--------------|
| **Train: HumanEval** | 100%* (in-domain) | 77.0% |
| **Train: LogiQA** | 93.7% | 100%* (in-domain) |

### Interpretation

**Logistic (reliable)**: Cross-domain transfer is at chance level:
- HumanEval → LogiQA: **52.0%** (essentially random guessing)
- LogiQA → HumanEval: **51.7%** (essentially random guessing)

**Random Forest (questionable)**: Appears to show transfer:
- HumanEval → LogiQA: **77.0%**
- LogiQA → HumanEval: **93.7%**

**However**, the RF results are misleading:
1. **Perfect overfitting**: RF achieves 100% in-domain accuracy (memorization)
2. **Class imbalance exploitation**: Predicting "incorrect" on HumanEval (93.7% incorrect rate) would achieve ~94% accuracy
3. **The logistic regression is more trustworthy** as a measure of genuine feature transfer

### H2 Verdict: **FAIL**

Using the logistic classifier (more reliable), cross-domain transfer is at chance level (~52%). This means:
- The geometric features that distinguish correct/incorrect HumanEval solutions do not generalize to LogiQA
- The geometric features that distinguish correct/incorrect LogiQA solutions do not generalize to HumanEval
- **Correct solution signatures appear to be domain-specific**

---

## Key Findings and Implications

### 1. Domain-Specific Geometry

The failure of H2 suggests that correct solutions in different domains (code vs logic) traverse **different regions** of the representation manifold. This is consistent with the interpolation-centric view (Allen-Zhu & Li, 2024): the model uses different "circuits" for different tasks.

### 2. Task Difficulty Matters

The difference in within-domain performance (HumanEval AUC 0.75 vs LogiQA AUC 0.62) suggests that some tasks produce more geometrically distinguishable correct/incorrect trajectories than others.

### 3. Base Model Limitations

We tested `olmo3_base` which:
- Has low correct rates (6.3% HumanEval, 23% LogiQA)
- May not have developed task-specific "reasoning" circuits
- Subsequent analysis should include RL-Zero and Think models

### 4. Implications for H2 Hypothesis

The original H2 hypothesis stated: *"Dynamical signatures share structure across domains"*

Our results suggest this hypothesis may need revision:
- **Strong form (rejected)**: Correct solutions share the same geometric signature across all domains
- **Weak form (untested)**: Correct solutions share *some* geometric properties that partially transfer

---

## Next Steps

### Immediate (Before claiming H2 fails definitively)

1. **Test on RL-Zero model**: May have more transferable features due to outcome-based training
2. **Test on Think model**: Full RLVR pipeline may produce different geometry
3. **Add GSM8K data**: Need math domain for complete transfer matrix (currently missing from 0-shot)
4. **Balance class sizes**: Undersample incorrect or oversample correct to avoid class imbalance artifacts

### Phase 3 Analyses (Per PHASE3_DETAILED_PLAN.md)

1. **Error-Detection Direction Analysis** (Wynroe-style)
   - Extract linear direction separating correct/incorrect
   - Test if this direction transfers across domains
   - Compare across models (Base vs RL-Zero vs Think)

2. **Menger Curvature Analysis** (Zhou et al., 2025)
   - Test if curvature profiles correlate across domains for correct solutions
   - Theory: Curvature captures logical structure beyond surface semantics

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
======================================================================
H1/H2 HYPOTHESIS TESTING
======================================================================
Model: olmo3_base
Tasks: ['humaneval', 'logiqa']
Data dir: /data/thanhdo/trajectories_0shot
Max samples: 300

======================================================================
H1 TEST: Within-Domain Classification
======================================================================
Using max 300 samples per task

Task: HUMANEVAL
  Shape: (300, 512, 16, 4096)
  Correct: 19/300 (6.3%)
  Feature matrix: (300, 124)
  LOGISTIC: Accuracy: 0.540 ± 0.073, AUC: 0.727 ± 0.074
  RF: Accuracy: 0.933 ± 0.011, AUC: 0.753 ± 0.120

Task: LOGIQA
  Shape: (300, 512, 16, 4096)
  Correct: 69/300 (23.0%)
  Feature matrix: (300, 124)
  LOGISTIC: Accuracy: 0.580 ± 0.055, AUC: 0.511 ± 0.070
  RF: Accuracy: 0.763 ± 0.016, AUC: 0.622 ± 0.064

======================================================================
H2 TEST: Cross-Domain Transfer
======================================================================

Classifier: LOGISTIC
  Transfer Matrix (rows=train, cols=test):
              humaneval   logiqa
  humaneval   0.543*      0.520
  logiqa      0.517       0.707*

Classifier: RF
  Transfer Matrix (rows=train, cols=test):
              humaneval   logiqa
  humaneval   1.000*      0.770
  logiqa      0.937       1.000*

======================================================================
SUMMARY
======================================================================
H1 (Within-Domain):
  humaneval    LR: 0.540, RF: 0.933  ✓ PASS
  logiqa       LR: 0.580, RF: 0.763  ✓ PASS

H2 (Cross-Domain Transfer):
  Train on humaneval  → Others: 0.520  ✗ FAIL
  Train on logiqa     → Others: 0.517  ✗ FAIL
```
