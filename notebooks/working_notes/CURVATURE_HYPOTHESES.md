# Hypotheses About Menger Curvature Findings

**Date**: 2026-01-20
**Context**: We found r=0.996 curvature profile correlation between HumanEval and LogiQA on olmo3_base

---

## The Puzzle

If the curvature profile correlation is ~1.0 between domains, what does this tell us?

**Two interpretations**:

1. **Strong (exciting)**: Correct solutions have a universal geometric structure that transfers
2. **Weak (concerning)**: Curvature profile is just an architectural property, not task-dependent

The weak interpretation seems more likely because:
- We computed curvature on ALL samples (correct + incorrect mixed)
- The r=0.996 is for the mean curvature profile, not conditioned on correctness
- Within-domain effect (correct vs incorrect) was NOT significant (d~0.3, p>0.2)

---

## Hypotheses to Test

### H_curv1: Curvature profile is architecture-dependent, not task-dependent

**Prediction**: All OLMo-3 models (base, sft, rl_zero, think) will have nearly identical curvature profiles (r > 0.95) regardless of training.

**Test**:
- Compute curvature profiles for same task across all 4 models
- If r > 0.95 across models → architecture property
- If r varies significantly → training affects curvature

**Implication**: If true, curvature profile is a "null feature" for our purposes.

---

### H_curv2: Curvature profile reflects layer specialization

**Prediction**: The curvature profile shape (high at certain layers, low at others) corresponds to layers with different "roles" (e.g., early=tokenization, middle=reasoning, late=output).

**Test**:
- Compare curvature peaks to known layer functions (from probing studies)
- Check if curvature correlates with attention entropy or MLP activation patterns

**Implication**: Would explain WHY architecture determines curvature.

---

### H_curv3: Curvature MAGNITUDE (not profile shape) differs by correctness

**Observation**: Correct solutions had slightly higher mean curvature (d~0.3, not significant)

**Prediction**: This effect should be stronger in reasoning-trained models:
- think > rl_zero > sft > base (in terms of curvature separation)

**Test**:
- Compute effect size for curvature (correct vs incorrect) across all 4 models
- If pattern holds → reasoning training amplifies curvature signal
- If no pattern → curvature magnitude is also not useful

---

### H_curv4: Cross-domain correlation holds for ALL task pairs

**Prediction**: Adding GSM8K (math) will still show r > 0.95 with HumanEval/LogiQA

**Test**:
- Need to recollect GSM8K (currently corrupted for base model)
- Compute 3×3 curvature correlation matrix

**Alternative H_curv4b**: Dissimilar tasks will have lower correlation
- E.g., HumanEval↔LogiQA: r=0.99 (both "reasoning")
- E.g., HumanEval↔Translation: r=0.70 (different cognitive demands)

**Implication**: Would tell us if "reasoning tasks" form a curvature cluster.

---

### H_curv5: Correct-vs-incorrect curvature correlation differs from all-samples correlation

**Key insight**: We computed r=0.996 on ALL samples. What if we condition on correctness?

**Prediction**:
- Correct_HumanEval ↔ Correct_LogiQA: r ≈ 0.99 (correct solutions similar across domains)
- Incorrect_HumanEval ↔ Incorrect_LogiQA: r ≈ 0.99 (incorrect also similar)
- Correct ↔ Incorrect (within domain): r < 0.90 (correct and incorrect differ)

**Test**:
- Compute curvature profiles separately for correct and incorrect
- Compare within-correctness cross-domain vs cross-correctness within-domain

**Implication**: If correct≠incorrect curvature profiles, we have a useful signal even if cross-domain transfers.

---

## The Geodesic Question

You asked: "If we think of the path as the geodesic on whichever manifold, there will be different geodesics on different model manifolds that all transfer well"

**Interpretation**:
- Each model (base/sft/rl_zero/think) defines a different "manifold"
- Trajectories are paths on these manifolds
- If all manifolds have similar curvature structure → the "shape" of the manifold is preserved by training

**What training might change**:
1. **Location** on manifold where correct solutions land
2. **Attractor basins** that pull trajectories toward correct answers
3. **Direction** that separates correct/incorrect

**What training might NOT change**:
1. **Curvature profile** (intrinsic geometry of transformer computation)
2. **Layer-wise processing structure** (early/middle/late layer roles)

---

## Proposed Experiments (Priority Order)

### Experiment 1: Cross-model curvature comparison
**Goal**: Test H_curv1 (is curvature architectural?)
**Data needed**: Same task on multiple models
**Currently available**: HumanEval on olmo3_base, olmo3_sft, olmo3_rl_zero, olmo3_think
**Compute**: ~30 min on eyecog

### Experiment 2: Correctness-conditioned curvature
**Goal**: Test H_curv5 (do correct/incorrect have different curvature?)
**Data needed**: Already have (current data)
**Compute**: ~10 min on eyecog

### Experiment 3: Add GSM8K to correlation matrix
**Goal**: Test H_curv4 (does math task fit the pattern?)
**Data needed**: Need to recollect GSM8K for olmo3_base
**Compute**: Need GPU (~1 hr collection)

### Experiment 4: Non-reasoning task comparison
**Goal**: Test H_curv4b (do non-reasoning tasks differ?)
**Data needed**: Need to collect on translation/QA task
**Compute**: Need GPU + new task setup

---

## Quick Test We Can Do Now

**H_curv5 quick test**: Compute curvature profile separately for correct vs incorrect samples in existing data.

```python
# Pseudocode
correct_mask = labels == True
incorrect_mask = labels == False

curv_correct = [compute_profile(t) for t in traj[correct_mask]]
curv_incorrect = [compute_profile(t) for t in traj[incorrect_mask]]

# Compare within-correctness cross-domain
r_correct_crossdomain = corr(mean(curv_correct_humaneval), mean(curv_correct_logiqa))
r_incorrect_crossdomain = corr(mean(curv_incorrect_humaneval), mean(curv_incorrect_logiqa))

# Compare cross-correctness within-domain
r_humaneval_cross_correctness = corr(mean(curv_correct_humaneval), mean(curv_incorrect_humaneval))
```

**If** r_cross_correctness < r_cross_domain → curvature distinguishes correct/incorrect better than domain
**If** r_cross_correctness ≈ r_cross_domain → curvature is domain-invariant AND correctness-invariant (not useful)

---

## Summary

The r=0.996 finding is potentially a **red herring**:
- It might just mean "transformers process information similarly through layers regardless of task"
- The *interesting* question is whether curvature differs by CORRECTNESS, not by domain

**Priority**: Test H_curv5 (correctness-conditioned curvature) before drawing conclusions about transfer.
