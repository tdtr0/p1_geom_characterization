# Geometric Signatures of Reasoning in LLMs

**Project: ManiVer (Manifold Verification)**

## Executive Summary

This research investigates whether correct reasoning has a characteristic geometric signature in the flow of activations through transformer layers. If such signatures exist and transfer across domains, we can:

1. **Detect** bad reasoning on non-verifiable domains (where we can't check answers)
2. **Steer** models toward correct reasoning trajectories
3. **Understand** what distinguishes correct from incorrect reasoning at the representation level

**Core insight**: Reasoning is a *process*, not a static representation. The trajectory of activations through layers IS the computation. We should analyze flow geometry, not just static subspaces.

---

## Background: What Phase 1 Established

Phase 1 (complete) showed RLVR and SFT models have different static geometry:

| Model | Subspace Preservation vs Base |
|-------|-------------------------------|
| RL-Zero | 98.6% ± 1.1% |
| SFT | 52.4% ± 12.6% |
| Think (RLVR) | 50.4% ± 12.9% |

**Key finding**: RL-Zero preserves base model subspace structure. SFT variants do not.

**Limitation**: This tells us models *differ*, but not whether geometry captures *reasoning quality*.

---

## Novelty and Prior Work

### What Already Exists

**Hidden states encode correctness** (Zhang et al. 2025, Afzal et al. 2025, Azaria & Mitchell 2023):
- Probes can predict whether a model will answer correctly
- This works even before the answer is generated
- **Limitation**: Tested only within single domains (math, or QA, or code)

**Truth has geometric structure** (Marks & Tegmark 2023):
- Linear truth directions transfer across factual datasets
- **Limitation**: All tests are on factual recall, not reasoning processes

**Activation steering works** (Turner et al. 2023, Meng et al. 2022):
- Can modify model behavior by editing activations
- **Limitation**: Tested on simple attributes (honesty, toxicity), not complex reasoning

### What's Novel in This Work

**1. Cross-domain reasoning transfer (H2)**
- **New**: Test if correctness geometry transfers math → code → logic
- **Why it matters**: Prior work only tested within-domain or across similar factual tasks
- **Risk**: This may fail (geometry could be domain-specific)

**2. Trajectory-based analysis**
- **New**: Analyze activation *flow* through layers, not just static representations
- **Method**: Path signatures (reparameterization-invariant trajectory features)
- **Why it matters**: Reasoning is a process; trajectories capture temporal dynamics

**3. Causal intervention on reasoning (H4)**
- **New**: Steer trajectories to improve reasoning accuracy
- **Why it matters**: Tests whether geometry is causally relevant, not just correlational
- **Risk**: Steering may break the model or only work within-domain

**4. Systematic confound controls**
- **New**: Explicit difficulty stratification, baseline comparisons, control tasks
- **Why it matters**: Prior work often conflates correctness with difficulty/length

### What We're NOT Claiming

- ❌ That we've "solved" reasoning or interpretability
- ❌ That geometry *causes* good reasoning (only that it correlates, unless H4 succeeds)
- ❌ That this works on all models or all tasks
- ✅ That we're testing a specific, falsifiable hypothesis about universal reasoning geometry

---

## The Real Research Question

> **Can we learn the geometry of correct reasoning from verifiable domains (where we know the right answer) and use it on non-verifiable domains (where we don't)?**

This is fundamentally different from "transfer prediction":
- Transfer asks: "Does training on A help with B?"
- We ask: "Does CORRECT reasoning have universal geometric signatures?"

**Why this matters**: Verifiable domains (math, code) provide ground truth. Non-verifiable domains (open-ended reasoning, ethics, strategy) are where we *need* a reasoning quality detector.

---

## Hypotheses

### H1: Correct vs Incorrect Reasoning Have Distinguishable Trajectories

On verifiable tasks, trajectories for problems solved correctly vs incorrectly should be geometrically distinguishable.

**Test**: Train classifier on trajectory signatures (correct vs incorrect). Cross-validate within domain.

**Success criterion**: >65% accuracy (significantly above 50% chance)

**Risk**: May just learn problem difficulty, not reasoning quality.

### H2: The Correct Reasoning Signature is Domain-Invariant

If correct reasoning has universal geometry, a classifier trained on math should work on code.

**Test**: Train on GSM8K correct/incorrect, test on HumanEval correct/incorrect (zero-shot).

**Success criterion**: >55% transfer accuracy

**This is the critical test**. If it fails, "reasoning" may be domain-specific.

### H3: Detector Works on Non-Verifiable Domains

Apply the trained detector to domains without ground truth. Validate against human judgments.

**Test**: Compute geometric "reasoning quality" scores on philosophical questions. Correlate with human ratings.

**Success criterion**: r > 0.25 correlation with human judgments

### H4: Trajectories Can Be Steered Toward Correct Reasoning

If we know the geometry of correct reasoning, can we constrain generation to stay in that manifold?

**Test**: Project activations onto "correct reasoning" manifold during inference. Measure accuracy change.

**Success criterion**: >2% accuracy improvement on held-out verifiable problems

### H5: Correct Reasoning Has Lower Curvature

Correct reasoning may follow straighter paths (model knows where it's going). Incorrect reasoning may wander.

**Test**: Compare path curvature for correct vs incorrect trajectories.

---

## Experimental Design

### Data Collection (Phase 2A)

**What we collect**:
- Trajectories at even layers: [0, 2, 4, ..., 30] = 16 layers
- 500 samples per task (GSM8K, HumanEval, LogiQA)
- 4 models (Base, SFT, RL-Zero, Think)
- **Critically**: Record model outputs and correctness labels

**Storage**: ~56 GB total on eyecog

### Geometric Features

**Path Signatures** (via signatory library):
- Reparameterization-invariant trajectory features
- Captures curvature, winding, self-intersection
- Project to 64 dims before computing (d=4096 too large)

**Curvature Measures**:
- Local turning angles between consecutive layers
- Aggregate statistics (mean, variance, max)

**Trajectory Distance**:
- Wasserstein distance between correct/incorrect distributions
- DTW distance between individual trajectories

### Classification Pipeline

```python
# H1 Test: Within-domain classification
for model in models:
    for task in tasks:
        correct_sigs = get_signatures(trajectories[correct])
        incorrect_sigs = get_signatures(trajectories[incorrect])

        clf = RandomForestClassifier()
        accuracy = cross_val_score(clf, all_sigs, labels, cv=5).mean()

# H2 Test: Cross-domain transfer
clf = train(math_correct, math_incorrect)
code_accuracy = clf.evaluate(code_correct, code_incorrect)
logic_accuracy = clf.evaluate(logic_correct, logic_incorrect)
```

### Confound Controls

**Critical confounds to address**:

**1. Problem Difficulty**
- Easy problems may have shorter, more direct trajectories
- Hard problems may exhibit more "wandering"
- **Control**: Stratify by difficulty proxy (problem length, model perplexity)
- **Test**: Train separate classifiers for easy/medium/hard problems
- **Success**: Accuracy remains >60% within each difficulty stratum

**2. Output Length**
- Correct solutions may systematically differ in length
- **Control**: Match correct/incorrect pairs by token count (±10%)
- **Test**: Classification accuracy on length-matched pairs

**3. Surface Format**
- Different domains have different output formats
- **Control**: Test within-format transfer (GSM8K → MATH, both math)
- **Test**: If within-format transfer succeeds but cross-format fails, it's format not reasoning

**4. Random Label Control** (Hewitt & Liang 2019)
- Train classifier on shuffled labels
- **Success criterion**: Control accuracy ≈50% (geometry encodes meaningful signal)
- **Failure criterion**: Control accuracy >55% (classifier learns spurious features)

### Baseline Comparisons

Geometry-based methods should outperform or complement simpler baselines:

**Baseline 1: Model Confidence**
- Use model's own probability estimates: P(correct | logits)
- Expected performance: 60-70% (Kadavath et al. 2022)

**Baseline 2: Output Length**
- Classify based on token count alone
- Expected performance: 55-60%

**Baseline 3: Semantic Entropy** (if compute allows)
- Sample multiple outputs, measure semantic consistency
- Expected performance: 70-75% (Farquhar et al. 2024)

**Success criterion**: Trajectory geometry achieves >65% AND adds value beyond baselines (e.g., ensemble with confidence improves over either alone)

---

## Timeline

### Weeks 1-2: Data Collection
- Run cleanup on eyecog (`./scripts/cleanup_smallworld.sh`)
- Collect trajectories with correctness labels
- 500 samples × 3 tasks × 4 models

### Weeks 3-4: H1 Test
- Compute path signatures
- Train correct/incorrect classifiers per domain
- Report accuracy and feature importance

### Weeks 5-6: H2 Test (Decision Point)
- Cross-domain transfer of classifiers
- **If fails**: Pivot to understanding WHY (what's domain-specific?)
- **If succeeds**: Proceed to intervention

### Weeks 7-10: H4 Test (If H2 Succeeds)
- Implement trajectory steering
- Test on held-out verifiable problems

### Weeks 11-12: Write-up
- Document results (including negative results)
- Prepare paper

---

## Anticipated Challenges and Mitigation Strategies

### Challenge 1: The Decision-Before-Reasoning Problem

**Issue**: Recent work (Afzal et al. 2025, David 2025) shows models commit to answers early in CoT generation. Trajectory geometry may capture elaboration quality, not decision-making.

**Mitigation**:
- Analyze layer-wise: Test if early layers (0-15) are more predictive than late layers (16-31)
- If early layers dominate, we're capturing decision structure (valuable)
- If late layers dominate, we're capturing elaboration (still potentially useful)

**Implication**: Reframe findings accordingly—be explicit about what we're measuring.

### Challenge 2: Confounds May Dominate Signal

**Issue**: Geometry may distinguish difficulty/length/format rather than reasoning quality.

**Mitigation**:
- **Difficulty stratification**: Mandatory for all analyses
- **Length matching**: Compare correct/incorrect of similar token count
- **Within-format transfer**: Test GSM8K → MATH before GSM8K → HumanEval
- **Control tasks**: Random label baseline must fail (≈50% accuracy)

**Decision point**: If controls show confounds dominate, pivot to understanding what geometry actually captures.

### Challenge 3: H2 May Fail (Critical Risk)

**Issue**: Cross-domain transfer is the linchpin hypothesis. If it fails, the "universal geometry" premise fails.

**Mitigation**:
- **Hierarchical transfer**: Test closer domains first (GSM8K → MATH)
- **Asymmetric transfer**: Check if math → code works better than code → math
- **Feature decomposition**: Identify which geometric features transfer vs which don't

**Pivot strategies** (if H2 fails):
1. Characterize domain-specific vs shared geometric features
2. Measure transfer as function of domain similarity
3. Focus on within-domain applications (still useful)

### Challenge 4: Sample Size for Transfer Tests

**Issue**: 500 samples per task may yield imbalanced classes (e.g., 350 correct, 150 incorrect).

**Mitigation**:
- Monitor class balance during collection
- If model is too accurate (>80%), sample harder problems
- Ensure minimum 100 samples per class for robust classification

### Challenge 5: HumanEval Correctness is Expensive

**Issue**: Running test cases for every sample is slow and potentially unsafe (code execution).

**Mitigation**:
- Use sandboxed execution environment (Docker container with timeouts)
- Collect HumanEval last (after GSM8K and LogiQA validate the approach)
- Consider using pass@1 from existing benchmarks if available

### Challenge 6: Path Signatures May Be Brittle

**Issue**: Signatures require dimensionality reduction (4096 → 64 dims). Choice of projection may affect results.

**Mitigation**:
- Test multiple projection methods (PCA, random projection, UMAP)
- Test multiple signature depths (2, 3, 4)
- Report robustness across choices

---

## Success Criteria

| Outcome | Implication | Next Step |
|---------|-------------|-----------|
| H1 success, H2 success | Reasoning has universal geometry | Proceed to H4 (intervention) |
| H1 success, H2 fails | Reasoning is domain-specific | Analyze what differs across domains |
| H1 fails | Geometry doesn't capture reasoning | Major pivot needed |
| H4 success | We can improve reasoning via geometry | Write paper, major contribution |
| H4 fails | Steering is too crude, or geometry isn't causal | Try finer-grained intervention |

---

## Honest Assessment of Risks

### The Core Confound

**"Correct reasoning" geometry might just be "easy problem" geometry.**

If easy problems have certain signatures and hard problems have others, we learn a difficulty detector, not a reasoning quality detector.

**Mitigation**: Analyze within difficulty strata. Compare correct vs incorrect on problems of similar difficulty.

### What We're Likely to Show
- H1 will probably succeed (geometry distinguishes *something*)
- The question is whether it's reasoning vs. difficulty vs. length vs. other confounds

### What's Uncertain
- H2 (cross-domain transfer) - this is the critical unknown
- Whether any intervention (H4) will work

### Even if We Fail
- We learn geometry doesn't capture transferable reasoning
- This is valuable negative result that saves others from this path

---

## File Structure

```
ManiVer/
├── RESEARCH_PLAN.md              # This file (main plan)
├── PHASE2_PLAN.md                # Data collection details
├── phase1_implementation_plan.md # Phase 1 (complete)
├── archive_transfer_correlation_plan.md  # Old approach (archived)
├── paper/
│   └── geometric_compression_research_plan.md  # Full technical details
├── scripts/
│   ├── collect_trajectories_half_layers.py  # Collection script
│   ├── cleanup_smallworld.sh                # Disk cleanup
│   └── run_analysis.py                      # Analysis pipeline
├── src/
│   ├── activation_collector.py
│   ├── task_data.py
│   └── geometric_measures.py
└── data/
    ├── activations/    # Phase 1 data
    └── trajectories/   # Phase 2 data (to be collected)
```

---

## Compute Resources

- **Server**: eyecog (2x RTX 3090 24GB)
- **Storage**: ~160 GB available after cleanup
- **Estimated GPU hours**: 40-60 for collection + analysis

---

## Key Decisions Made

1. **Even layers only**: Layer smoothness analysis shows negligible difference between consecutive layers
2. **500 samples**: Need enough for correct/incorrect split with sufficient N
3. **Correctness labels**: Critical addition - without these, we can't test H1/H2
4. **Path signatures**: Primary trajectory feature (reparameterization-invariant)
5. **Focus on verifiable domains first**: Math, code, logic have ground truth

---

## What's NOT in This Plan

1. **Optimal transport / rectified flow**: Deferred until H1-H2 show signal
2. **Human evaluation (H3)**: Only if H2 succeeds and resources allow
3. **Fine-grained layer analysis**: Start with aggregate, refine if needed
4. **Multiple temperature sampling**: Focus on greedy first

These can be added if initial results warrant.
