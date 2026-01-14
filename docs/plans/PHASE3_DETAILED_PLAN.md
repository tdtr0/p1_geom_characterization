# Phase 3: Cross-Domain Transfer Testing (H2)

**Status**: ⏳ Pending (depends on Phase 2 success)  
**Duration**: Weeks 5-6 (2 weeks)  
**Objective**: Test H2 - whether correct reasoning geometry transfers across domains (math → code → logic)

---

## Overview

Phase 3 is the **critical test** of the research hypothesis. If H2 fails, the entire "universal geometry of reasoning" premise fails.

**The question**: Can a classifier trained to distinguish correct vs incorrect trajectories on math problems work on code and logic problems (zero-shot, no training on target domains)?

**Success criterion**: >55% transfer accuracy (above 50% chance)

---

## Experimental Design

### Transfer Matrix

We test all pairwise domain transfers:

| Train Domain | Test Domain | Expected Difficulty |
|--------------|-------------|-------------------|
| GSM8K (math) | HumanEval (code) | Hard (different reasoning types) |
| GSM8K (math) | LogiQA (logic) | Medium (both symbolic) |
| HumanEval (code) | GSM8K (math) | Hard (different reasoning types) |
| HumanEval (code) | LogiQA (logic) | Medium (both structured) |
| LogiQA (logic) | GSM8K (math) | Medium (both symbolic) |
| LogiQA (logic) | HumanEval (code) | Hard (different reasoning types) |

**Total**: 6 transfer tests per model = 24 tests (4 models × 6 transfers)

### Within-Format Control

To distinguish domain transfer from format transfer, also test:

| Train | Test | Purpose |
|-------|------|---------|
| GSM8K | MATH | Within-format (both math, different difficulty) |
| HumanEval | MBPP | Within-format (both code, different style) |

**Prediction**: Within-format should work better than cross-format if format is a confound.

---

## Week-by-Week Breakdown

### Week 5: Cross-Domain Classification

**Day 1-2: Prepare Transfer Pipeline**

Create `scripts/test_h2_transfer.py`:

```python
def test_transfer(train_model, train_task, test_task):
    # Load signatures
    train_sigs = load_signatures(train_model, train_task)
    test_sigs = load_signatures(train_model, test_task)
    
    # Load labels
    train_labels = load_labels(train_model, train_task)
    test_labels = load_labels(train_model, test_task)
    
    # Train classifier on source domain
    clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    clf.fit(train_sigs, train_labels)
    
    # Test on target domain (zero-shot)
    test_accuracy = clf.score(test_sigs, test_labels)
    
    return {
        'train_model': train_model,
        'train_task': train_task,
        'test_task': test_task,
        'accuracy': test_accuracy,
        'n_test_samples': len(test_labels),
        'test_class_balance': test_labels.mean()
    }
```

**Day 3-5: Run All Transfer Tests**

For each model (Base, SFT, RL-Zero, Think):
- Train on GSM8K, test on HumanEval and LogiQA
- Train on HumanEval, test on GSM8K and LogiQA
- Train on LogiQA, test on GSM8K and HumanEval

**Day 6-7: Analyze Results**

Compute:
- Mean transfer accuracy across all pairs
- Best/worst transfer pairs
- Model comparison (does RL-Zero transfer better than SFT?)
- Asymmetry analysis (is math → code easier than code → math?)

**Deliverable**: `results/h2_transfer_matrix.csv`

### Week 6: Deep Dive and Controls

**Day 1-2: Difficulty Stratification**

Test if transfer works within difficulty strata:
- Bin problems by difficulty (easy/medium/hard)
- Train on easy math, test on easy code
- Train on hard math, test on hard code

**Hypothesis**: If transfer fails overall but succeeds within strata, difficulty is a confound.

**Day 3-4: Feature Decomposition**

Identify which geometric features transfer:
- Train separate classifiers on different feature subsets:
  - Signature coefficients only
  - Curvature measures only
  - Trajectory length only
- Test which features transfer best

**Day 5: Baseline Comparisons**

Compare transfer performance to baselines:
- **Baseline 1**: Train on random labels (should be ~50%)
- **Baseline 2**: Transfer model confidence (logits) instead of geometry
- **Baseline 3**: Transfer output length

**Success criterion**: Geometry transfer > all baselines

**Day 6-7: Write Transfer Report**

Document:
- Transfer matrix with confidence intervals
- Best/worst performing transfers
- Comparison to baselines
- Feature importance for transfer
- Decision: Does H2 succeed or fail?

**Deliverable**: `results/phase3_transfer_report.md`

---

## Analysis Details

### Statistical Testing

For each transfer test, compute:
- **Accuracy**: Fraction correct
- **Confidence interval**: Bootstrap 95% CI (1000 samples)
- **Significance**: Binomial test against 50% chance
- **Effect size**: Cohen's h for proportion difference

**Report format**:
```
GSM8K → HumanEval (olmo3_rl_zero):
  Accuracy: 58.3% [54.1%, 62.5%]
  p-value: 0.003 (vs 50% chance)
  Effect size: h = 0.17 (small)
  Interpretation: Weak but significant transfer
```

### Transfer Success Criteria

**Strong transfer** (supports H2):
- Mean accuracy > 60% across all transfers
- At least 4/6 transfers significantly above chance (p < 0.05)
- Beats all baselines

**Weak transfer** (challenges H2):
- Mean accuracy 52-58%
- Only 2-3/6 transfers significant
- Comparable to some baselines

**No transfer** (falsifies H2):
- Mean accuracy ≤52%
- No transfers significantly above chance
- Baselines perform equally well

### Model Comparison

**Hypothesis**: RL-Zero may transfer better than SFT (if RLVR learns more general reasoning)

**Test**: Compare mean transfer accuracy across models
- RL-Zero vs SFT: t-test on 6 transfer accuracies
- RL-Zero vs Think: t-test
- Base vs all fine-tuned: ANOVA

**Interpretation**:
- If RL-Zero > SFT: RLVR may learn more transferable geometry
- If no difference: Training paradigm doesn't affect transfer
- If Base > fine-tuned: Post-training may hurt transfer (unlikely)

---

## Confound Analysis

### Difficulty Confound

**Test**: Stratified transfer
- Bin problems by difficulty (using problem length as proxy)
- Train/test within same difficulty bin

**Interpretation**:
- If stratified transfer succeeds but overall fails: Difficulty is confound
- If both fail: Geometry doesn't transfer, regardless of difficulty

### Format Confound

**Test**: Within-format vs cross-format transfer
- Within: GSM8K → MATH (both math)
- Cross: GSM8K → HumanEval (math → code)

**Interpretation**:
- If within >> cross: Format is confound
- If within ≈ cross: Format is not the issue

### Length Confound

**Test**: Length-matched transfer
- Match correct/incorrect pairs by output length (±10%)
- Re-run transfer tests on matched subset

**Interpretation**:
- If matched transfer >> unmatched: Length is confound
- If similar: Length is not the issue

---

## Decision Tree

### If H2 Succeeds (Mean Transfer > 58%)

**Next steps**:
1. Proceed to Phase 4 (H4: Steering)
2. Write up positive results
3. Test H3 (non-verifiable domains) if resources allow

**Paper framing**: "Geometric Signatures of Reasoning Transfer Across Domains"

### If H2 Partially Succeeds (Some Transfers Work)

**Next steps**:
1. Characterize which transfers work and why
2. Measure transfer as function of domain similarity
3. Identify transferable vs domain-specific features

**Paper framing**: "Domain Similarity and Geometric Transfer in LLM Reasoning"

### If H2 Fails (No Transfer Above Chance)

**Pivot strategies**:

**Pivot 1: Domain-Specific Geometry**
- Characterize what differs across domains
- Train domain-specific detectors (still useful)
- Paper: "Domain-Specific Geometric Signatures of Reasoning"

**Pivot 2: Hierarchical Transfer**
- Test finer-grained domain similarities
- Math → Physics → Chemistry → Biology
- Measure transfer decay with domain distance

**Pivot 3: Feature Engineering**
- Current features may not capture transferable aspects
- Try: Attention patterns, gradient flow, layer-wise changes
- Paper: "What Geometric Features Capture Reasoning?"

---

## Risks and Mitigation

### Risk 1: Transfer Tests Are Underpowered

**Problem**: With 500 test samples, detecting small effects (55% vs 50%) requires large N.

**Mitigation**:
- Use bootstrap confidence intervals
- Report effect sizes, not just p-values
- Consider collecting more test samples if results are borderline

### Risk 2: Class Imbalance Affects Transfer

**Problem**: If test set is 80% correct, classifier can achieve 80% by always predicting "correct."

**Mitigation**:
- Report precision, recall, F1 (not just accuracy)
- Use balanced accuracy: (TPR + TNR) / 2
- Stratified sampling to balance test set

### Risk 3: Overfitting on Source Domain

**Problem**: Classifier may overfit to source domain specifics.

**Mitigation**:
- Use cross-validation on source domain
- Regularize classifier (max_depth, min_samples_leaf)
- Try simpler classifiers (logistic regression)

---

## Deliverables

### Data Files

- `results/h2_transfer_matrix.csv`: All transfer results
- `results/h2_feature_importance.csv`: Which features transfer
- `results/h2_difficulty_stratified.csv`: Stratified transfer results

### Visualizations

- Transfer heatmap (6×4 matrix: transfers × models)
- Feature importance bar chart
- Accuracy vs domain similarity scatter plot

### Reports

- `results/phase3_transfer_report.md`: Full analysis
- `results/phase3_decision.md`: Go/no-go for Phase 4

---

## Success Criteria

**Minimum viable**:
- All 24 transfer tests completed
- Statistical tests performed
- Decision made on H2 (succeed/fail/partial)

**Target**:
- Mean transfer accuracy > 55%
- At least 50% of transfers significantly above chance
- Clear interpretation of results

**Stretch**:
- Mean transfer accuracy > 60%
- Identified specific transferable geometric features
- Model comparison shows RL-Zero > SFT for transfer

---

## Timeline Contingencies

**If H2 clearly succeeds** (by Day 3 of Week 5):
- Skip some redundant tests
- Move to Phase 4 early
- Use extra time for H4 implementation

**If H2 clearly fails** (by Day 3 of Week 5):
- Stop transfer tests
- Focus on understanding why (confound analysis)
- Begin pivot planning

**If results are ambiguous**:
- Extend analysis by 1 week
- Collect more samples if needed
- Consult literature for similar effect sizes
