# Phase 4: Trajectory Steering (H4) - Causal Intervention

**Status**: ⏳ Pending (depends on Phase 3 success)  
**Duration**: Weeks 7-10 (4 weeks)  
**Objective**: Test H4 - whether steering trajectories toward "correct reasoning" geometry improves accuracy

---

## Overview

Phase 4 is the **causal test**. If geometry matters for reasoning, intervening on it should change behavior.

**The question**: Can we project activations onto a "correct reasoning" manifold during inference and improve accuracy on held-out problems?

**Success criterion**: >2% accuracy improvement on held-out verifiable problems

**Why this matters**: 
- H1-H3 are correlational (geometry correlates with correctness)
- H4 is causal (modifying geometry changes correctness)
- Only H4 can establish that geometry is causally relevant

---

## Steering Methods

### Method 1: Subspace Projection (Primary)

**Concept**: Project activations onto the subspace spanned by correct trajectories.

**Algorithm**:
```python
# Learn "correct reasoning" manifold from training data
correct_trajectories = get_trajectories(train_set, correct_only=True)
pca = PCA(n_components=k)  # k = 64 or 128
correct_manifold = pca.fit(correct_trajectories)

# At inference, project activations onto manifold
def steered_forward(model, input, layers_to_steer, alpha):
    activations = []
    x = input
    
    for layer_idx, layer in enumerate(model.layers):
        x = layer(x)
        
        if layer_idx in layers_to_steer:
            # Project onto correct manifold
            x_proj = correct_manifold.inverse_transform(
                correct_manifold.transform(x)
            )
            # Blend original and projected
            x = (1 - alpha) * x + alpha * x_proj
        
        activations.append(x)
    
    return x, activations
```

**Hyperparameters to tune**:
- `k`: Manifold dimensionality (32, 64, 128)
- `alpha`: Steering strength (0.1, 0.3, 0.5, 0.7, 1.0)
- `layers_to_steer`: Which layers (early, middle, late, all)

### Method 2: Activation Addition (Alternative)

**Concept**: Add a "correct reasoning" steering vector (à la Turner et al. 2023).

**Algorithm**:
```python
# Compute steering vector
correct_mean = mean(correct_trajectories)
incorrect_mean = mean(incorrect_trajectories)
steering_vector = correct_mean - incorrect_mean

# At inference, add steering vector
def steered_forward(model, input, layers_to_steer, alpha):
    x = input
    for layer_idx, layer in enumerate(model.layers):
        x = layer(x)
        if layer_idx in layers_to_steer:
            x = x + alpha * steering_vector[layer_idx]
    return x
```

**Simpler but less principled than subspace projection.**

### Method 3: Optimal Transport Map (Stretch Goal)

**Concept**: Learn a transport map from incorrect → correct distribution.

**Algorithm**:
```python
# Learn OT map using neural network
ot_map = train_ot_map(incorrect_trajectories, correct_trajectories)

# At inference, apply map
def steered_forward(model, input, layers_to_steer, alpha):
    x = input
    for layer_idx, layer in enumerate(model.layers):
        x = layer(x)
        if layer_idx in layers_to_steer:
            x_mapped = ot_map(x)
            x = (1 - alpha) * x + alpha * x_mapped
    return x
```

**Most sophisticated but computationally expensive.**

---

## Week-by-Week Breakdown

### Week 7: Implementation and Validation

**Day 1-2: Implement Steering Pipeline**

Create `src/trajectory_steering.py`:
- Subspace projection method
- Activation addition method
- Hook-based intervention during generation

**Day 3-4: Validate on Training Set**

Sanity check:
- Steer on training set (should improve accuracy)
- Try different alpha values
- Verify steering doesn't break generation (no gibberish)

**Day 5-7: Hyperparameter Search**

Grid search over:
- Manifold dimensionality: k ∈ {32, 64, 128}
- Steering strength: α ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
- Layers: {early [0-10], middle [11-20], late [21-31], all}

**Metric**: Accuracy on validation set (20% of training data)

**Deliverable**: Best hyperparameters for each model/task

### Week 8: Held-Out Evaluation

**Day 1-3: Test on Held-Out Verifiable Problems**

For each model/task:
- Baseline: Generate without steering
- Steered: Generate with best hyperparameters
- Measure: Accuracy improvement

**Held-out set**: 100 samples per task (not used in training)

**Day 4-5: Layer-Wise Analysis**

Test steering at different layer groups:
- Early only (0-10): Tests if decision structure matters
- Middle only (11-20): Tests if intermediate reasoning matters
- Late only (21-31): Tests if elaboration matters

**Hypothesis**: Middle layers should be most effective (where decisions form).

**Day 6-7: Failure Analysis**

For samples where steering fails:
- Why did steering not help?
- Did steering make it worse?
- Are there patterns (e.g., only helps on easy problems)?

**Deliverable**: `results/h4_steering_results.csv`

### Week 9: Robustness and Ablations

**Day 1-2: Robustness Checks**

Test if steering generalizes:
- Different manifold learning methods (PCA, UMAP, Autoencoder)
- Different steering strengths
- Different random seeds

**Day 3-4: Ablation Studies**

What components matter?
- Steering vs no steering (main effect)
- Correct manifold vs random manifold (is it specific to correct?)
- Correct manifold vs incorrect manifold (should make it worse)

**Day 5-7: Cross-Domain Steering**

If H2 succeeded, test:
- Learn manifold on math, steer on code
- Does steering transfer across domains?

**Deliverable**: Ablation results table

### Week 10: Analysis and Write-Up

**Day 1-3: Statistical Analysis**

For each model/task:
- Paired t-test: Baseline vs Steered accuracy
- Effect size: Cohen's d
- Confidence intervals: Bootstrap 95% CI

**Day 4-5: Visualization**

Create:
- Accuracy improvement bar chart
- Steering strength vs accuracy curve
- Layer-wise steering effectiveness heatmap

**Day 6-7: Write Steering Report**

Document:
- Does H4 succeed? (>2% improvement)
- Which models benefit most?
- Which layers are most effective?
- Failure modes and limitations

**Deliverable**: `results/phase4_steering_report.md`

---

## Evaluation Protocol

### Baseline Measurement

For each held-out sample:
1. Generate answer without steering (greedy decoding)
2. Check correctness
3. Record: sample_id, baseline_output, baseline_correct

### Steered Measurement

For each held-out sample:
1. Generate answer with steering (same seed)
2. Check correctness
3. Record: sample_id, steered_output, steered_correct

### Comparison

Compute:
- **Baseline accuracy**: % correct without steering
- **Steered accuracy**: % correct with steering
- **Improvement**: Steered - Baseline
- **Significance**: Paired t-test (or McNemar's test for binary outcomes)

**Success criterion**: Improvement > 2% AND p < 0.05

---

## Anticipated Challenges

### Challenge 1: Steering May Break Generation

**Symptom**: Steered outputs are gibberish or truncated.

**Diagnosis**:
- Steering strength too high (α > 0.7)
- Wrong layers steered (early layers disrupted)
- Manifold learned from insufficient data

**Mitigation**:
- Reduce α to 0.1-0.3
- Steer only middle layers (11-20)
- Increase training set for manifold learning

### Challenge 2: Steering Only Helps on Easy Problems

**Symptom**: Improvement on easy problems, no effect on hard problems.

**Diagnosis**: Steering regresses to the mean (pushes toward average).

**Mitigation**:
- Stratify by difficulty
- Learn separate manifolds for easy/medium/hard
- Report results separately by difficulty

### Challenge 3: No Improvement Despite H1/H2 Success

**Symptom**: Geometry distinguishes correct/incorrect, but steering doesn't help.

**Interpretation**: Geometry is correlate, not cause.

**Implications**:
- Still publishable (negative causal result)
- Geometry may be downstream of reasoning, not upstream
- Paper: "Geometric Signatures Correlate But Don't Cause Reasoning Quality"

### Challenge 4: Steering Improves Some Models, Not Others

**Symptom**: Works for RL-Zero, not for SFT.

**Interpretation**: RL-Zero may have more structured geometry.

**Implications**:
- Model-specific findings
- RLVR geometry may be more amenable to intervention
- Paper: "Training Paradigm Affects Steerability of Reasoning"

---

## Safety and Validation

### Sanity Checks

**Check 1: Steering on training set should help**
- If not, implementation is broken

**Check 2: Random manifold should not help**
- If it does, steering is just adding noise that happens to help

**Check 3: Incorrect manifold should hurt**
- If not, steering is not specific to correctness

**Check 4: Steering strength α=0 should match baseline**
- If not, implementation has bugs

### Output Quality Checks

For steered outputs, manually inspect 20 samples:
- Are outputs coherent?
- Are reasoning steps logical?
- Or is model just outputting memorized patterns?

---

## Compute Requirements

### GPU Hours

| Activity | Hours per Model | Total (4 models) |
|----------|----------------|------------------|
| Manifold learning | 2 | 8 |
| Hyperparameter search | 10 | 40 |
| Held-out evaluation | 5 | 20 |
| Ablations | 5 | 20 |
| **Total** | **22** | **88** |

### Storage

- Manifold models: ~100 MB per model/task = 1.2 GB
- Steered outputs: ~500 MB
- Results: ~50 MB

**Total**: ~2 GB

---

## Deliverables

### Code

- `src/trajectory_steering.py`: Steering implementation
- `scripts/test_h4_steering.py`: Evaluation script
- `scripts/hyperparameter_search.py`: Grid search

### Data

- `models/manifolds/{model}_{task}_manifold.pkl`: Learned manifolds
- `results/h4_baseline_outputs.json`: Baseline generations
- `results/h4_steered_outputs.json`: Steered generations

### Analysis

- `results/h4_steering_results.csv`: Main results table
- `results/h4_ablations.csv`: Ablation studies
- `results/phase4_steering_report.md`: Full report

### Visualizations

- Accuracy improvement bar chart
- Steering strength curves
- Layer-wise effectiveness heatmap

---

## Success Criteria

**Minimum viable**:
- Steering implemented and validated
- Tested on held-out set
- Statistical tests performed
- Decision made on H4 (succeed/fail)

**Target**:
- >2% accuracy improvement on at least 2/3 of model/task combinations
- Statistically significant (p < 0.05)
- Improvement not due to confounds (validated via ablations)

**Stretch**:
- >5% accuracy improvement
- Works across all models and tasks
- Cross-domain steering succeeds
- Identified optimal layers and steering strength

---

## Publication Implications

### If H4 Succeeds

**Paper title**: "Geometric Steering of Reasoning in Large Language Models"

**Key contributions**:
1. Showed geometry distinguishes correct/incorrect (H1)
2. Showed geometry transfers across domains (H2)
3. Showed steering geometry improves reasoning (H4 - causal)

**Venue**: NeurIPS, ICML, ICLR (main conference)

**Impact**: High (novel method + causal evidence + practical application)

### If H4 Fails

**Paper title**: "Geometric Signatures of Reasoning: Correlates Without Causation"

**Key contributions**:
1. Showed geometry correlates with correctness (H1)
2. Tested domain transfer (H2 results)
3. Showed geometry is not causally relevant (H4 negative result)

**Venue**: NeurIPS workshop, ICLR workshop, ACL

**Impact**: Medium (negative result, but well-characterized)

---

## Next Steps

**If H4 succeeds**: Proceed to Phase 5 (write-up and publication)

**If H4 fails**: 
- Analyze why (wrong method? wrong features? wrong hypothesis?)
- Try alternative steering methods
- Pivot to understanding what geometry actually captures
