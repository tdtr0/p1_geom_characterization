# Phase 2: Trajectory Analysis Plan

**CRITICAL REVISION (2026-01-11)**: This plan now focuses on **correct vs incorrect reasoning classification**, not just static trajectory analysis. We collect correctness labels and test whether geometry can distinguish correct from incorrect reasoning.

## Core Research Question (Revised)

**Can we learn the geometry of correct reasoning from verifiable domains and use it on non-verifiable domains?**

Key tests:
- **H1**: Can we distinguish correct vs incorrect trajectories? (within-domain)
- **H2**: Does the classifier transfer across domains? (math → code → logic)
- **H4**: Can we steer trajectories toward correct reasoning manifold?

## Validation: Half-Layer Collection is Safe

**Layer Smoothness Analysis Results:**
- Max consecutive jump: **0.09%** (negligible)
- Mean consecutive jump: **0.05%**
- Even vs odd layer bias: **< 0.01%** (essentially identical)
- **Conclusion**: ✓ No critical non-linear transitions detected

Half-layer sampling will NOT miss important geometric dynamics.

## Storage Plan

### Current Status
- Disk: 6% free (122 GB available)
- SmallWorldMasking: 60 GB (cleanable)
- Phase 1 activations: 1.1 GB

### Cleanup Target (~38-45 GB)
```bash
# Run cleanup script
./scripts/cleanup_smallworld.sh
```

**What gets deleted:**
1. `vast_backups/` - 3.9 GB (old training checkpoints)
2. `clean/classification/pregenerated_masks/` - 5.9 GB (duplicate masks)
3. `clean/saturation_investigation/results/` - 5.8 GB (old investigation)
4. `clean/pregenerated_masks/` - 7.6 GB (regenerable mask tensors)
5. `clean/final_dit_evals/run_202508*/202509*/202510*/` - ~15 GB (old runs)
6. Large log files in `clean/classification/` - 150 MB

**After cleanup:** ~160 GB free (10% disk usage)

### Phase 2 Storage Requirements

**Half-Layer Collection (Chosen Strategy):**
- Layers: Even indices [0, 2, 4, ..., 30] = **16 layers** (vs 32 full)
- Samples: **500 per task** (increased for correct/incorrect split)
- Max sequence length: 512 tokens
- Storage per file: ~6.7 GB raw, ~4.7 GB compressed
- Total: 4 models × 3 tasks × 4.7 GB = **~56 GB**
- **Remaining margin: 104 GB** (very safe)

## Collection Script

```bash
# On eyecog
ssh eyecog
cd ~/p1_geom_characterization

# Activate environment
conda activate geometric_transfer

# Run half-layer trajectory collection
python scripts/collect_trajectories_half_layers.py
```

**Features:**
- Even layers only: [0, 2, 4, 6, ..., 30]
- Full token trajectories (not just last_token)
- Fault-tolerant checkpointing
- HDF5 with gzip compression
- Memory-efficient batching

**Expected runtime:** ~8-12 GPU hours

## What Gets Collected

For each model/task combination:
- **File**: `data/trajectories/{model}/{task}_trajectories.h5`
- **Shape**: (500 samples, 512 tokens, 16 layers, 4096 dims)
- **Format**: float16 with gzip compression

**Datasets in each HDF5 file:**
- `trajectories`: (500, 512, 16, 4096) - activation trajectories
- `sequence_lengths`: (500,) - actual lengths before padding
- `prompts`: (500,) - input prompts
- `model_outputs`: (500,) - generated answers
- `is_correct`: (500,) - boolean correctness labels
- `ground_truth`: (500,) - expected answers
- Metadata: model, task, layers, collection date

**CRITICAL**: Must record correctness labels for H1/H2 tests.

---

## Phase 2 Analysis Tasks (Revised)

### Task 1: Correct vs Incorrect Classification (H1)

**Goal**: Can we distinguish correct from incorrect trajectories within a domain?

```python
for model in models:
    for task in tasks:
        correct_mask = data[model][task]['is_correct']
        trajectories = data[model][task]['trajectories']

        correct_sigs = compute_signatures(trajectories[correct_mask])
        incorrect_sigs = compute_signatures(trajectories[~correct_mask])

        # 5-fold cross-validation
        clf = RandomForestClassifier()
        accuracy = cross_val_score(clf, all_sigs, labels, cv=5).mean()

        results[model][task] = {
            'accuracy': accuracy,
            'n_correct': correct_mask.sum(),
            'n_incorrect': (~correct_mask).sum()
        }
```

**Success criterion**: Mean accuracy > 65% across models/tasks

### Task 2: Cross-Domain Transfer (H2)

**Goal**: Does the classifier trained on math work on code?

```python
# Train on GSM8K
math_clf = train_classifier(gsm8k_correct_sigs, gsm8k_incorrect_sigs)

# Test on HumanEval (zero-shot)
code_accuracy = math_clf.evaluate(humaneval_correct_sigs, humaneval_incorrect_sigs)

# Test on LogiQA (zero-shot)
logic_accuracy = math_clf.evaluate(logiqa_correct_sigs, logiqa_incorrect_sigs)

# Also test all transfer directions
transfer_matrix = compute_all_transfers(math, code, logic)
```

**Success criterion**: Mean transfer accuracy > 55% (above chance)

### Task 3: Curvature Analysis (H5)

**Goal**: Do correct trajectories have lower curvature?

```python
correct_curvatures = [compute_curvature(t) for t in correct_trajectories]
incorrect_curvatures = [compute_curvature(t) for t in incorrect_trajectories]

t_stat, p_value = ttest_ind(correct_curvatures, incorrect_curvatures)
effect_size = cohens_d(correct_curvatures, incorrect_curvatures)
```

**Success criterion**: Significant difference (p < 0.05) with meaningful effect size (d > 0.3)

### Task 4: Difficulty Confound Analysis

**Goal**: Ensure we're detecting reasoning quality, not problem difficulty

```python
# Stratify by difficulty (e.g., problem length, complexity score)
for difficulty_bin in difficulty_bins:
    subset = problems_in_bin(difficulty_bin)
    correct_sigs = get_signatures(subset[correct])
    incorrect_sigs = get_signatures(subset[incorrect])

    # Classification within same difficulty
    accuracy = classify(correct_sigs, incorrect_sigs)

    # If accuracy drops to chance within strata, we're detecting difficulty
    results[difficulty_bin] = accuracy
```

---

## Success Criteria (Revised)

| Test | Threshold | Meaning if Pass | Meaning if Fail |
|------|-----------|-----------------|-----------------|
| H1 (within-domain) | >65% accuracy | Geometry captures *something* | Fundamental failure |
| H2 (cross-domain) | >55% accuracy | Reasoning has universal signature | Reasoning is domain-specific |
| H5 (curvature) | p<0.05, d>0.3 | Correct = straighter paths | Curvature not informative |
| Difficulty control | Maintains >60% | Not confounded by difficulty | Actually detecting difficulty |

**Decision point after H2**:
- If H2 succeeds → Proceed to intervention experiments (H4)
- If H2 fails → Analyze what differs across domains (pivot to understanding)

---

## Collection Script Requirements

The collection script needs modification to:
1. **Generate answers** (not just collect activations on prompts)
2. **Check correctness** (compare to ground truth)
3. **Record labels** (is_correct boolean)

```python
# Pseudocode for collection with correctness
for prompt, ground_truth in task_data:
    # Generate answer
    output = model.generate(prompt, max_tokens=256)

    # Check correctness (task-specific)
    is_correct = check_answer(output, ground_truth, task_type)

    # Collect trajectories during generation
    trajectories = collect_trajectories_during_generation(model, prompt)

    # Store with labels
    save(trajectories, output, is_correct, ground_truth)
```

---

## Next Steps

1. **Cleanup**: `./scripts/cleanup_smallworld.sh`
2. **Verify space**: `df -h ~/` (target: ~160 GB free)
3. **Modify collection script**: Add answer generation and correctness checking
4. **Sync to eyecog**: `rsync` from local
5. **Run collection**: `python scripts/collect_trajectories_with_labels.py`
6. **Monitor**: Check `data/checkpoints/`
7. **Verify**: 12 HDF5 files with correctness labels

---

## Open Questions

1. **Which token's trajectory?** Last token of prompt? All tokens? Average?
   - Recommendation: Start with last token of prompt (before generation)

2. **What counts as "correct"?**
   - GSM8K: Extract numerical answer, exact match
   - HumanEval: Run tests, pass@1
   - LogiQA: Multiple choice, exact match

3. **What if model gets most problems right/wrong?**
   - Need balanced samples for classification
   - May need to sample harder/easier problems to balance

4. **How to handle generation randomness?**
   - Use greedy decoding (temperature=0) for reproducibility
   - Consider sampling multiple times if needed for variance analysis
