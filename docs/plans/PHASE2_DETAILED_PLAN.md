# Phase 2: Trajectory Collection with Correctness Labels

**Status**: 🔄 In Progress  
**Duration**: Weeks 1-4 (4 weeks)  
**Objective**: Collect activation trajectories with correctness labels to test H1 (distinguishable trajectories) and H2 (domain-invariant signatures)

---

## Overview

Phase 2 extends Phase 1 by collecting **full activation trajectories** (not just final states) and recording **correctness labels** for each sample. This enables testing whether correct vs incorrect reasoning have distinguishable geometric signatures.

**Key difference from Phase 1**: We now generate answers and check correctness, not just collect activations on prompts.

---

## Data Collection Specifications

### What We Collect

**Per sample**:
- Activation trajectories at even layers: [0, 2, 4, ..., 30] = 16 layers
- Shape: (seq_len, n_layers, d_model) = (512, 16, 4096)
- Model output (generated answer)
- Ground truth answer
- **Correctness label**: Boolean (model answer matches ground truth)

**Samples**:
- 500 per task × 3 tasks × 4 models = 6,000 total samples
- Expected class balance: ~60-70% correct, ~30-40% incorrect (varies by model/task)

**Storage**:
- Per file: ~4.7 GB compressed (gzip)
- Total: 4 models × 3 tasks × 4.7 GB = **~56 GB**

### Models

1. **olmo3_base**: Baseline (no post-training)
2. **olmo3_sft**: Supervised fine-tuning only
3. **olmo3_rl_zero**: Pure RL (no SFT)
4. **olmo3_think**: Full pipeline (SFT + DPO + RLVR)

### Tasks

1. **GSM8K**: Math word problems
   - Correctness: Extract `#### <number>`, exact numerical match
   - Expected accuracy: Base ~20%, SFT ~60%, RL-Zero ~40%, Think ~70%

2. **HumanEval**: Python coding
   - Correctness: Run test cases (sandboxed execution)
   - Expected accuracy: Base ~10%, SFT ~40%, RL-Zero ~30%, Think ~50%

3. **LogiQA**: Logical reasoning
   - Correctness: Extract A/B/C/D, exact match
   - Expected accuracy: Base ~25%, SFT ~50%, RL-Zero ~35%, Think ~55%

---

## Week-by-Week Breakdown

### Week 1: Setup and GSM8K Collection

**Day 1-2: Environment Setup**
- Clean up eyecog disk space: `./scripts/cleanup_smallworld.sh`
- Verify 160+ GB free
- Test collection script on 10 samples (synthetic data)

**Day 3-7: GSM8K Collection**
- Run: `python scripts/collect_trajectories_with_labels.py --task gsm8k`
- 4 models × 500 samples = 2,000 samples
- Estimated time: 20-30 GPU hours
- Monitor: Correctness rates, checkpoint every 25 samples

**Deliverable**: 4 HDF5 files with GSM8K trajectories + labels

### Week 2: LogiQA and HumanEval Collection

**Day 1-3: LogiQA Collection**
- Run: `python scripts/collect_trajectories_with_labels.py --task logiqa`
- Faster than GSM8K (shorter outputs)
- Estimated time: 15-20 GPU hours

**Day 4-7: HumanEval Collection**
- Run: `python scripts/collect_trajectories_with_labels.py --task humaneval`
- Slower (code execution for correctness)
- Use Docker sandbox for safety
- Estimated time: 25-35 GPU hours

**Deliverable**: 8 more HDF5 files (LogiQA + HumanEval)

### Week 3: Path Signature Computation

**Compute path signatures for all trajectories**:
- Use `signatory` library
- Signature depth: 3 (captures up to 3rd-order interactions)
- Projection: PCA to 64 dims before signature (4096 too large)

**Script**: `scripts/compute_signatures.py` (to be created)

```python
import signatory
from sklearn.decomposition import PCA

# For each trajectory file
trajectories = load_trajectories(file)  # (n_samples, seq_len, n_layers, d_model)

# Project to lower dimension
pca = PCA(n_components=64)
trajectories_proj = pca.fit_transform(trajectories.reshape(-1, d_model))
trajectories_proj = trajectories_proj.reshape(n_samples, seq_len, n_layers, 64)

# Compute signatures
signatures = []
for traj in trajectories_proj:
    sig = signatory.signature(torch.tensor(traj), depth=3)
    signatures.append(sig.numpy())

# Save
save_signatures(signatures, output_file)
```

**Deliverable**: Signature files for all 12 trajectory files

### Week 4: Initial Analysis (H1 Test)

**Within-domain classification**:
- For each model/task, train Random Forest classifier
- Features: Path signatures
- Labels: Correct vs incorrect
- Evaluation: 5-fold cross-validation

**Script**: `scripts/test_h1.py` (to be created)

**Success criterion**: Mean accuracy > 65% across model/task combinations

**Deliverable**: 
- H1 results table (12 rows: 4 models × 3 tasks)
- Feature importance analysis
- Decision: Proceed to H2 if H1 succeeds

---

## Correctness Checking Details

### GSM8K

**Extraction**:
1. Look for `#### <number>` pattern
2. Fallback: "the answer is X" pattern
3. Fallback: Last number in text

**Comparison**: Numerical match (handle floating point)

**Edge cases**:
- Commas in numbers: Remove before comparison
- Negative numbers: Handle sign
- No number found: Mark incorrect

### LogiQA

**Extraction**:
1. Look for "Answer: X" or "The answer is X"
2. Fallback: Standalone letter at end
3. Fallback: First A/B/C/D found

**Comparison**: Exact letter match (case-insensitive)

**Edge cases**:
- Multiple letters: Take first
- No letter found: Mark incorrect

### HumanEval

**Execution** (in sandbox):
1. Compile code (syntax check)
2. Execute code in isolated namespace
3. Check if entry point exists
4. Run test cases
5. Any exception = incorrect

**Safety**:
- Use Docker container with resource limits
- Timeout: 5 seconds per test
- No network access
- Read-only filesystem

---

## Quality Checks

### During Collection

**Monitor**:
- Correctness rates per model/task
- Sample distribution (ensure not all correct or all incorrect)
- Generation failures (track error rate)
- Disk space (ensure not running out)

**Alerts**:
- If correctness rate < 10% or > 90%: May need to adjust difficulty
- If error rate > 5%: Investigate generation issues
- If disk space < 50 GB: Stop and clean up

### After Collection

**Validation**:
- Check for NaN values in trajectories
- Verify correctness labels are balanced (aim for 30-70% correct)
- Spot-check: Manually verify 10 samples per task
- Compare to published benchmarks (if available)

---

## Checkpointing and Fault Tolerance

**Checkpoint every 25 samples**:
- Save: `data/checkpoints/labeled_{model}_{task}.json`
- Contains: `completed_samples`, `n_correct`, `n_incorrect`
- On restart: Resume from last checkpoint

**If collection fails**:
1. Check logs: `collection.log`
2. Identify failed sample
3. Skip or retry with different seed
4. Continue from checkpoint

---

## Compute Resources

### GPU Requirements

- **Per model**: 14-16 GB VRAM (7B models with float16)
- **Available**: 2x RTX 3090 (24 GB each)
- **Strategy**: Run one model at a time, use device_map="auto"

### Time Estimates

| Task | Samples | Time per Sample | Total GPU Hours |
|------|---------|----------------|-----------------|
| GSM8K | 2000 | 30-45 sec | 20-30 |
| LogiQA | 2000 | 20-30 sec | 15-20 |
| HumanEval | 2000 | 40-60 sec | 25-35 |
| **Total** | **6000** | - | **60-85** |

### Storage Breakdown

| Component | Size per File | Total |
|-----------|---------------|-------|
| Trajectories | 4.5 GB | 54 GB |
| Metadata | 0.2 GB | 2 GB |
| **Total** | **4.7 GB** | **56 GB** |

---

## Risks and Mitigation

### Risk 1: Imbalanced Classes

**Problem**: If model is too accurate (>80%), insufficient incorrect samples for classification.

**Mitigation**:
- Monitor correctness rate during collection
- If imbalanced, sample harder problems (e.g., MATH instead of GSM8K)
- Aim for 30-70% correct per model/task

### Risk 2: HumanEval Execution Failures

**Problem**: Code execution may be unsafe or fail frequently.

**Mitigation**:
- Use Docker sandbox with strict resource limits
- Timeout after 5 seconds
- Collect HumanEval last (after validating approach on GSM8K/LogiQA)
- Fallback: Use syntax check only if execution is too problematic

### Risk 3: Disk Space Exhaustion

**Problem**: 56 GB is close to available space (122 GB free).

**Mitigation**:
- Run cleanup script first
- Monitor disk usage during collection
- Compress files immediately after collection
- Offload to cloud storage if needed

### Risk 4: Path Signature Computation Fails

**Problem**: Signatory library may have issues with high-dimensional data.

**Mitigation**:
- Test on small subset first
- Use PCA projection (4096 → 64 dims) before signature
- Try multiple signature depths (2, 3, 4)
- Fallback: Use simpler trajectory features (curvature, length)

---

## Deliverables

### Data Files

- `data/trajectories/{model}/{task}_trajectories.h5` (12 files)
- Each contains: trajectories, sequence_lengths, is_correct, prompts, model_outputs, ground_truth

### Signature Files

- `data/signatures/{model}/{task}_signatures.npy` (12 files)
- Shape: (n_samples, signature_dim)

### Analysis Results

- `results/h1_within_domain_classification.csv`
- Columns: model, task, accuracy, precision, recall, f1, n_correct, n_incorrect

### Documentation

- Collection logs: `collection.log`
- Checkpoints: `data/checkpoints/`
- Summary report: `results/phase2_summary.md`

---

## Success Criteria

**Minimum viable**:
- All 12 trajectory files collected successfully
- At least 100 samples per class (correct/incorrect) per model/task
- H1 test shows >60% accuracy on at least 2/3 of model/task combinations

**Target**:
- All files collected with <5% errors
- Balanced classes (30-70% correct)
- H1 test shows >65% accuracy on all model/task combinations

**Stretch**:
- Path signatures computed successfully
- H1 test shows >70% accuracy
- Feature importance analysis identifies interpretable geometric features

---

## Next Phase Preview

**Phase 3** (if H1 succeeds): Test H2 (cross-domain transfer)
- Train classifier on GSM8K, test on HumanEval and LogiQA
- Success criterion: >55% transfer accuracy
- This is the **critical test** for universal reasoning geometry
