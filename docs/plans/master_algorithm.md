# ManiVer: Master Algorithm and File Map

**Project**: Manifold Verification - Geometric Signatures of Correct Computation in LLMs
**Last Updated**: 2026-01-17

---

## 🎯 One-Line Summary

**Test whether correct solutions have distinguishable dynamical signatures in activation trajectories, and whether these signatures share structure across verifiable domains (math → code → logic).**

---

## 📚 Theoretical Framework

We adopt an **interpolation-centric view** (Allen-Zhu & Li, 2024):

- Transformers compute smooth functions over representation manifolds
- There is no "reasoning mode" vs "recall mode" — all is interpolation
- What differs is the *region* and *dynamics* of the manifold traversal

**Key theoretical connections**:

| Concept | Source | Our Application |
|---------|--------|-----------------|
| Everything is interpolation | Allen-Zhu & Li (2024) | Don't detect "reasoning" — characterize interpolation geometry |
| Curvature regimes | Merullo et al. (2025) | High-curvature = general; low-curvature = memorization |
| Attractor dynamics | Ren & Liu (2026) | Correct solutions find right attractors; incorrect get trapped |
| Belief state geometry | Shai et al. (2024) | Residual stream represents belief states |

**What we're testing**: Do correct solutions have domain-general dynamical signatures (stability, attractor basins, curvature profiles)?

---

## 📂 Directory Structure

```
ManiVer/
├── main/                           # Main project (correct vs incorrect geometry)
│   ├── paper/                      # Research plans and literature reviews
│   ├── src/                        # Core implementation modules
│   ├── scripts/                    # Executable scripts
│   ├── data/                       # Collected data (activations, trajectories)
│   ├── results/                    # Analysis outputs
│   ├── configs/                    # Configuration files
│   └── PHASE{1-5}_DETAILED_PLAN.md # Detailed execution plans
│
├── lit_review/                     # Literature review materials
│   ├── papers/                     # PDF papers (moved from /Papers to read/)
│   └── *.md                        # Literature review notes
│
├── Implementation/                 # Other implementation experiments
├── Topology_of_Reasoning/          # Related project (separate)
└── [other directories]             # Not primary focus
```

---

## 📄 File Map (main/)

### Research Plans and Papers (`main/paper/`)

| File | Purpose | Lines |
|------|---------|-------|
| **RESEARCH_PLAN.md** | Main research plan: H1-H5 hypotheses, experimental design, timeline, controls | 411 |
| **LITERATURE_REVIEW_SHORT.md** | Concise lit review for each hypothesis (supporting + critical evidence) | 209 |
| **LITERATURE_REVIEW_LONG.md** | Extended lit review with detailed analysis and methodology recommendations | 732 |
| **geometric_compression_research_plan.md** | Background on RLVR vs SFT (NOT main focus, for context only) | 1372 |
| **claude.md** | Navigation guide for LLMs working on this project | 177 |

### Phase Execution Plans (`main/`)

| File | Phase | Status | Purpose |
|------|-------|--------|---------|
| **PHASE1_DETAILED_PLAN.md** | 1 | ✅ Complete | Static geometry characterization (RLVR vs SFT) - established baseline |
| **PHASE2_DETAILED_PLAN.md** | 2 | 🔄 In Progress | Trajectory collection with correctness labels (H1 test) |
| **PHASE3_DETAILED_PLAN.md** | 3 | ⏳ Pending | Cross-domain transfer testing (H2 - critical test) |
| **PHASE4_DETAILED_PLAN.md** | 4 | ⏳ Pending | Trajectory steering (H4 - causal intervention) |
| **PHASE5_DETAILED_PLAN.md** | 5 | ⏳ Pending | Write-up and publication |
| **PHASE2_PLAN.md** | 2 | 📝 Original | Original Phase 2 plan (less detailed than DETAILED version) |
| **phase1_implementation_plan.md** | 1 | 📝 Original | Original Phase 1 plan with results summary |
| **archive_transfer_correlation_plan.md** | - | 🗄️ Archived | Old approach (archived, not current) |

### Implementation Code (`main/src/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| **activation_collector.py** | Collect activations from transformers using TransformerLens or hooks | `ActivationCollector`, `collect_activations`, `save_to_hdf5` |
| **geometric_measures.py** | Compute geometric properties of activation manifolds | `compute_effective_rank`, `compute_spectral_decay`, `compute_subspace_preservation` |
| **task_data.py** | Load and format datasets (GSM8K, HumanEval, LogiQA) | `prepare_gsm8k`, `prepare_humaneval`, `prepare_logiqa` |

### Executable Scripts (`main/scripts/`)

| File | Purpose | When to Run |
|------|---------|-------------|
| **collect_trajectories_with_labels.py** | Phase 2: Collect trajectories WITH correctness labels | Phase 2 data collection |
| **collect_trajectories_half_layers.py** | Collect trajectories (even layers only, no labels) | Alternative to above |
| **collect_activations.py** | Phase 1: Collect static activations (last token only) | Phase 1 (complete) |
| **run_analysis.py** | Compute geometric measures on collected data | After data collection |
| **curvature_and_stats.py** | Curvature analysis and statistical tests | Phase 1 analysis |
| **check_layer_smoothness.py** | Validate that even-layer sampling is safe | Pre-Phase 2 validation |
| **cleanup_smallworld.sh** | Free up disk space on eyecog | Before Phase 2 collection |
| **vast_launcher.py** | Manage vast.ai instances for compute | When using cloud compute |
| **verify_pipeline.py** | Test collection pipeline on small sample | Before full collection |

### Configuration (`main/configs/`)

| File | Purpose |
|------|---------|
| **models.yaml** | Model configurations (OLMo 3 family, DeepSeek, Llama) |

### Data (`main/data/`)

| Directory | Contents | Size |
|-----------|----------|------|
| **activations/** | Phase 1 static activations (last token, all layers) | 1.1 GB |
| **trajectories/** | Phase 2 full trajectories (even layers, all tokens) | ~56 GB (target) |
| **checkpoints/** | Collection checkpoints for fault tolerance | <1 MB |

### Results (`main/results/`)

| File | Contents | Created By |
|------|----------|------------|
| **geometric_analysis_detailed.csv** | Per-layer geometric measures | `run_analysis.py` |
| **geometric_analysis_summary.csv** | Aggregated statistics | `run_analysis.py` |
| **h1_within_domain_classification.csv** | H1 test results | Phase 2 analysis |
| **h2_transfer_matrix.csv** | H2 cross-domain transfer results | Phase 3 analysis |
| **h4_steering_results.csv** | H4 steering intervention results | Phase 4 analysis |

---

## 🔬 The Algorithm (High-Level)

### Phase 1: Establish Baseline (✅ Complete)

```
Input: 4 models (Base, SFT, RL-Zero, Think) × 3 tasks (GSM8K, HumanEval, LogiQA)
Process: Collect last-token activations at all layers
Measure: Effective rank, spectral decay, subspace preservation
Output: RLVR preserves base geometry (98%), SFT reshapes it (52%)
```

### Phase 2: Collect Trajectories with Labels (🔄 In Progress)

```
Input: Same 4 models × 3 tasks
Process: 
  1. Generate answers for 500 problems per task
  2. Check correctness (model answer vs ground truth)
  3. Collect activation trajectories (even layers: 0, 2, 4, ..., 30)
  4. Store with correctness labels
Output: 12 HDF5 files with trajectories + is_correct labels (~56 GB)
```

### Phase 3: Test Cross-Domain Transfer (⏳ Pending)

```
Input: Trajectories from Phase 2
Process:
  1. Original analysis:
     - Compute path signatures (via signatory library)
     - Compute Frenet-Serret curvature
  2. NEW: Dynamical systems analysis:
     - Vector field decomposition (potential vs rotational)
     - Lyapunov exponent analysis (stability)
     - Attractor analysis (clustering, convergence rates)
     - Curvature regime activation (if model weights available)
  3. Train classifier on math correct/incorrect (all features)
  4. Test on code and logic (zero-shot)
  5. Analyze which features transfer
Output: H2 result - do dynamical signatures transfer? (>55% = success)
```

### Phase 4: Steering Intervention (⏳ Pending, if H2 succeeds)

```
Input: Trajectories + H2 classifier
Process:
  1. Learn "correct reasoning" manifold from training data
  2. At inference, project activations onto manifold
  3. Measure accuracy improvement on held-out problems
Output: H4 result - does steering help? (>2% improvement = success)
```

### Phase 5: Write-Up (⏳ Pending)

```
Input: All results from Phases 1-4
Process: Write paper, prepare code/data release, submit
Output: Publication + reproducibility package
```

---

## 🔑 Key Concepts

### Original Concepts

| Concept | Definition | Where Used |
|---------|------------|------------|
| **Trajectory** | Activation path through layers: (seq_len, n_layers, d_model) | Phase 2-4 |
| **Path signature** | Reparameterization-invariant trajectory features (via signatory) | Phase 3-4 |
| **Correctness label** | Boolean: model answer matches ground truth | Phase 2-4 |
| **Subspace preservation** | How much base model geometry is preserved after fine-tuning | Phase 1 |
| **Cross-domain transfer** | Classifier trained on domain A works on domain B | Phase 3 (H2) |
| **Activation steering** | Modify activations during inference to change behavior | Phase 4 (H4) |

### New Dynamical Systems Concepts (Phase 3)

| Concept | Definition | Hypothesis |
|---------|------------|------------|
| **Vector field** | Layer transition dynamics: v(x) = x_{l+1} - x_l | Characterize flow structure |
| **Helmholtz decomposition** | Split v into potential (∇φ) and rotational (∇×A) components | Correct = more potential flow |
| **Lyapunov exponent** | Rate of trajectory divergence/convergence | Correct = more stable (λ < 0) |
| **Attractor basin** | Region of state space that converges to a fixed point | Correct/incorrect = different basins |
| **Curvature regime** | High-curvature (general) vs low-curvature (memorization) weight directions | Correct = more high-curvature (PROXY ONLY - see note) |
| **Belief state** | Latent representation of posterior over data-generating process | Correct = better belief updates |

---

## 🎲 Decision Tree

```
Phase 1 (Complete) → Different geometry found
    ↓
Phase 2 (In Progress) → Collect trajectories + labels
    ↓
Phase 3 → Test H2 (cross-domain transfer)
    ├─ H2 succeeds (>55% transfer) → Phase 4 (steering)
    │   ├─ H4 succeeds (>2% improvement) → Major contribution, publish at top venue
    │   └─ H4 fails → Correlation without causation, publish at workshop
    │
    └─ H2 fails (≤55% transfer) → Pivot to domain-specific analysis
        └─ Characterize what differs across domains → Publish at ACL/EMNLP
```

---

## 📊 Data Flow

```
Raw Data (Datasets)
    ↓
[collect_trajectories_with_labels.py]
    ↓
Trajectories + Labels (HDF5)
    ↓
[compute_signatures.py] (to be created)
    ↓
Path Signatures (numpy)
    ↓
[test_h1.py] → Within-domain classification
[test_h2_transfer.py] → Cross-domain transfer
[test_h4_steering.py] → Steering intervention
    ↓
Results (CSV, figures)
    ↓
Paper + Code Release
```

---

## 🔧 Information Flow (Trajectory Geometry)

### What We Measure

**Trajectory**: Sequence of activation vectors as information flows through layers

```
Input → Layer 0 → Layer 2 → ... → Layer 30 → Output
         ↓         ↓              ↓
       h_0       h_2            h_30
       
Trajectory = [h_0, h_2, h_4, ..., h_30]  # Shape: (16, 4096)
```

**Path Signature**: Captures geometric properties of this path
- Curvature (how much the path bends)
- Winding (how much the path twists)
- Self-intersection (does path cross itself)

### Why Trajectories (Not Static States)

**Reasoning is a process**, not a snapshot:
- Each layer transforms the representation
- The *flow* through layers IS the computation
- Static analysis (Phase 1) misses temporal dynamics

**Analogy**: 
- Static analysis = taking a photo of a car
- Trajectory analysis = recording the car's path through space

### Critique: Is This Meaningful?

**Strengths**:
- Captures temporal dynamics of computation
- Path signatures are mathematically principled (reparameterization-invariant)
- Enables causal interventions (steering)

**Weaknesses**:
- Decision may be made early (Afzal et al. 2025) → later trajectory is just elaboration
- Post-hoc rationalization (Turpin et al. 2023) → trajectory may not reflect true reasoning
- High-dimensional (4096 dims) → need dimensionality reduction → introduces artifacts

### Alternative Methods (If Trajectory Geometry Fails)

**Alternative 1: Early-Layer Probing**
- Focus on layers 0-15 (where decision forms)
- Ignore late layers (elaboration)
- Simpler, more direct

**Alternative 2: Attention Pattern Analysis**
- Analyze attention weights, not activations
- May capture reasoning structure more directly
- See: "Circuit Tracing" (Anthropic 2025)

**Alternative 3: Gradient-Based Attribution**
- Use gradients to identify which activations matter for correctness
- More causal than correlational probing
- Computationally expensive

**Alternative 4: Semantic Entropy (Farquhar et al. 2024)**
- Sample multiple outputs, cluster by meaning
- Entropy over clusters predicts correctness
- Already validated, but expensive

---

## 🚀 Next Steps

**Immediate** (Phase 2):
1. Run cleanup: `./scripts/cleanup_smallworld.sh`
2. Collect GSM8K trajectories: `python scripts/collect_trajectories_with_labels.py --task gsm8k`
3. Monitor correctness rates (aim for 30-70% correct per model)

**After Phase 2** (Phase 3):
1. Compute path signatures
2. Test H1 (within-domain classification)
3. Test H2 (cross-domain transfer) - **THE CRITICAL TEST**

**If H2 succeeds** (Phase 4):
1. Implement trajectory steering
2. Test on held-out problems
3. Measure accuracy improvement

**Final** (Phase 5):
1. Write paper
2. Release code and data
3. Submit to conference

---

## 📚 Key References

### Theoretical Framework (New)

- **Allen-Zhu & Li (2024)**: Physics of Language Models — everything is interpolation
- **Merullo et al. (2025)**: Loss curvature separates memorization from generalization
- **Ren & Liu (2026)**: HRM analysis — attractor dynamics, "grokking" transitions
- **Shai et al. (2024)**: Belief state geometry in residual stream
- **Bigelow et al. (2025)**: Belief dynamics unify ICL and activation steering
- **Gosztolai & Bhattacharyya (MARBLE)**: Vector field decomposition for neural dynamics

### Supporting Evidence

- Zhang et al. (2025): Hidden states predict correctness
- Marks & Tegmark (2023): Truth has geometric structure
- Turner et al. (2023): Activation steering works
- Hosseini & Fedorenko (2023): Trajectories straighten with success

### Critical Challenges

- Turpin et al. (2023): CoT can be unfaithful
- Afzal et al. (2025): Decision before reasoning
- Hewitt & Liang (2019): Probes need control tasks
- Ley et al. (2024): Faithfulness interventions fail to transfer

---

## ⚡ Quick Commands

```bash
# Cleanup disk space
./scripts/cleanup_smallworld.sh

# Collect trajectories (Phase 2)
python scripts/collect_trajectories_with_labels.py

# Run geometric analysis (Phase 1)
python scripts/run_analysis.py

# Check collection status
ls -lh data/trajectories/

# Monitor GPU
nvidia-smi
```

---

## ⚙️ GPU Optimization Lessons (Phase 2 - 2026-01-18)

### Discovery: Batched Collection Was GPU-Idle 80% of the Time

During Phase 2 LogiQA collection, we discovered that our "batched" collection script (`collect_logiqa_batched.py`) had **0% GPU utilization** despite running on RTX 4090s. Investigation revealed **4 critical bottlenecks** causing the GPU to sit idle while the CPU worked.

#### Performance Impact

| Metric | Sequential | Batched (broken) | Optimized (target) |
|--------|-----------|------------------|-------------------|
| Time per sample | 90s | 40-50s (2x) | 15-20s (4-5x) |
| GPU utilization | 60-70% | **0-10%** | 90-95% |
| Total time (500 samples) | 12.5 hrs | 5.5 hrs | **2-3 hrs** |

### The 4 Bottlenecks

#### Bottleneck 1: GPU→CPU Transfer During Generation
**Location**: [scripts/collection/collect_logiqa_batched.py:124](../scripts/collection/collect_logiqa_batched.py#L124)

```python
# BEFORE (blocking transfer in hook)
def hook(module, input, output):
    layer_outputs[layer_idx].append(hidden.detach().cpu())  # ❌ BLOCKS GPU!
```

**Impact**: GPU waits for CPU transfer during EVERY layer's forward pass (~16 layers × 512 tokens = 8,192 blocking transfers per batch)

**Fix**:
```python
# AFTER (keep on GPU, transfer once at end)
def hook(module, input, output):
    layer_outputs[layer_idx].append(hidden.detach())  # ✓ Stay on GPU

# Transfer to CPU once after generation completes
activations_cpu = {k: [t.cpu() for t in v] for k, v in layer_outputs.items()}
```

**Speedup**: ~2x faster generation

---

#### Bottleneck 2: Individual Forward Passes for Activation Collection
**Location**: [scripts/collection/collect_logiqa_batched.py:159-187](../scripts/collection/collect_logiqa_batched.py#L159-L187)

```python
# BEFORE (sequential forward passes)
for i in range(batch_size):  # ❌ 4 sequential passes!
    full_text = prompts[i] + outputs[i]
    sample_inputs = self.tokenizer(full_text, ...).to(self.model.device)
    with torch.no_grad():
        _ = self.model(**sample_inputs)  # 30s each × 4 = 120s
    # Extract activations...
```

**Impact**: After fast batched generation (~40s), we do 4 sequential forward passes (~30s each = 120s). This is **75% of batch time**!

**Fix**:
```python
# AFTER (batched forward pass with padding)
# Tokenize all prompt+output pairs together
full_texts = [prompts[i] + outputs[i] for i in range(batch_size)]
batch_inputs = self.tokenizer(
    full_texts,
    return_tensors='pt',
    padding=True,  # Pad to max length in batch
    truncation=True,
).to(self.model.device)

# Single batched forward pass
with torch.no_grad():
    _ = self.model(**batch_inputs)  # 35s for all 4 samples

# Separate activations by sample using attention_mask
for i in range(batch_size):
    # Find actual sequence length (non-padding tokens)
    seq_len = batch_inputs['attention_mask'][i].sum().item()
    # Extract trajectory for this sample
    trajectory = layer_outputs[:, i, :seq_len, :]  # [layers, seq_len, d_model]
```

**Speedup**: 120s → 35s (~3.4x faster)

---

#### Bottleneck 3: HDF5 Writes Blocking Next Batch
**Location**: [scripts/collection/collect_logiqa_batched.py:268-292](../scripts/collection/collect_logiqa_batched.py#L268-L292)

```python
# BEFORE (synchronous I/O blocks GPU)
for batch_start in pbar:
    outputs, trajectories = collector.generate_batch(...)  # GPU working

    # Write to disk - GPU SITS IDLE for ~20-30s!
    for i in range(actual_batch_size):  # ❌ Blocking I/O
        f['trajectories'][idx] = trajectories[i]
        f['model_outputs'][idx] = outputs[i]
```

**Impact**: GPU idle for 20-30s per batch (~15-20% of total time)

**Fix**:
```python
# AFTER (async I/O with threading)
import queue
import threading

write_queue = queue.Queue(maxsize=2)  # Buffer 2 batches

def writer_thread():
    while True:
        batch_data = write_queue.get()
        if batch_data is None:  # Poison pill
            break
        # Write to HDF5
        for idx, trajectory, output in batch_data:
            f['trajectories'][idx] = trajectory
            f['model_outputs'][idx] = output
        write_queue.task_done()

# Start writer thread
writer = threading.Thread(target=writer_thread, daemon=True)
writer.start()

# Main loop - pipeline generation and writing
for batch_start in pbar:
    outputs, trajectories = collector.generate_batch(...)  # GPU working
    write_queue.put(batch_data)  # Non-blocking (unless queue full)
    # GPU immediately starts next batch!

# Cleanup
write_queue.put(None)
writer.join()
```

**Speedup**: Eliminates 20-30s idle time per batch by pipelining

---

#### Bottleneck 4: Increasing Batch Times (Memory Fragmentation)
**Observation**: Batch times increased from 160s → 259s over 5 batches

**Cause**: Memory not being freed between batches, causing fragmentation or cache effects

**Fix**:
```python
# Add after each batch completes
torch.cuda.empty_cache()  # Free unused cached memory
gc.collect()  # Python garbage collection

# Also add to hook cleanup
finally:
    for handle in handles:
        handle.remove()
    layer_outputs.clear()
    torch.cuda.empty_cache()  # ✓ Clean GPU memory
```

**Impact**: Prevents slowdown over time

---

### Combined Optimization: Pipelined Execution

**Optimal flow** (all 4 fixes applied):

```
Time →
GPU:    [Gen B1][Collect B1][Gen B2][Collect B2][Gen B3][Collect B3]
CPU:                        [Write B1]          [Write B2]
Memory:         [Clean]              [Clean]             [Clean]
        ↑               ↑              ↑
      GPU busy       GPU busy       GPU busy (NO IDLE TIME!)
```

**vs Current flow**:

```
GPU:    [Gen B1] IDLE (CPU transfer) [Fwd 1][Fwd 2][Fwd 3][Fwd 4] IDLE (HDF5 write)
                 ^^^^                                               ^^^^
                 25% idle                                          15% idle
```

---

### Implementation: `collect_logiqa_optimized.py`

New optimized script with all 4 fixes applied:
- **Location**: [scripts/collection/collect_logiqa_optimized.py](../scripts/collection/collect_logiqa_optimized.py)
- **Expected speedup**: 4-5x vs sequential, 2-3x vs batched
- **Expected time**: 2-3 hours for 500 samples (vs 12.5 hrs sequential, 5.5 hrs batched)

**Key changes**:
1. ✓ Keep tensors on GPU until final transfer
2. ✓ Batched activation collection with padding
3. ✓ Async HDF5 writes with threading
4. ✓ Explicit memory cleanup

---

### Lessons for Phase 4 (Trajectory Steering)

Phase 4 will require **batched interventions** on activation trajectories. The same principles apply:

1. **Keep interventions on GPU** - Don't transfer tensors to CPU during forward pass
2. **Batch the steering** - Steer multiple samples simultaneously
3. **Pipeline generation** - While evaluating batch N, steer batch N+1
4. **Memory management** - Clean cache after each batch

---

## 🎓 For Future Reference

**If you're an LLM working on this project**:
1. Start with `main/paper/RESEARCH_PLAN.md`
2. Check current phase plan (e.g., `PHASE2_DETAILED_PLAN.md`)
3. Read `main/paper/claude.md` for detailed instructions
4. **Update this file** when you create new files

**If you're a human reading this**:
1. See `main/paper/RESEARCH_PLAN.md` for the full research plan
2. See `main/paper/LITERATURE_REVIEW_SHORT.md` for literature context
3. See current phase plan for execution details
4. See `main/claude.md` for development workflow
