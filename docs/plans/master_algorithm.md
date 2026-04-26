# ManiVer: Master Algorithm — Complete Audit & Documentation

**Project**: Manifold Verification — Geometric Signatures of Correct Computation in LLMs
**Last Updated**: 2026-02-09
**Scope**: Comprehensive audit of every file, algorithm, finding, and limitation in the project

---

## Table of Contents

1. [Purpose & Core Question](#1-purpose--core-question)
2. [Theoretical Framework](#2-theoretical-framework)
3. [Hypotheses & Status](#3-hypotheses--current-status)
4. [Phase-by-Phase Algorithm](#4-phase-by-phase-algorithm)
5. [Complete File Map](#5-complete-file-map-with-version-history)
6. [Core Library (src/)](#6-core-library-src)
7. [Collection Scripts — Evolution & Lineage](#7-collection-scripts--evolution--lineage)
8. [Analysis Scripts — Algorithms & Limitations](#8-analysis-scripts--algorithms--limitations)
9. [Experiments — Self-Contained Studies](#9-experiments--self-contained-studies)
10. [Results — What We Found](#10-results--what-we-found)
11. [Methodological Critiques](#11-methodological-critiques)
12. [GPU Optimization Lessons](#12-gpu-optimization-lessons)
13. [Decision Tree & Next Steps](#13-decision-tree--next-steps)
14. [References](#14-references)

---

## 1. Purpose & Core Question

**One-line**: Test whether correct solutions have distinguishable dynamical signatures in activation trajectories, and whether these signatures share structure across verifiable domains (math → code → logic).

**Motivation**: Phase 1 showed RLVR and SFT models have dramatically different static geometry (98% vs 52% subspace preservation). But static analysis doesn't tell us whether geometry correlates with *correctness*. We need to track activations during computation and label them with ground-truth correctness.

**What we avoid**: We don't claim to detect "reasoning" — we characterize the geometry of correct vs incorrect solutions. This distinction matters because:
- "Reasoning" implies cognitive process (unfalsifiable at this level)
- "Correct solution geometry" is measurable (trajectories + labels)
- Confounds (difficulty, length, format) must be controlled before claiming "reasoning"

---

## 2. Theoretical Framework

We adopt an **interpolation-centric view** (Allen-Zhu & Li, 2024):

- Transformers compute smooth functions over representation manifolds
- There is no "reasoning mode" vs "recall mode" — all is interpolation
- What differs is the *region* and *dynamics* of manifold traversal

| Concept | Source | Our Application | Status |
|---------|--------|-----------------|--------|
| Everything is interpolation | Allen-Zhu & Li (2024) | Don't detect "reasoning" — characterize interpolation geometry | Framing |
| Curvature regimes | Merullo et al. (2025) | High-SV = distributed; low-SV = localized (proxy for curvature) | ❌ Architectural artifact |
| Attractor dynamics | Ren & Liu (2026) | Correct solutions find right attractors; incorrect get trapped | Not tested |
| Belief state geometry | Shai et al. (2024) | Residual stream represents belief states | ✅ Partial support |
| Menger curvature | Zhou et al. (Oct 2025) | Curvature captures logical structure beyond surface semantics | ❌ Architectural in our data |
| Decision-before-reasoning | Afzal et al. (2025) | Decision made early; later CoT is elaboration | ✅ Confirmed (early tokens) |
| CoT unfaithfulness | Turpin et al. (2023) | Trajectories may reflect post-hoc rationalization | Unknown |

**Critical caveat**: Multiple theoretical connections were adopted *before* empirical testing. Several (curvature regimes, Menger curvature, Lyapunov stability) failed empirically on our architecture (OLMo-3). The framework remains useful as framing but should not be taken as validated.

---

## 3. Hypotheses & Current Status

| Hypothesis | Statement | Success Criterion | Status | Evidence |
|------------|-----------|-------------------|--------|----------|
| **H1** | Correct vs incorrect trajectories are distinguishable | >65% accuracy | ✅ **TRUE** | AUC 0.68-0.75 via linear probe |
| **H2** | Signatures transfer across domains | >55% transfer accuracy | ⚠️ **PARTIAL** | SFT: cos=0.355, p<0.001; RL-Zero: cos=0.098, no transfer |
| **H3** | Detector works on non-verifiable domains | r > 0.25 with human judgment | ⏳ Not tested | Requires human annotation |
| **H4** | Steering interventions improve accuracy | >2% improvement | ⏳ Not tested | Activation patching shows promise (91% recovery) |
| **H5** | Correct solutions have more stable dynamics | Lyapunov exponent difference | ❌ **FAILED** | True Jacobian λ ≈ 0 for all (orthogonality); linear probe beats all dynamical measures |

### Key surprise findings

1. **Static geometry beats dynamics**: A simple linear probe on mean activation (AUC 0.75) outperforms every dynamical measure we tested (Lyapunov, Menger curvature, velocity, acceleration, path signatures)
2. **SFT creates domain-general error patterns; RL-Zero does not**: Error direction alignment cos=0.355 (SFT) vs 0.098 (RL-Zero). Cross-domain transfer only works for SFT
3. **Correctness encoded early**: Signal peaks at tokens 2-8% into sequence (problem statement), not at answer tokens
4. **RL-Zero preserves base geometry**: CKA≈0.995, eigenvector preservation 90-96%. SFT reshapes dramatically (eigenvector preservation 18-39%)
5. **Distillation ≠ native RLVR**: DeepSeek-R1-Distill has sharp L18 spike; OLMo RL-Zero has gradual ramp. Fundamentally different internal mechanisms

---

## 4. Phase-by-Phase Algorithm

### Phase 1: Static Geometry Baseline (✅ Complete)

**Purpose**: Establish that training method affects activation geometry.

```
Input: 4 models (Base, SFT, RL-Zero, Think) × 3 tasks
Process: Collect last-token activations at all 32 layers
Measure: Effective rank, spectral decay, subspace preservation, CKA
Output: RLVR preserves base geometry (98.6% ± 1.1%), SFT reshapes it (52.4% ± 12.6%)
```

**Key result**: Cohen's d > 4.0, p < 10⁻¹⁷ for all comparisons. Effect is real but tells us nothing about correctness.

**Limitation**: Static analysis. Geometry differs by training, but we don't know if it correlates with getting answers right.

### Phase 2: Trajectory Collection with Labels (🔄 92% Complete)

**Purpose**: Collect activation trajectories during generation, labeled with correctness.

```
Input: 4 models × 3 tasks × 500 samples (164 for HumanEval)
Process:
  1. Generate full model response (up to 512 new tokens)
  2. Extract ground-truth answer, check correctness
  3. Collect activations at even layers [0, 2, ..., 30] = 16 layers
  4. Store trajectory shape: (512 tokens, 16 layers, 4096 dims)
  5. Checkpoint every 25 samples
Output: 12 HDF5 files with trajectories + is_correct labels (~52 GB)
```

**Status**: 11/12 files collected and uploaded to B2. Missing: `olmo3_base/gsm8k` (corrupted HDF5 gzip filter error).

**Correctness checking**:
- GSM8K: Extract `#### <number>` via regex, exact float match (1e-6 tolerance)
- LogiQA: Extract A/B/C/D letter, case-insensitive match
- HumanEval: Syntax check only (`compile()`) — **NOT full test execution**

**Known issue with HumanEval correctness**: We only compile-check, not run test cases. This means some "correct" code may not actually pass tests, and some "incorrect" code (syntax errors) might be close to correct. This introduces label noise.

### Phase 3: Cross-Domain Transfer & Dynamical Analysis (🔄 Mostly Complete)

**Purpose**: Test H2 (cross-domain transfer) and characterize dynamical properties.

**What was planned vs what was done**:

| Planned Method | Implemented? | Result |
|----------------|-------------|--------|
| Path signatures (signatory, depth 3) | ✅ Yes | AUC 0.78 for SFT GSM8K→HumanEval; weak elsewhere |
| Error-detection direction (Wynroe) | ✅ Yes | d=1.70 exists, but correlational |
| Activation patching (Wynroe, causal) | ✅ Yes | 91% recovery at best layer |
| Lyapunov exponents (Frobenius) | ✅ Yes | ❌ FAILED (architectural) |
| Lyapunov exponents (true Jacobian) | ✅ Yes | ❌ FAILED (orthogonality → λ ≈ 0) |
| Directional Lyapunov | ✅ Yes | ❌ FAILED (data leakage → null with CV) |
| Menger curvature | ✅ Yes | ❌ Architectural (r≈0.999 correct vs incorrect) |
| Vector field (Helmholtz decomposition) | ❌ Not tested | — |
| Attractor analysis (K-means) | ✅ Yes | High purity but class imbalance confound |
| Belief state tracking | ✅ Yes | RL-Zero=smooth, SFT=jumpy (opposite patterns) |
| Per-sample CKA | ✅ Yes | ✅ d=-0.64 (correct solutions restructure more) |
| Cross-model alignment | ✅ Yes | ✅ RL-Zero preserves base (CKA=0.995) |

### Phase 4: Steering Intervention (⏳ Not Started)

**Purpose**: Causally test whether trajectory geometry matters by steering activations.

**Planned methods**:
1. Subspace projection: Learn PCA manifold from correct trajectories, project activations onto it
2. Activation addition: steering_vector = mean(correct) - mean(incorrect)
3. Optimal transport: Map incorrect → correct distribution

**Depends on**: Phase 3 identifying features worth steering on.

### Phase 5: Write-Up (⏳ Not Started)

**Purpose**: Publication. Target depends on results:
- If H1+H2+H4 succeed: NeurIPS/ICML/ICLR main
- If H1+H2 succeed, H4 fails: NeurIPS workshop
- If only H1 succeeds: ACL/EMNLP

---

## 5. Complete File Map with Version History

### Documentation (`docs/`)

| File | Purpose | Status | Notes |
|------|---------|--------|-------|
| `docs/plans/PHASE1_DETAILED_PLAN.md` | Phase 1 plan + results | ✅ Complete | 150 lines |
| `docs/plans/PHASE2_DETAILED_PLAN.md` | Phase 2 plan (current) | 🔄 Active | 350 lines, supersedes PHASE2_PLAN.md |
| `docs/plans/PHASE2_PLAN.md` | Original Phase 2 plan | 📁 Superseded | Less detailed; kept for reference |
| `docs/plans/PHASE2_EXECUTION_PLAN.md` | Production run guide for vast.ai | 📁 Reference | 460 lines, deployment-specific |
| `docs/plans/PHASE3_DETAILED_PLAN.md` | Phase 3 plan (comprehensive) | 🔄 Active | 1100+ lines, most detailed |
| `docs/plans/PHASE4_DETAILED_PLAN.md` | Phase 4 steering plan | ⏳ Pending | 430 lines |
| `docs/plans/PHASE5_DETAILED_PLAN.md` | Write-up plan | ⏳ Pending | 460 lines |
| `docs/plans/phase1_implementation_plan.md` | Original Phase 1 plan | 📁 Superseded | Has results summary |
| `docs/plans/archive_transfer_correlation_plan.md` | Old Phase 2-3 approach | 🗄️ Archived | Completely superseded |
| `docs/plans/master_algorithm.md` | **THIS FILE** | 🔄 Active | Complete project reference |
| `docs/paper/RESEARCH_PLAN.md` | Main hypotheses + experimental design | 🔄 Active | 711 lines |
| `docs/paper/LITERATURE_REVIEW_SHORT.md` | Concise lit review per hypothesis | ✅ Complete | 240 lines |
| `docs/paper/LITERATURE_REVIEW_LONG.md` | Extended analysis | ✅ Complete | 732 lines |
| `docs/paper/TRAJECTORY_GEOMETRY_CRITIQUE.md` | Self-critique of approach | ✅ Complete | 552 lines, valuable for honesty |
| `docs/paper/geometric_compression_research_plan.md` | Background (NOT main focus) | 📁 Context only | 1372 lines |
| `docs/guides/PHASE2_PIPELINE.md` | Pipeline guide | ✅ Complete | |
| `docs/guides/SLURM_QUICKSTART.md` | SLURM quick reference | ✅ Complete | |
| `docs/guides/SLURM_CLUSTER_GUIDE.md` | Detailed SLURM guide | ✅ Complete | |
| `docs/guides/VLLM_GPU_GUIDE.md` | vLLM compatibility | ✅ Complete | |
| `docs/guides/B2_SETUP.md` | Backblaze B2 setup | ✅ Complete | |
| `docs/guides/B2_QUICKSTART.md` | B2 quick reference | ✅ Complete | |

### Core Library (`src/`)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/activation_collector.py` | Hook-based activation collection | ~410 | ✅ Production |
| `src/geometric_measures.py` | SVD-based geometry measures | ~325 | ✅ Production |
| `src/task_data.py` | Dataset loading (GSM8K, HumanEval, LogiQA) | ~474 | ✅ Production |
| `src/checkpointing.py` | Fault-tolerant checkpoint system | ~203 | ✅ Production |

### Collection Scripts (`scripts/collection/`)

| File | Purpose | Status | Prior Version Of |
|------|---------|--------|------------------|
| `collect_trajectories_with_labels.py` | **CANONICAL** Phase 2 pipeline | ✅ Production (11/12 done) | — |
| `collect_single_model.py` | Parallel wrapper for above | ✅ Production | — |
| `collect_8shot_trajectories.py` | 8-shot extension | ⚠️ Partial | — |
| `collect_logiqa_optimized.py` | Optimized (4 bottleneck fixes) | ✅ Production | collect_logiqa_batched.py |
| `collect_logiqa_vllm_fully_optimized.py` | H100-optimized (all 6 fixes) | ⚠️ Untested on SLURM | collect_logiqa_vllm.py |
| `collect_generation_trajectories.py` | Per-step generation dynamics | 🔬 Experimental | — |
| `collect_logiqa_batched.py` | Naive batching (baseline) | ❌ Deprecated | — |
| `collect_logiqa_vllm.py` | vLLM attempt (memory fail) | ❌ Dead code | — |
| `collect_activations.py` | Phase 1 static collection | 📁 Superseded | — |
| `collect_trajectories_half_layers.py` | Even-layer exploration | 📁 Superseded | — |
| `collect_single_logiqa.py` | Single-model LogiQA test | 🔧 Testing | — |
| `collect_missing_logiqa.py` | Gap recollection | 🔧 One-time use | — |
| `test_logiqa_collection.py` | N=3 validation | 🔧 Testing | — |

**Version lineage**:
```
Phase 1:
  collect_activations.py (static, no generation)
    └→ collect_trajectories_half_layers.py (explore even layers)

Phase 2 (canonical):
  collect_trajectories_with_labels.py (main pipeline)
    ├→ collect_single_model.py (parallel wrapper)
    ├→ collect_single_logiqa.py (LogiQA-only test)
    ├→ collect_missing_logiqa.py (gap fill)
    └→ collect_8shot_trajectories.py (8-shot extension)

Optimization attempts:
  collect_logiqa_batched.py (naive, GPU-idle 80%)
    ├→ collect_logiqa_optimized.py (4 fixes, 4-5x speedup)
    └→ collect_logiqa_vllm.py (abandoned, 28GB needed)
         └→ collect_logiqa_vllm_fully_optimized.py (H100, dual-model)

Phase 3:
  collect_generation_trajectories.py (per-token dynamics)
```

### Analysis Scripts (`scripts/analysis/`)

| File | Algorithm | H-Connection | Status |
|------|-----------|-------------|--------|
| `h1_h2_classifier.py` | 124-feature logistic regression, 5-fold CV | H1, H2 | ✅ Complete |
| `empirical_jacobian_lyapunov.py` | True regression Jacobian + Lyapunov | H5 | ✅ Complete (null result) |
| `full_lyapunov_analysis.py` | 3-method Lyapunov (SVD, directional, spectrum) | H5 | ⚠️ Data leakage found |
| `phase3_dynamical_analysis.py` | Error direction + Menger + Lyapunov + attractor | H1, H2, H5 | ✅ Complete |
| `path_signature_analysis.py` | Signatory depth-3 path signatures | H2 | ✅ Complete |
| `jacobian_diagnostic.py` | Diagnostic: delta vs true Jacobian validity | H5 | ✅ Complete (identified orthogonality) |
| `cross_model_alignment.py` | CKA, eigenvector correspondence, Procrustes | — | ✅ Complete |
| `cross_domain_all_models.py` | Error direction alignment across tasks | H2 | ✅ Complete |
| `cka_deep_analysis.py` | Per-sample CKA at each layer | H1 | ✅ Complete |
| `pair_trajectory_divergence.py` | Base↔RL-Zero↔SFT layer comparison | — | ✅ Complete |
| `curvature_and_stats.py` | Perturbation-based curvature proxy | — | ✅ Complete (Phase 1) |
| `run_analysis.py` | Phase 1 geometric analysis | — | ✅ Complete (Phase 1) |
| `check_layer_smoothness.py` | Validate even-layer subsampling | — | ✅ Complete |
| `verify_pipeline.py` | Pipeline test | — | 🔧 Testing |

### Experiments (`experiments/`)

| Experiment | Directory | Sub-parts | Status |
|------------|-----------|-----------|--------|
| Aha Moment (error detection) | `experiments/aha_moment/` | A, A', B, C, D | ✅ Mostly complete |
| SVD Reasoning Separability | `experiments/svd_reasoning_separability/` | 1 script | ✅ Complete |
| Generation Dynamics | `experiments/generation_dynamics/` | Collection done, analysis pending | 🔄 In progress |
| Belief Tracking | `experiments/belief_tracking/` | Bootstrap probe | 🔬 Early stage |

### Results (`results/`)

| File | Contents | Key Finding |
|------|----------|-------------|
| `PHASE3_COMPLETE_FINDINGS.md` | All Phase 3 results (P1-P9 positive, N1-N11 negative) | Static geometry beats dynamics |
| `PHASE3_H1H2_FINDINGS.md` | H1/H2 results + belief tracking + cross-model alignment | SFT transfers, RL-Zero doesn't |
| `DYNAMICAL_ANALYSIS_FINDINGS.md` | Token-level and dynamical analysis | Early tokens carry correctness signal |
| `phase1_summary.json` | Phase 1 numeric results | — |
| `statistical_tests.json` | Phase 1 statistical tests | — |

---

## 6. Core Library (src/)

### `src/activation_collector.py` (~410 lines)

**What it does**: Collects transformer activations using PyTorch hooks during forward pass.

**Algorithm**:
1. Register `register_forward_hook` on specified layers
2. Hook captures output tensor: `hidden.detach()` (stays on GPU)
3. After forward pass, transfer all collected tensors to CPU
4. Store as float16 in HDF5 with gzip compression

**Design decisions**:
- Float16 storage (halves size, sufficient for SVD/probing)
- Even layers only [0,2,...,30] = 16 layers (validated: max consecutive jump 0.09%)
- Residual stream only (not attention/MLP separately) — 3x storage reduction
- Supports TransformerLens backend (recommended) or raw PyTorch hooks

**Canonical**: Hook-based activation extraction is the standard approach in interpretability (used by TransformerLens, Baukit, nnsight).

**Limitations**:
- Hook overhead ~5-10% (acceptable)
- Sequence length fixed at collection time
- No validation for NaN/Inf values in collected activations

### `src/geometric_measures.py` (~325 lines)

**What it does**: Computes SVD-based geometric properties of activation manifolds.

**Measures** (all standard linear algebra):

| Measure | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| Effective rank | exp(H(σ̂)) where σ̂ = σ/Σσ | [1, min(n,d)] | Low = compressed, High = distributed |
| Spectral decay | Power-law fit σᵢ ∝ i⁻ᵅ | α > 0 | Higher α = faster decay = more concentrated |
| Subspace preservation | Principal angles between top-k subspaces | [0, 1] | 1 = identical subspaces |
| Participation ratio | (Σσᵢ)² / Σ(σᵢ²) | [1, d] | More robust to noise than effective rank |
| Stable rank | Σ(σᵢ²) / max(σᵢ)² | [1, d] | Resistant to outliers |
| CKA | Gram matrix similarity (scale-invariant) | [0, 1] | 1 = identical representations |

**Canonical**: All measures are textbook (scipy.linalg.svd, scipy.linalg.subspace_angles). Implementation is sound.

**Limitation**: All measures are linear. If correctness boundaries are nonlinear, these will miss them. This turned out to matter — linear probes work, but dynamical (nonlinear) measures failed.

### `src/task_data.py` (~474 lines)

**What it does**: Unified interface for loading GSM8K, HumanEval, LogiQA with proper prompt formatting.

**Prompt formats**:
- GSM8K: 8-shot exemplars with CoT reasoning (matches lm-evaluation-harness)
- HumanEval: 0-shot code completion
- LogiQA: 8-shot exemplars with reasoning + A/B/C/D answer

**Correctness checking** (implemented here and in collection scripts):

| Task | Extraction | Comparison | Limitation |
|------|-----------|------------|------------|
| GSM8K | Regex: `####\s*(-?[\d,\.]+)` | Float match (1e-6 tolerance) | Misses non-standard formats |
| LogiQA | Regex: multiple patterns for A/B/C/D | Case-insensitive string match | Can match wrong letter if multiple appear |
| HumanEval | `compile(code, '<string>', 'exec')` | Syntax check only | **NOT full test execution — label noise** |

**Bug risk**: HumanEval uses `exec()` without sandboxing. Not a security risk in practice (we're running our own generated code) but theoretically unsafe.

### `src/checkpointing.py` (~203 lines)

**What it does**: JSON-based checkpoint system for crash recovery.

**Saves**: model, task, samples_completed, total_samples, output_file, timestamp, status.
**Resumes**: Detects existing checkpoint, skips completed samples.
**Limitation**: JSON-only (not streaming). If process dies during checkpoint write, file may be corrupted. Also, partial HDF5 files from crashed runs require manual merge.

---

## 7. Collection Scripts — Evolution & Lineage

### The Canonical Pipeline: `collect_trajectories_with_labels.py`

This is the script that produced 11/12 Phase 2 files (~52GB).

**Algorithm (per sample)**:
```
1. Load model with Flash Attention 2 (fallback: standard attention)
2. For each sample (prompt, ground_truth):
   a. Register hooks on layers [0, 2, 4, ..., 30]
   b. Generate response (max 512 new tokens)
   c. Hooks capture hidden states at each layer → shape: (seq_len, 16, 4096)
   d. Pad/truncate to MAX_SEQ_LEN=512
   e. Check correctness against ground_truth
   f. Store trajectory, sequence_length, is_correct, prompt, output, ground_truth in HDF5
   g. Checkpoint every 25 samples
3. Final checkpoint, cleanup
```

**HDF5 schema**:
```
/trajectories          (500, 512, 16, 4096)   float16  gzip-4
/sequence_lengths      (500,)                 int32
/is_correct            (500,)                 bool
/prompts               (500,)                 string
/model_outputs         (500,)                 string (max 10KB)
/ground_truth          (500,)                 string (max 5KB)

Attributes: model, task, n_samples, layers, d_model, collection_date
```

**Why batch size = 1**: Sequential processing avoids model replication overhead and simplifies hook management. Trade-off: slower but more reliable. The optimization variants (below) batch for speed.

### Optimization Evolution

The project went through several optimization attempts after discovering that the batched script had 0-10% GPU utilization:

**Generation 1**: `collect_logiqa_batched.py` (Naive batching)
- Batches generation but does sequential forward passes for activation collection
- Result: 2x speedup vs sequential, but GPU still idle 80%

**Generation 2**: `collect_logiqa_optimized.py` (4 fixes)
- Fix 1: Keep tensors on GPU (`detach()` not `detach().cpu()`)
- Fix 2: Batched activation collection (single forward pass)
- Fix 3: Async HDF5 I/O (background writer thread)
- Fix 4: Memory cleanup (`torch.cuda.empty_cache()` + `gc.collect()`)
- Result: 4-5x speedup vs sequential

**Generation 3**: `collect_logiqa_vllm.py` (vLLM attempt)
- Dual-model: vLLM for generation, HF for activations
- **Abandoned**: Requires 28GB (vLLM 14GB + HF 14GB), doesn't fit on 24GB GPUs

**Generation 4**: `collect_logiqa_vllm_fully_optimized.py` (H100 target)
- Same dual-model approach but designed for 80GB H100
- All 6 bottlenecks fixed
- Expected: 8-10x speedup vs sequential
- **Status**: Written but never deployed on SLURM

### Generation Dynamics Collection: `collect_generation_trajectories.py`

A fundamentally different collection approach for the `experiments/generation_dynamics/` study. Instead of collecting at even layers for the full prompt, this collects **per generated token**:

**Per token, collects**:
- Hidden states: 16 layers (even layers 0-30)
- Attention patterns: 8 layers × 8 heads
- Top-100 token probabilities + entropy

**Purpose**: Investigate where RL-Zero's 7% accuracy improvement comes from, since input-time trajectories are nearly identical (cos_sim 0.995).

**Status**: Data collected for olmo3_base (495/500 GSM8K, 164 HumanEval, 500 LogiQA). RL-Zero collection in progress.

---

## 8. Analysis Scripts — Algorithms & Limitations

### 8.1 H1/H2 Classifier (`h1_h2_classifier.py`)

**Algorithm**: Extract 124 features per trajectory, train logistic regression.

**Features** (per layer, 16 layers):
1. Mean activation norm (16 features)
2. Activation variance (16)
3. Velocity: ||h_{l+1} - h_l|| (15)
4. Curvature: ||v_{l+1} - v_l|| (14)
5. Cosine similarity between consecutive layers (15)
6. Activation entropy (16)
7. First-last token difference (16)
8. Max activation (16)

**Classifier**: StandardScaler → LogisticRegression(C=0.1, class_weight='balanced') → 5-fold stratified CV

**Result**: AUC 0.68-0.75 within-domain (H1 confirmed); ≈52% cross-domain (H2 fails for classification, though direction alignment works for SFT).

**Limitation**:
- C=0.1 regularization never justified or ablated
- 124 features on 500 samples risks overfitting (though CV mitigates)
- Assumes linear separability
- HumanEval base model: 3.8% correct → ~2 correct samples per fold

**Canonical**: Standard ML pipeline. Feature design is heuristic but reasonable.

### 8.2 True Jacobian Lyapunov (`empirical_jacobian_lyapunov.py`)

**Algorithm**: Compute true layer transition Jacobian via regularized regression.

**Mathematical formulation**:
```
For consecutive layers X_l ∈ ℝ^{n×d}, X_{l+1} ∈ ℝ^{n×d}:

1. Center: X̄_l = X_l - mean(X_l), X̄_{l+1} = X_{l+1} - mean(X_{l+1})
2. SVD of X̄_l: U Σ V^T
3. Regularized pseudo-inverse: Σ_reg = Σ / (Σ² + λ), λ = 1e-6
4. Jacobian: J^T = V Σ_reg^{-1} U^T X̄_{l+1}
5. SVD of J: S_J = singular values
6. Lyapunov exponents: λ_i = log(S_J[i])
7. Max Lyapunov = log(S_J[0])
```

**Result**: No significant difference between correct/incorrect (all d ≈ 0.15-0.28, p > 0.3). **H5 FAILED**.

**Why it failed** (identified by `jacobian_diagnostic.py`):
- cos(X_l, X_{l+1}) ≈ 0.10-0.12 for all OLMo-3 layers
- Layer transitions are approximately orthogonal rotations
- Rotation matrices have singular values ≈ 1
- log(1) = 0 → all Lyapunov exponents ≈ 0
- **This is an architectural property of OLMo-3**, not a measurement error

**Canonical**: Regression-based Jacobian estimation is standard numerical analysis. Application to transformers is novel but not inappropriate — the null result is valid.

**Limitation**: Only tested on OLMo-3. Other architectures (Llama, Mistral, GPT) may have different orthogonality properties.

### 8.3 Fast Lyapunov (`full_lyapunov_analysis.py`)

**Three methods implemented, all failed**:

**Method 1: SVD of layer delta** (Ad-hoc)
```
delta = X_{l+1} - X_l
SVD(delta) → S
λ = log(S / (||X_l||_F / √n))
```
- **Critically flawed**: Measures displacement magnitude, not sensitivity
- `jacobian_diagnostic.py` showed 1.4-8.9x inflation vs true Jacobian
- **Not canonical**. This is a common mistake — conflating displacement with the Jacobian

**Method 2: Directional Lyapunov** (Data-leaked)
```
direction d = mean(incorrect) - mean(correct), normalized
projection p_l = X_l @ d
λ_dir = 0.5 * log(var(p_{l+1}) / var(p_l))
```
- **Data leakage**: Direction computed on full dataset, tested on same data
- Without CV: d=1.68, AUC=0.82 (impressive but fake)
- With 5-fold CV: d=-0.03, AUC<0.50 (null, worse than random)
- **Lesson**: Always cross-validate data-driven directions

**Method 3: Spectrum width**
```
width = std(log(S_J))
```
- No significant differences. Measures spectral flatness, not correctness.

**Canonical assessment**: Method 1 is **not canonical** (displacement ≠ Jacobian). Method 2 is standard (projection onto discriminant direction) but requires CV. Method 3 is standard but uninformative here.

### 8.4 Menger Curvature (`phase3_dynamical_analysis.py`)

**Algorithm**: For three consecutive layer activations p₁, p₂, p₃:
```
κ = 4A / (|p₁-p₂| · |p₂-p₃| · |p₃-p₁|)
where A = (1/2)√(det(Gram matrix))
```

**Result**: Curvature profiles correlate r ≈ 0.999 for correct vs incorrect. Also r > 0.97 across tasks. **The curvature profile is determined by transformer architecture, not by task or correctness.**

**Exception**: For SFT model specifically, curvature *magnitude* (not profile) differs: d=0.53-0.75, p=0.002.

**Canonical**: Menger curvature is textbook differential geometry. Zhou et al. (2025) applied it to reasoning, finding it captures logical structure. In our case, the signal is architectural rather than semantic — possibly because we analyze residual stream (which has skip connections forcing near-constant curvature).

**Limitation**: Applied to discrete points (16 layers), not continuous curves. Also, 4096-dimensional space may require different curvature notions than 3D.

### 8.5 Path Signatures (`path_signature_analysis.py`)

**Algorithm**: Using `signatory` library (rough path theory):
```
1. PCA reduce trajectories: (n_samples, 16_layers, 4096) → (n_samples, 16, 32)
2. Compute depth-3 path signature → ~5,984 features
3. Logistic regression on signature features, 5-fold CV
4. Cross-domain: train on GSM8K, test on HumanEval
```

**Result**: SFT shows significant transfer (GSM8K→HumanEval AUC=0.78). Other models weaker.

**Canonical**: Path signatures are mathematically principled (reparameterization-invariant, complete characterization of paths up to tree-like equivalence). The `signatory` library implements standard computation.

**Limitation**:
- Depth 3 is shallow (higher depths capture more but are exponentially larger)
- PCA to 32 dims may lose important structure
- 5,984 features on 500 samples → regularization is critical
- Depends on `signatory` package (installation can be tricky)

### 8.6 Error Direction Analysis (`phase3_dynamical_analysis.py`, `aha_moment/analyze_wynroe_direction.py`)

**Algorithm**:
```
1. For each layer l:
   d_l = mean(X_l[incorrect]) - mean(X_l[correct])
   d_l = d_l / ||d_l||
2. Project all samples: score = X_l @ d_l
3. Classify: threshold at median
```

**Probing result**: d=1.70 at layer 14 (strong signal). But signal exists from layer 0, suggesting it's trivially available (problem difficulty encoded in input tokens), not a specialized error detection mechanism.

**Patching result** (causal, `replicate_wynroe_patching.py`):
- DeepSeek-R1-Distill: Sharp spike at L18 (+21.5% logit-diff recovery)
- OLMo RL-Zero: Gradual ramp, no spike
- OLMo SFT: Hybrid (L18 spike +10.3% AND L30 spike +32.8%)
- OLMo Think: Final spike only (+43.8% at L30)

**Key discovery**: Distillation creates localized "imitation circuits" at L16-18. Native RLVR creates distributed processing.

**Bug found and fixed**: Original patching code patched ALL token positions. Corrected to patch FINAL token only (standard TransformerLens methodology).

### 8.7 Perturbation Curvature (`curvature_and_stats.py`)

**Algorithm**:
```
1. Forward pass on original input → baseline activation a_0
2. Perturb random tokens with vocabulary probability ε
3. Forward pass on perturbed input → a_pert
4. Curvature ≈ var(a_pert - a_0) / ε²
```

**Canonical**: **NOT canonical**. True curvature requires Hessian (second derivatives). This measures input sensitivity, not manifold curvature. It's a proxy at best.

**Result**: No significant difference across models for Phase 1 analysis. Abandoned in favor of other methods.

### 8.8 Jacobian Diagnostic (`jacobian_diagnostic.py`)

**Purpose**: Validation tool, not analysis. Compares delta-based vs true Jacobian to determine which Lyapunov methods are valid.

**Key finding**: Delta-based SVD inflates effect sizes 1.4-8.9x. True Jacobian reveals orthogonality (cos ≈ 0.1 between consecutive layers), making Lyapunov ≈ 0 for all samples. This is the definitive explanation for why H5 failed.

---

## 9. Experiments — Self-Contained Studies

### 9.1 Aha Moment (`experiments/aha_moment/`)

**Purpose**: Investigate whether LLMs have internal error-detection signals and whether these correlate with error correction behavior.

**Sub-experiments**:

| Part | Name | Method | Finding | Canonical? |
|------|------|--------|---------|-----------|
| A | Error Detection Probing | Mean-difference direction | d=1.70 at L14 (strong but trivial) | Standard probing |
| A' | Activation Patching | Causal replacement at each layer | Distillation = localized L18; RLVR = distributed | Standard (TransformerLens) |
| B | Natural Pivot Detection | Velocity/curvature at "Wait" tokens | Pivots are SLOWER, not faster (d=-0.22) | Standard metrics |
| C | Active Error Correction | Corrupt calculation, test if model corrects | 26.7% detection, 11% correction (same as base!) | Novel experiment |
| D | MI Pivot Analysis | Correctness probe MI at token positions | RLVR shows -0.061 MI delta at corrections | Ad-hoc (v1→v2→v3) |

**Key findings**:
1. **Detection ≠ Correction**: Models detect errors internally (d=1.70) but can't fix them (11% correction rate = same as base model)
2. **Pivots are pauses, not insights**: "Wait, that's wrong" tokens correspond to SLOWER, MORE LINEAR trajectories (opposite of hypothesis)
3. **Distillation creates shortcuts**: DeepSeek-R1-Distill has localized L18 circuit; native RLVR has distributed processing

**Version chain**: MI analysis went through 3 versions (v1→v2→v3), fixing data contamination (GSM8K hallucinated "Passage:" text), improving window size, and separating correction vs deliberation tokens.

**Deprecated files**:
- `detect_pivots.py` — Phase 2 data doesn't contain generation activations
- `analyze_phase2_pivots.py` — Analyzing zeros (not actual dynamics)
- `mi_pivot_analysis.py` (v1) — Replaced by v3
- `mi_pivot_analysis_v2.py` — Intermediate version

### 9.2 SVD Reasoning Separability (`experiments/svd_reasoning_separability/`)

**Purpose**: Test whether RLVR training changes specific eigenvectors (separable reasoning) or all eigenvectors (entangled reasoning).

**Algorithm**:
```
1. Load base and RL-Zero activations on same prompts
2. Truncated SVD (randomized, k=100)
3. For each rank k: delta_k = 1 - |cos(U_base[:,k], U_rlvr[:,k])|
4. Compare mean_delta(top-10) vs mean_delta(tail-50)
```

**Result**:
| Task | Top-10 Delta | Tail-50 Delta | Ratio | Interpretation |
|------|--------------|---------------|-------|----------------|
| HumanEval | 0.480 | 0.574 | 0.84 | Tail changes MORE |
| GSM8K | 0.461 | 0.539 | 0.86 | Tail changes MORE |

**Conclusion**: RLVR preserves top eigenvectors (core structure) and refines tail eigenvectors (fine-grained). Reasoning is NOT linearly separable in eigenvector space. This motivates looking at dynamics rather than static geometry — but dynamics also failed (see Phase 3).

**Canonical**: SVD is textbook. The cosine similarity metric is standard. Top-10 vs tail-50 cutoff is arbitrary but reasonable.

**Limitation**: Only 50 samples per task (memory), 256 token truncation, randomized SVD approximation.

### 9.3 Generation Dynamics (`experiments/generation_dynamics/`)

**Purpose**: Investigate where RL-Zero's 7% accuracy improvement comes from. Input-time trajectories are nearly identical (cos_sim 0.995), so the gain must be in generation dynamics.

**Data collected**:
- olmo3_base: 495/500 GSM8K (55 HumanEval truncated at 1024 tokens), 500 LogiQA
- olmo3_rl_zero: Collection in progress

**Planned analysis** (not yet implemented):
- Phase A: Entropy profiles over generation steps
- Phase B: Correctness prediction from generation features (beat AUC 0.75?)
- Phase C: Divergence analysis (when do base/RL-Zero split?)
- Phase D: Critical moment detection (entropy inflection points)

**Status**: Data collection ~complete, analysis scripts not written.

### 9.4 Belief Tracking (`experiments/belief_tracking/`)

**Purpose**: Track P(correct|hidden_state) over clause boundaries during generation.

**Algorithm**:
```
1. Train LogisticRegression probe on mean layer activations → P(correct)
2. Parse model outputs into clauses (regex: "First", "So", "Therefore", "\n\n")
3. Apply probe retroactively to each clause position
4. Compare belief evolution: correct vs incorrect, RL-Zero vs SFT
```

**Finding**: RL-Zero correct solutions show SMOOTH belief evolution; SFT/Think show JUMPY evolution (discrete jumps at correction tokens). Opposite patterns suggest different internal mechanisms.

**Limitation**:
- Clause detection is regex-based (fragile)
- Text→token alignment uses heuristic (char_pos / 4)
- Probe accuracy only AUC 0.66 — may not reliably represent "belief"
- Incomplete (early stage)

---

## 10. Results — What We Found

### ⚠️ CRITICAL UPDATE (2026-02-12): v2 Analysis with PCA-Bias Fixed

**The Problem**: PCA-based methods in v1 analyses were fundamentally flawed due to:
1. Superposition bias: PCA captures high-variance (frequent features), not correctness-relevant directions
2. Gradient disconnect: Activation eigenvectors ≠ Jacobian singular vectors
3. RLVR feature loss: Tail eigenvectors change 3-8× more than top → PCA→32 discards the signal

**v1 Impact**:
- Path signatures: Cross-domain transfer AUC 0.3-0.5 (near random)
- Attractor analysis: Biased toward high-variance clusters

**v2 Fix**: Three projection methods tested:
1. Random Projection (GaussianRandomProjection) - no variance bias
2. Velocity-space PCA (on v = x_{l+1} - x_l) - dynamics not position
3. Probe-informed CV (probe weight + random orthogonal) - correctness-informed

**v2 Results** (2026-02-12, 3hr runtime, 9 files):

| Finding | v1 (PCA-biased) | v2 (Fixed) | Method |
|---------|-----------------|------------|--------|
| **H1**: Within-domain path signatures | Not tested | **0.6-0.85 AUC** | Probe-informed CV |
| Best: olmo3_rl_zero HumanEval | — | **0.845 AUC** | Probe-informed CV |
| **H2**: Cross-domain path signatures | 0.3-0.5 AUC | **0.7-0.9 AUC** | Probe-informed CV |
| Best: olmo3_rl_zero gsm8k→humaneval | — | **0.891 AUC** 🔥 | Probe-informed CV |
| Menger curvature (HumanEval) | r≈0.999 (architectural) | **d=1.7-2.0, p<0.0001** | Magnitude diff |
| Wynroe error direction within-domain | Not tested | **56-81% accuracy** | Diff-in-means |
| Wynroe error direction cross-domain | Not tested | 19-85% (RL-Zero only) | Diff-in-means |
| Attractor clustering | PCA-biased | **Purity 64-94%** | Random projection |

**Key Insights**:
- ✅ **H2 CONFIRMED**: Domain-invariant signatures exist when using probe-informed projections
- ❌ **H2 FAILS** without correctness-informed projection (random/PCA: AUC 0.2-0.6)
- **Correctness geometry is NON-LINEAR**: Linear directions fail cross-domain, path signatures succeed
- **Code ≠ Math**: HumanEval shows strong curvature (d=1.7-2.0), GSM8K doesn't (d=0.05-0.5)
- **Error modes are stereotyped**: Incorrect solutions cluster (6-8 attractors), correct solutions scattered (0-5)
- **RL-Zero shows best transfer**: 0.891 AUC cross-domain, 85% error direction transfer

**Files**: See `results/V2_ANALYSIS_FINDINGS.md` (comprehensive), `results/v2_0shot_20260211_212500/` (outputs)

**Scripts**: `scripts/analysis/*_v2.py` (fixed), deprecated originals have warnings

**Archived**: `results/archive_v1_pca_biased/` (old PCA-biased results)

---

### Positive Findings (v1 - Not PCA-affected, Still Valid)

| ID | Finding | Method | Effect Size | p-value | Implications |
|----|---------|--------|-------------|---------|-------------|
| P1 | Correct/incorrect distinguishable | Linear probe | AUC 0.68-0.75 | <0.001 | H1 confirmed |
| P2 | SFT error direction transfers across domains | Direction cosine | cos=0.355 | <0.001 | H2 partial (SFT only) |
| P3 | Correctness encoded early (tokens 2-8%) | Position analysis | d=-1.16 at T11 | <0.01 | Supports decision-before-reasoning |
| P4 | Incorrect solutions preserve token structure more | Per-sample CKA | d=-0.64 | 0.029 | Correct solutions "work harder" |
| P5 | RL-Zero preserves base representation | CKA + eigenvec | CKA=0.995 | — | Minimal intervention mechanism |
| P6 | Distillation ≠ native RLVR internally | Activation patching | L18 spike (distillation) vs gradual (RLVR) | — | Different architectures of improvement |
| P7 | Detection ≠ Correction capability | Error injection | 26.7% detect, 11% fix | — | Internal signal doesn't enable repair |
| P8 | Belief dynamics differ by training | Clause-level probe | RL-Zero=smooth, SFT=jumpy | — | Different inference strategies |

### Negative Findings (v1)

**⚠️ Some v1 negative findings were artifacts of PCA bias - see v2 updates above**

| ID | Finding | Method | Expected | Found | Status (v2) |
|----|---------|--------|----------|-------|-------------|
| N1 | True Jacobian Lyapunov null | Regression Jacobian | Correct = more stable | d≈0.15, p>0.3 | ✅ Still valid (orthogonality) |
| N2 | Delta-based Lyapunov invalid | SVD(delta) | Valid proxy | 1.4-8.9x inflation | ✅ Still valid (measures displacement) |
| N3 | Directional Lyapunov = data leakage | Direction + variance | d=1.68 signal | d=-0.03 with CV | ✅ Still valid (CV fixed) |
| N4 | Path sigs weak cross-domain | PCA→32 + signatures | Strong transfer | 0.3-0.5 AUC | ❌ **REVERSED** → v2: 0.7-0.9 AUC with probe |
| N5 | SVD separability reversed | Eigenvector delta | Top change more | Tail change 3-8x more | ✅ Still valid (RLVR refines tail) |
| N6 | Pivot tokens = induction heads | Velocity at pivots | Sharp transitions | SLOWER, MORE LINEAR | ✅ Still valid |
| N7 | Procrustes alignment null | Rotation + scale | Correct differs | Same transformation | ✅ Still valid (architectural) |
| N8 | SVCCA alignment null | Canonical correlation | Correct differs | Same CCA structure | ✅ Still valid (architectural) |
| N9 | Sequence velocity null | Token-to-token norm | Correct differs | No difference | ✅ Still valid (architectural) |
| N10 | Acceleration null | Second derivative | Correct differs | d=-0.02 | ✅ Still valid (architectural) |
| N11 | RL-Zero cross-domain null (linear) | Error direction | Transfers | cos=0.098 (orthogonal) | ⚠️ **PARTIAL** → v2: Path sigs transfer at 0.89 AUC |
| N12 | Menger curvature profile correlation | 3-point curvature | Profile differs | r≈0.999 (architectural) | ⚠️ **REFINED** → v2: Magnitude differs (d=1.7-2.0) |
| N13 | Attractor PCA-biased | PCA→50 + K-means | Meaningful clusters | Variance-biased | ❌ **FIXED** → v2: Random projection works |

**v2 Key Reversals**:
- **N4**: PCA-bias artifact - path signatures DO transfer cross-domain (0.7-0.9 AUC) with probe-informed projection
- **N11**: Linear directions fail, but path signature geometry transfers (RL-Zero: 0.89 AUC gsm8k→humaneval)
- **N12**: Profile correlation is architectural (r≈0.999), but curvature MAGNITUDE differs (d=1.7-2.0 on HumanEval)
- **N13**: Attractor clustering works with random projection (purity 64-94%), not PCA

### The Orthogonality Problem (Root Cause of H5 Failure)

**Observation**: cos(X_l, X_{l+1}) ≈ 0.10-0.12 for all OLMo-3 layers.

**Consequence**: Layer transitions are approximately orthogonal rotations, not expansions. This means:
- True Jacobian singular values ≈ 1 → Lyapunov ≈ 0 for ALL samples
- Procrustes sees the same rotation quality for correct and incorrect
- SVCCA and CCA see the same canonical structure
- Any measure based on "expansion rate" is blind

**Scope**: Architecture-specific to OLMo-3. Other models may differ. We have not tested Llama, Mistral, or GPT.

### Paradoxical Findings

1. **Base model has strongest H1 separation** (d=2.17) despite lowest accuracy (12.6%). Explanation: When only 12% are correct, those rare correct samples are geometrically very distinct from the incorrect majority.

2. **SFT transfers cross-domain but RL-Zero doesn't**, even though RL-Zero is "better aligned" with base. SFT's dramatic reshaping (52% preservation) creates a uniform "error signature" that generalizes; RL-Zero's minimal change (98% preservation) means its error patterns are domain-specific.

3. **Correct solutions have LOWER CKA** (less self-similar across layers). Interpretation: correct solutions restructure token relationships more aggressively — they "work harder" than incorrect solutions.

---

## 11. Methodological Critiques

### What Was Right

1. **Cross-validation everywhere**: 5-fold stratified CV on all classifiers prevents overfitting claims
2. **Effect sizes + p-values**: Cohen's d reported alongside significance (avoids "p-hacking")
3. **Multiple baselines**: Compared against confidence, length, format, random labels
4. **Honest negative reporting**: 11 null results documented alongside 9 positive findings
5. **Causal follow-up**: Activation patching (Experiment A') adds causality beyond probing

### What Was Wrong or Questionable

1. **PCA for task-relevant feature extraction** (2026-02-12): This was the **most impactful error** in the project. Using PCA for dimensionality reduction before path signature and attractor analyses introduced systematic bias:
   - **Superposition problem**: PCA captures high-variance directions (dominated by frequent features in superposition), NOT correctness-relevant directions
   - **Gradient disconnect**: Activation covariance eigenvectors have no mathematical correspondence to Jacobian singular vectors (functionally important directions)
   - **RLVR signal loss**: RLVR changes tail eigenvectors 3-8× more than top eigenvectors → `PCA(n_components=32)` discarded exactly the refinement that matters
   - **Impact**: Path signature cross-domain transfer appeared to fail (0.3-0.5 AUC), when in reality it succeeds (0.7-0.9 AUC) with probe-informed projection
   - **Fix**: v2 analyses use (a) random projection (no variance bias), (b) velocity-space PCA (dynamics not position), (c) probe-informed CV (correctness direction)
   - **Lesson**: PCA for visualization = fine. PCA for finding task-relevant features in superposition = dangerous. Always validate projection choices and test multiple methods.

2. **Fast Lyapunov via Frobenius norm**: This was **never valid** as a Lyapunov approximation. The Frobenius norm ||X_{l+1}||/||X_l|| measures displacement magnitude, not the Jacobian eigenspectrum. The Jacobian J = ∂f/∂x requires computing derivatives (or regression), not just output norms. **This is a common mistake in the ML literature** (confusing "how far the output moves" with "how sensitive the mapping is"). The project correctly identified this error via the diagnostic script, but several analyses were run with the invalid method before the diagnostic was created.

2. **Directional Lyapunov data leakage**: Computing the error direction on the full dataset and then testing on the same data creates circular inference. The direction is partially fit to noise, inflating effect sizes by 133-200%. This was caught and corrected with CV, but initial reports (before correction) may have influenced research direction.

3. **Menger curvature in high dimensions**: The Menger curvature formula κ = 4A/(abc) assumes points lie in a low-dimensional space where curvature is meaningful. In 4096 dimensions with only 3 points (consecutive layers), the curvature is dominated by the ambient dimension, not the manifold structure. The r≈0.999 correlation between correct/incorrect curvature profiles confirms this — the curvature is architectural, not semantic.

4. **Path signature depth**: Depth 3 captures interactions up to 3rd order. For 16-layer trajectories in 32-dimensional PCA space, this may miss critical higher-order structure. Literature on rough paths suggests depth 5+ for complex signals.

5. **HumanEval correctness labels**: Syntax-check-only means we're labeling "compiles" as correct. A solution that compiles but fails tests is "correct" in our labels but wrong in reality. This introduces unknown label noise.

6. **Small sample sizes in some analyses**: MI pivot analysis (Experiment D) has n=2-3 correction tokens per model. Statistical power is essentially zero for detecting real effects.

7. **Architecture-specific conclusions**: All analysis on OLMo-3 7B. The orthogonality property, curvature profiles, and RL-Zero preservation may not generalize to other architectures.

### Hyperparameters Never Justified or Ablated

| Parameter | Value | Location | Should Be |
|-----------|-------|----------|-----------|
| Regularization C | 0.1 | h1_h2_classifier.py | Ablated over {0.01, 0.1, 1, 10} |
| Jacobian regularization λ | 1e-6 | empirical_jacobian_lyapunov.py | Sensitivity analysis |
| PCA dimensions | 32 | path_signature_analysis.py | **INVALID METHOD** (v2: replaced with random/probe projection) |
| Signature depth | 3 | path_signature_analysis.py | {2, 3, 4, 5} |
| K-means clusters | 8 | phase3_dynamical_analysis.py | Elbow/silhouette analysis |
| SVD components | 100 | svd_reasoning_separability | {50, 100, 200} |
| Top-k vs tail cutoff | 10 vs 50 | svd_reasoning_separability | {5,10,20} vs {30,50,100} |
| MI window size | 5 | mi_pivot_analysis_v3.py | {3, 5, 10} |

---

## 12. GPU Optimization Lessons

### Discovery: Batched Collection Was GPU-Idle 80%

During Phase 2 collection, `collect_logiqa_batched.py` had 0-10% GPU utilization despite running on RTX 4090s. Four bottlenecks identified:

| # | Bottleneck | Impact | Fix | Speedup |
|---|-----------|--------|-----|---------|
| 1 | GPU→CPU transfer in hook | 25% idle | `.detach()` not `.detach().cpu()` | 2x |
| 2 | Sequential forward passes | 75% of batch time | Batched with padding | 3.4x |
| 3 | Blocking HDF5 writes | 15-20% idle | Async writer thread | 1.2x |
| 4 | Memory fragmentation | Slowdown over time | `empty_cache()` + `gc.collect()` | Prevents degradation |
| 5 | CPU tokenization | Minor | vLLM handles on GPU | Minor |
| 6 | Slow HF generation | Major | vLLM 3-5x faster | 3-5x |

**Combined**: 4-5x with fixes 1-4 (24GB GPUs), 8-10x with all 6 (H100).

### Pipelined Execution Pattern

```
Optimal:
GPU:    [Gen B1][Collect B1][Gen B2][Collect B2][Gen B3]
CPU:                        [Write B1]          [Write B2]
Memory:         [Clean]              [Clean]

vs Original:
GPU:    [Gen B1] IDLE (transfer) [Fwd1][Fwd2][Fwd3][Fwd4] IDLE (write)
                 ^^^^                                       ^^^^
                 25% idle                                   15% idle
```

---

## 13. Decision Tree & Next Steps

### Current State (2026-02-09)

```
Phase 1 (✅ Complete) → Static geometry differs by training method
    ↓
Phase 2 (🔄 92%) → 11/12 trajectory files collected (missing: olmo3_base/gsm8k)
    ↓
Phase 3 (🔄 Mostly done) → Multiple analysis methods applied
    ├─ H1: ✅ TRUE (AUC 0.68-0.75)
    ├─ H2: ⚠️ PARTIAL (SFT yes, RL-Zero no)
    ├─ H5: ❌ FAILED (orthogonality kills Lyapunov)
    ├─ Static > Dynamics (linear probe beats everything)
    └─ Many null results (N1-N11)
    ↓
Phase 4 (⏳ Not started) → Steering intervention
    ↓
Phase 5 (⏳ Not started) → Write-up

Experiments:
├─ Aha Moment (✅ Mostly done) → Detection ≠ Correction; distillation ≠ RLVR
├─ SVD Separability (✅ Complete) → Tail-heavy refinement (not separable)
├─ Generation Dynamics (🔄 Collecting) → Where does RL-Zero improvement live?
└─ Belief Tracking (🔬 Early) → Different belief evolution by training
```

### What Should Happen Next

1. **Finish generation dynamics analysis** — This is the most promising open question. Input trajectories are identical (cos 0.995) yet RL-Zero is 7% more accurate. The answer must be in generation dynamics.

2. **Recollect olmo3_base/gsm8k** — Corrupted file, needed for completeness.

3. **Consider Phase 4 steering** — Despite weak H2, the SFT error direction does transfer. Steering along this direction could work for SFT at least.

4. **Write up negative results** — The 11 null findings (especially the orthogonality bottleneck) are publishable. "Why Lyapunov exponents don't work for transformers" would be useful for the community.

5. **Test on other architectures** — The orthogonality property may be OLMo-specific. Testing Llama-3.1-8B or Mistral-7B would determine generalizability.

### Publication Strategy Given Current Results

| Scenario | Venue | Story |
|----------|-------|-------|
| Generation dynamics beat AUC 0.75 | NeurIPS/ICML main | "Where RLVR improvement lives: generation, not representation" |
| Steering works for SFT | NeurIPS/ICML main | "Training method determines error geometry: SFT universal, RLVR local" |
| Only negative + characterization | ACL/EMNLP | "Orthogonality bottleneck: why dynamical systems analysis fails for transformers" |
| Strong aha moment story | ICLR workshop | "Detection without correction: internal error signals in LLMs" |

---

## 14. References

### Theoretical Framework
- Allen-Zhu & Li (2024): Physics of Language Models — everything is interpolation
- Merullo et al. (2025): Loss curvature separates memorization from generalization
- Ren & Liu (2026): HRM analysis — attractor dynamics, grokking transitions
- Shai et al. (2024): Belief state geometry in residual stream
- Zhou et al. (Oct 2025): Menger curvature captures logical structure

### Supporting Evidence
- Zhang et al. (2025): Hidden states predict correctness
- Marks & Tegmark (2023): Truth has geometric structure
- Turner et al. (2023): Activation steering works
- Hosseini & Fedorenko (2023): Trajectories straighten with success
- Azaria & Mitchell (2023): Internal truth detection

### Critical Challenges
- Turpin et al. (2023): CoT can be unfaithful
- Afzal et al. (2025): Decision before reasoning
- Hewitt & Liang (2019): Probes need control tasks
- Ley et al. (2024): Faithfulness interventions fail to transfer
- Jin et al. (2025): RLVR artifacts

### Methods
- Signatory library: Path signatures from rough path theory
- TransformerLens: Hook-based activation access
- CKA: Kornblith et al. (2019): Centered kernel alignment
- Wynroe et al.: Error-detection direction via activation patching
