# Phase 1: Geometric Characterization (COMPLETE)

**Status**: ✅ Complete  
**Duration**: Completed Jan 2026  
**Objective**: Establish that RLVR and SFT produce measurably different static geometry

---

## Summary of Results

Phase 1 successfully demonstrated that different post-training paradigms produce dramatically different geometric structures:

| Model | Subspace Preservation vs Base | Interpretation |
|-------|------------------------------|----------------|
| RL-Zero | **98.6% ± 1.1%** | Preserves base geometry |
| SFT | **52.4% ± 12.6%** | Reshapes geometry |
| Think (RLVR) | **50.4% ± 12.9%** | Reshapes geometry |

**Statistical significance**: Cohen's d > 4.0, p < 10⁻¹⁷ for all comparisons

---

## What Was Measured

### Static Geometric Properties

1. **Effective Rank**: Intrinsic dimensionality of activation space
2. **Spectral Decay**: Power-law exponent of singular value decay
3. **Subspace Preservation**: Principal angles between base and fine-tuned subspaces
4. **Local Curvature**: Perturbation response at layer 15

### Data Collected

- **Models**: 4 (Base, SFT, RL-Zero, Think)
- **Tasks**: 3 (GSM8K, HumanEval, LogiQA)
- **Samples**: 500 per task
- **Layers**: Last token activations at all 32 layers
- **Storage**: 1.1 GB total

---

## Key Findings

### Finding 1: RL-Zero Preserves Base Geometry

- RL-Zero maintains >96% subspace overlap with base model across all tasks
- SFT variants preserve <53%
- **Implication**: Pure RL makes minimal geometric changes while achieving fine-tuning

### Finding 2: SFT Dramatically Reshapes Geometry

- SFT and Think (SFT+DPO+RLVR) both show ~50% preservation
- Adding DPO+RLVR to SFT doesn't further change geometry (Think ≈ SFT)
- **Implication**: The SFT step is what causes geometric reshaping

### Finding 3: Curvature Differences Are Subtle

- RL-Zero maintains base curvature (0.624 vs 0.626 on GSM8K)
- SFT slightly increases curvature (0.644 vs 0.626)
- **Implication**: Curvature is less discriminative than subspace preservation

---

## Limitations Identified

### What Phase 1 Did NOT Show

1. **No link to reasoning quality**: We showed models differ, but not whether geometry correlates with correctness
2. **Static analysis only**: Measured final representations, not activation flow through layers
3. **No causal evidence**: Correlational findings; no interventions

### Confounds Not Addressed

- Problem difficulty
- Output length
- Task-specific format

---

## Transition to Phase 2

Phase 1 established that geometry differs across training paradigms. Phase 2 asks: **Does geometry distinguish correct from incorrect reasoning?**

**Key changes for Phase 2**:
1. Collect **trajectories** (all layers), not just final activations
2. Record **correctness labels** (model answer vs ground truth)
3. Analyze **flow geometry** (path signatures, curvature through layers)
4. Test **cross-domain transfer** (H2: the critical test)

---

## Files and Artifacts

### Data Files (on eyecog)
- `data/activations/olmo3_base/gsm8k.h5` (269 MB)
- `data/activations/olmo3_base/humaneval.h5`
- `data/activations/olmo3_base/logiqa.h5`
- Similar for SFT, RL-Zero, Think

### Analysis Scripts
- `scripts/run_analysis.py`: Compute geometric measures
- `scripts/curvature_and_stats.py`: Statistical tests
- `src/geometric_measures.py`: Core measurement functions

### Results
- `results/geometric_analysis_detailed.csv`: Per-layer measures
- `results/geometric_analysis_summary.csv`: Aggregated statistics

---

## Lessons Learned

### What Worked Well

1. **TransformerLens integration**: Clean activation extraction
2. **HDF5 storage**: Efficient, compressed storage
3. **Checkpointing**: Fault-tolerant collection
4. **Even-layer sampling**: Validated via smoothness analysis

### What to Improve for Phase 2

1. **Add correctness labels**: Critical for H1-H2 tests
2. **Collect full trajectories**: Not just last token
3. **Balance classes**: Ensure sufficient correct/incorrect samples
4. **Control for confounds**: Difficulty, length, format

---

## Compute and Storage

### Phase 1 Costs

- **GPU hours**: ~25 hours on 2x RTX 3090
- **Storage**: 1.1 GB (last token only)
- **Cost**: $0 (used existing hardware)

### Phase 2 Projections

- **Storage**: ~56 GB (full trajectories, even layers)
- **GPU hours**: ~40-60 hours
- **Requires**: Cleanup of eyecog disk space (currently 122 GB free)

---

## References to Phase 1 Work

- Full implementation plan: `phase1_implementation_plan.md`
- Results summary: Lines 21-73 of that file
- Code: `src/activation_collector.py`, `src/geometric_measures.py`
