# Sequence Flow Hypotheses (Residual Stream Dynamics)

**Date**: 2026-01-20
**Status**: Active hypotheses for testing

---

## Background

Previous analysis computed Menger curvature on **layer trajectories** (path through layers 0→30). This captured purely architectural patterns (r≈1.0 for all comparisons), not task or correctness signals.

**Key insight**: Raw activations are superpositioned — geometric measures on layer paths capture transformer architecture, not semantic content.

**New approach**: Compute geometry on **sequence trajectories at the last layer** — the path through token positions (token 0→1→...→512) at layer 30. This is the "belief state flow" as the model generates its answer.

**Why this might work**:
- Last layer is closest to output — encodes the "answer state"
- Sequence dynamics depend on actual content (what's being generated)
- Aligns with belief dynamics literature (Shai et al.) — residual stream tracks belief updates
- Less architectural, more semantic

---

## Hypotheses

### H_flow1: Velocity distribution at last layer

**Motivation**: The "speed" of representation change across tokens may differ for correct vs incorrect solutions.

**Prediction**:
- Correct solutions have different velocity distribution (mean, variance, or shape)
- Possible: Correct solutions "settle" more (lower velocity near end)
- Possible: Incorrect solutions "oscillate" more (higher variance)

**Measure**:
```
velocity_t = ||x_{t+1} - x_t||  at layer 30
```

**Test**:
1. Extract last layer activations: `(n_samples, 512, 4096)` → layer 30 only
2. Compute velocity at each token position
3. Compare distributions: correct vs incorrect
4. Metrics: mean velocity, velocity variance, velocity at end vs start

**Compute**: ~10 min on CPU (existing data, simple computation)

---

### H_flow2: Curvature of sequence path at last layer

**Motivation**: The "turning" of the representation path may capture reasoning structure.

**Prediction**:
- Correct solutions have different curvature profiles across sequence
- Possible: More "direct" paths (lower curvature) for correct
- Possible: Characteristic curvature patterns at decision points

**Measure**:
```
curvature_t = Menger(x_{t-1}, x_t, x_{t+1})  at layer 30
```

**Test**:
1. Extract last layer activations
2. Compute Menger curvature at each token transition
3. Compare: curvature profile correlation (correct vs incorrect)
4. If correlation < 0.95, we have signal (unlike layer curvature which was r≈1.0)

**Compute**: ~15 min on CPU

---

### H_flow3: Cross-domain flow comparison (manifold structure)

**Motivation**: If we have flow statistics for each dataset, we can compare the "manifold of flows" across domains.

**Prediction**:
- Within-domain: correct/incorrect have different flow manifolds
- Cross-domain: correct flows from HumanEval may resemble correct flows from LogiQA
- This tests if "correct reasoning flow" is domain-invariant

**Measure**:
```
For each sample: flow_features = [mean_velocity, var_velocity, mean_curvature, var_curvature, convergence_rate]
```

**Test**:
1. Extract flow features for all samples in HumanEval and LogiQA
2. Train classifier on flow features (correct vs incorrect) within domain
3. Test cross-domain transfer of flow-based classifier
4. Compare to error-direction transfer (which was asymmetric)

**Compute**: ~30 min on CPU

---

### Priority Order for Sequence Flow Experiments

| Priority | Hypothesis | Effort | Key Question |
|----------|------------|--------|--------------|
| 1 | H_flow1 (velocity) | Low (~10 min) | Does generation "speed" differ? |
| 2 | H_flow2 (curvature) | Low (~15 min) | Is sequence curvature content-dependent? |
| 3 | H_flow3 (cross-domain) | Medium (~30 min) | Does flow structure transfer? |

**Recommendation**: Run all three in sequence. H_flow2 is the key test — if sequence curvature shows r < 0.95 for correct vs incorrect (unlike layer curvature r≈1.0), we have found a content-dependent geometric signal.
