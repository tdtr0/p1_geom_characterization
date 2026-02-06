# Menger Curvature Analysis: Findings and Next Steps

**Date**: 2026-01-20 (Updated with critical correction)
**Model**: olmo3_base (0-shot)
**Tasks**: HumanEval, LogiQA

---

## Key Finding: Layer Curvature is Architectural

Menger curvature on **layer trajectories** (path through layers 0→30) is purely architectural:

1. **Within-domain**: Curvature does NOT significantly distinguish correct vs incorrect (p > 0.2)
2. **Cross-domain**: Curvature profiles are HIGHLY correlated (r = 0.996, p < 0.0001)

**CRITICAL UPDATE**: The r=0.996 finding is a **RED HERRING / NULL RESULT**!

Further analysis showed that curvature profiles are identical across:
- Correct vs Incorrect (within domain): r = 0.9999
- Correct vs Correct (cross domain): r = 0.9961
- All pairwise combinations: r > 0.995

This means curvature profile is an **architectural property** of the transformer, not a signal related to reasoning or correctness.

---

## Methods (for reference)

**Menger curvature** for three consecutive points P1, P2, P3:
```
κ = 4 * Area(P1, P2, P3) / (|P1-P2| * |P2-P3| * |P1-P3|)
```

**Data processing**:
1. Load trajectories: `(n_samples, 512, 16, 4096)`
2. Average across sequence: `(n_samples, 16, 4096)`
3. Compute curvature at each layer transition

---

## New Direction: Sequence Flow at Last Layer

Instead of curvature on **layer paths** (0→30), compute geometry on **token sequences** at the last layer (token 0→512 at layer 30).

**Why this might work**:
- Last layer encodes the "answer state" closest to output
- Sequence dynamics depend on actual content being generated
- Aligns with belief dynamics literature (Shai et al.)

---

## Execution Plan

### H_flow1: Velocity Distribution (~10 min)

**Measure**: `velocity_t = ||x_{t+1} - x_t||` at layer 30

**Test**:
1. Extract last layer (index 15) from trajectories
2. Compute velocity at each token transition
3. Compare correct vs incorrect: mean, variance, end-velocity

**Prediction**: Correct solutions may "settle" (lower velocity near end)

---

### H_flow2: Sequence Curvature (~15 min)

**Measure**: `curvature_t = Menger(x_{t-1}, x_t, x_{t+1})` at layer 30

**Test**:
1. Compute Menger curvature at each token transition
2. Compare curvature profiles: correct vs incorrect
3. If r < 0.95, we have content-dependent signal (unlike layer curvature r≈1.0)

**Prediction**: Correct solutions may have more "direct" paths (lower curvature)

---

### H_flow3: Cross-Domain Flow Transfer (~30 min)

**Features per sample**:
```
[mean_velocity, var_velocity, mean_curvature, var_curvature, convergence_rate]
```

**Test**:
1. Extract flow features for HumanEval and LogiQA
2. Train classifier on flow features (correct vs incorrect) within domain
3. Test cross-domain transfer
4. Compare to error-direction transfer (which was asymmetric)

---

## Priority

| # | Hypothesis | Effort | Key Question |
|---|------------|--------|--------------|
| 1 | H_flow1 | ~10 min | Does generation "speed" differ? |
| 2 | H_flow2 | ~15 min | Is sequence curvature content-dependent? |
| 3 | H_flow3 | ~30 min | Does flow structure transfer? |

| Task | Correct Curvature | Incorrect Curvature | Effect Size (d) | p-value |
|------|-------------------|---------------------|-----------------|---------|
| HumanEval | 2.383 | 2.111 | 0.315 | 0.291 |
| LogiQA | 1.266 | 1.196 | 0.244 | 0.322 |

**Interpretation**: The effect sizes are small (d ~ 0.2-0.3) and not statistically significant. This confirms that curvature alone is not a strong discriminator for correctness within a single domain.

### Cross-Domain Correlation

**Layer-wise curvature profiles** were computed by averaging curvature at each layer transition:

| Comparison | Correlation (r) | p-value |
|------------|-----------------|---------|
| HumanEval ↔ LogiQA | **0.996** | < 0.0001 |

**Initial Interpretation** (INCORRECT): ~~Despite being different tasks (code vs logic), the curvature profiles across layers are nearly identical. This suggests a domain-invariant geometric structure.~~

---

## CRITICAL CORRECTION: Correctness-Conditioned Analysis

After the initial finding, we tested whether curvature differs by correctness:

### Pairwise Curvature Profile Correlations

| Comparison | Correlation (r) |
|------------|-----------------|
| HumanEval Correct ↔ HumanEval Incorrect | **0.9999** |
| LogiQA Correct ↔ LogiQA Incorrect | **0.9998** |
| HumanEval Correct ↔ LogiQA Correct | 0.9961 |
| HumanEval Incorrect ↔ LogiQA Incorrect | 0.9963 |

### Interpretation

**ALL correlations are essentially r ≈ 1.0!**

This means:
1. Curvature profile is **identical** whether the solution is correct or incorrect
2. Curvature profile is **identical** across tasks (code vs logic)
3. **Curvature profile is a property of the ARCHITECTURE, not the task or correctness**

The r=0.996 cross-domain finding tells us **nothing** about reasoning transfer. It just shows that OLMo-3 processes information similarly through layers regardless of content.

---

## What About Curvature Magnitude?

Even though the **profile shape** is identical, could the **magnitude** differ?

| Task | Correct | Incorrect | Cohen's d | p-value |
|------|---------|-----------|-----------|---------|
| HumanEval | 2.38 ± 0.96 | 2.11 ± 0.84 | 0.319 | 0.291 |
| LogiQA | 1.27 ± 0.22 | 1.20 ± 0.30 | 0.246 | 0.322 |

**Result**: Small positive effect (correct has higher curvature) but not significant (p > 0.2).

---

## CORRECTED Key Finding: Curvature is Architectural, Not Task-Related

### H_flow1: Velocity Distribution — NO SIGNAL

| Task | Correct | Incorrect | Cohen's d | p |
|------|---------|-----------|-----------|---|
| HumanEval | 15.8 ± 5.7 | 18.5 ± 8.6 | -0.32 | 0.29 |
| LogiQA | 28.0 ± 5.1 | 30.2 ± 7.8 | -0.30 | 0.23 |

**Initially we thought**: The geometric structure (curvature profile) DOES transfer (r = 0.996).

**CORRECTED**: The r=0.996 is **NOT evidence for transfer**. Curvature profiles are identical whether:
- Solution is correct or incorrect (r=0.9999)
- Task is code or logic (r=0.996)

**What this actually tells us**:
1. Curvature profile is determined by **transformer architecture**, not task or correctness
2. All trajectories through OLMo-3 have nearly identical curvature profiles
3. The r=0.996 cross-domain finding is a **null result** - it doesn't support H2

**Analogy**: It's like measuring the "bumpiness" of different roads and finding they're all equally bumpy. This doesn't tell you which roads lead to the right destination - it just tells you the asphalt was laid the same way everywhere.

---

## Conclusion

**All three sequence flow hypotheses show weak or no signal:**

1. **H_flow1**: Velocity distribution does not distinguish correct/incorrect
2. **H_flow2**: Sequence curvature profile is still architectural (r > 0.95)
3. **H_flow3**: Weak transfer signal (AUC 0.54-0.66), direction opposite to error-direction

**Implication**: Geometric measures (curvature, velocity) on both layer paths AND token sequences capture architectural properties, not semantic content. The superposition problem persists even at the last layer.

**Next direction**: May need to analyze:
- Attention patterns (which tokens attend to which)
- Probe-based representations (trained linear probes)
- Specific reasoning tokens only (not full sequence)

---

## Code Reference

- Layer curvature: [scripts/analysis/phase3_dynamical_analysis.py](../../scripts/analysis/phase3_dynamical_analysis.py)
- Sequence flow: [scripts/analysis/sequence_flow_analysis.py](../../scripts/analysis/sequence_flow_analysis.py)
