# Menger Curvature Analysis: Findings and Next Steps

**Date**: 2026-01-21
**Status**: Layer curvature abandoned → Sequence flow analysis next

---

## Key Finding: Layer Curvature is Architectural

Menger curvature on **layer trajectories** (path through layers 0→30) is purely architectural:

| Comparison | Correlation (r) |
|------------|-----------------|
| Correct ↔ Incorrect (same domain) | 0.9999 |
| Cross-domain (HumanEval ↔ LogiQA) | 0.996 |

**Conclusion**: Curvature profile is determined by transformer architecture, not task or correctness. All trajectories through OLMo-3 have nearly identical curvature profiles regardless of content.

**Why this happened**: Raw activations are superpositioned — geometric measures on layer paths capture how the transformer processes information through layers, not semantic content.

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

**Critical test**: H_flow2 — if sequence curvature shows r < 0.95 for correct vs incorrect, we've found a content-dependent geometric signal.

---

## Results (2026-01-21)

### H_flow1: Velocity Distribution — NO SIGNAL

| Task | Correct | Incorrect | Cohen's d | p |
|------|---------|-----------|-----------|---|
| HumanEval | 15.8 ± 5.7 | 18.5 ± 8.6 | -0.32 | 0.29 |
| LogiQA | 28.0 ± 5.1 | 30.2 ± 7.8 | -0.30 | 0.23 |

Correct solutions have slightly lower velocity (d ~ -0.3) but not significant.

### H_flow2: Sequence Curvature — STILL ARCHITECTURAL

| Task | Profile Correlation (r) |
|------|------------------------|
| HumanEval | 0.9769 |
| LogiQA | 0.9903 |

Both r > 0.95, meaning sequence curvature profile at last layer is **also architectural**, not content-dependent.

### H_flow3: Cross-Domain Flow Transfer — WEAK SIGNAL

| Metric | HumanEval | LogiQA | HE→LQ | LQ→HE |
|--------|-----------|--------|-------|-------|
| Within-domain AUC | 0.560 | **0.662** | — | — |
| Cross-domain AUC | — | — | 0.629 | 0.541 |

- LogiQA shows moderate within-domain signal (AUC=0.662)
- LogiQA → HumanEval transfer works (0.629)
- HumanEval → LogiQA transfer weak (0.541)

**Note**: This is the opposite direction from error-direction transfer (which was HE→LQ).

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
