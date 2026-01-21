# Menger Curvature Analysis Findings

**Date**: 2026-01-20 (Updated with critical correction)
**Model**: olmo3_base (0-shot)
**Tasks**: HumanEval, LogiQA

---

## Summary

We computed Menger curvature across layer trajectories and found:

1. **Within-domain**: Curvature does NOT significantly distinguish correct vs incorrect (p > 0.2)
2. **Cross-domain**: Curvature profiles are HIGHLY correlated (r = 0.996, p < 0.0001)

**CRITICAL UPDATE**: The r=0.996 finding is a **RED HERRING / NULL RESULT**!

Further analysis showed that curvature profiles are identical across:
- Correct vs Incorrect (within domain): r = 0.9999
- Correct vs Correct (cross domain): r = 0.9961
- All pairwise combinations: r > 0.995

This means curvature profile is an **architectural property** of the transformer, not a signal related to reasoning or correctness.

---

## Methods

### Menger Curvature Definition

Menger curvature measures the curvature of a path through three consecutive points. For points P1, P2, P3:

```
curvature = 4 * Area(P1, P2, P3) / (|P1-P2| * |P2-P3| * |P1-P3|)
```

Where Area is computed via cross product magnitude.

### Implementation

```python
def compute_menger_curvature(trajectory):
    """
    Compute mean Menger curvature for a layer-wise trajectory.

    Args:
        trajectory: (n_layers, d_model) - activations at each layer

    Returns:
        mean_curvature: float
    """
    n_layers = trajectory.shape[0]
    curvatures = []

    for i in range(n_layers - 2):
        p1 = trajectory[i]
        p2 = trajectory[i + 1]
        p3 = trajectory[i + 2]

        # Side lengths
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        # Avoid division by zero
        if a * b * c < 1e-10:
            continue

        # Heron's formula for area
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        area = np.sqrt(max(0, area_sq))

        # Menger curvature
        curvature = 4 * area / (a * b * c)
        curvatures.append(curvature)

    return np.mean(curvatures) if curvatures else 0.0
```

### Data Processing

1. Load trajectories from HDF5 files
2. Average across sequence dimension: `(n_samples, 512, 16, 4096) -> (n_samples, 16, 4096)`
3. Compute curvature for each sample's layer trajectory
4. Split by correctness label (`is_correct` field)

---

## Results (Updated 2026-01-20, N=100 per task)

### Within-Domain Analysis

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

Our original H2 hypothesis asked: "Do linear directions that separate correct/incorrect transfer across domains?"

**Answer**: No, linear directions do NOT transfer (cross-domain AUC ~ 52%, chance level).

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

## Data Limitations

This analysis was conducted on **limited data**:

| Factor | Limitation |
|--------|------------|
| Model | Single model (olmo3_base) |
| Tasks | 2 tasks (HumanEval, LogiQA) - GSM8K not included |
| Samples | 50 per task (memory constraints) |
| HumanEval balance | Only 5/50 correct (10%) - severely imbalanced |
| LogiQA balance | 12/50 correct (24%) - acceptable |

**Recommendations for robust analysis**:
1. Run on all 4 models (need to fix corrupted LogiQA files first)
2. Include GSM8K for 3-way cross-domain correlation
3. Use full 500 samples (requires chunked processing or more RAM)
4. Balance samples or use stratified statistics

---

## Connection to Zhou et al. (2025)

Our findings align with Zhou et al. "The Geometry of Reasoning" (arXiv:2510.09782):

> "Menger curvature (second-order geometric structure) captures logical relationships between reasoning steps better than first-order metrics."

Our observation that curvature profiles correlate across domains supports their claim that curvature captures something fundamental about reasoning structure, not just surface task characteristics.

---

## Code Reference

Full implementation: [scripts/analysis/phase3_dynamical_analysis.py](../../scripts/analysis/phase3_dynamical_analysis.py)

Key functions:
- `compute_menger_curvature()` - Lines 145-175
- `menger_curvature_analysis()` - Lines 230-280
- Cross-domain correlation computed using `scipy.stats.pearsonr`

---

## Next Steps

1. **Fix data issues** (see DATA_COLLECTION_ISSUES.md)
2. **Run on all models** with fixed data
3. **Add GSM8K** to test 3-way cross-domain correlation
4. **Compute curvature derivatives** - do changes in curvature (not just magnitude) differ?
5. **Test predictive power** - can curvature profile + correctness train a cross-domain classifier?
