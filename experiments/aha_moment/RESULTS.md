# Aha Moment Experiment Results

## Executive Summary

We ran two experiments to investigate "error-detection" signals in LLM activation trajectories:

| Experiment | Question | Result | Effect Size |
|------------|----------|--------|-------------|
| **A: Wynroe Replication** | Can models detect errors in preceding context? | **Yes, strongly** | d=1.70 |
| **B: Natural Pivots** | Do self-correction phrases have distinct geometry? | **Partial** - opposite of hypothesis | d=-0.31 |

**Key Finding**: Models have strong internal "error detectors" that can identify mistakes in context (Experiment A). However, natural self-correction phrases ("Wait...", "But...") show *lower* velocity and *more linear* trajectories than random positions (Experiment B) - suggesting pivots are "reflection pauses" rather than sharp directional changes.

---

## Experiment A: Wynroe-Style Error Detection

### What We Tested
Following [Wynroe et al.](https://github.com/Ckwobra/OLMo-error-detection), we tested whether OLMo models can detect errors in preceding context. We:

1. Took GSM8K math problems with solutions
2. Created pairs: **clean** (correct solution) vs **corrupt** (same solution with one calculation error)
3. Computed the "error-detection direction" = mean(corrupt) - mean(clean)
4. Projected all activations onto this direction at the error position

### Results

| Model | Best Layer | Effect Size (d) | p-value |
|-------|------------|-----------------|---------|
| **olmo3_rl_zero** | Layer 14 | **1.70** | 8.8e-18 |
| **olmo3_think** | Layer 14 | **1.65** | 8.2e-17 |

**Interpretation**: Effect sizes >1.5 are considered "very large". The models clearly distinguish between correct and incorrect preceding calculations.

### Layer Profile (olmo3_rl_zero)

```
Layer  0: d=1.10  ████████████
Layer  4: d=1.19  █████████████
Layer  8: d=1.36  ███████████████
Layer 11: d=1.47  ████████████████
Layer 14: d=1.70  ███████████████████  ← Peak
Layer 15: d=1.53  █████████████████
```

The error-detection signal builds up through layers, peaking at layer 14 (out of 16).

### Example: Clean vs Corrupt

**Problem**: "A grocery store sells 5 apples for $2. How much do 15 apples cost?"

**Clean Solution** (correct):
```
15 apples = 3 × 5 apples
Cost = 3 × $2 = $6
```
→ Model activation projects **negative** on error-direction (-4.72 mean)

**Corrupt Solution** (error introduced):
```
15 apples = 3 × 5 apples
Cost = 3 × $2 = $8  ← ERROR (should be $6)
```
→ Model activation projects **positive** on error-direction (+6.17 mean)

**Why This is an "Error-Detection" Signal**:
The model's internal representation shifts dramatically when processing incorrect calculations. This is not about "knowing the right answer" but about detecting *inconsistency* - the model recognizes that "3 × $2 = $8" doesn't match what it expects from the preceding context.

---

## Experiment B: Natural Pivot Detection

### What We Tested
We investigated whether self-correction phrases ("Wait...", "But...", "Actually...") have distinctive geometric properties in activation trajectories.

### Data Collection
- Model: olmo3_think (reasoning-trained)
- Samples: 200 GSM8K problems with generated solutions
- Pivots detected: 234 total across 62 samples

**Pivot Pattern Distribution**:
| Pattern | Count |
|---------|-------|
| BUT | 101 |
| Wait | 48 |
| however | 45 |
| hmm | 19 |
| actually | 8 |
| BUT_WAIT | 5 |

### Results

| Metric | Pivot Mean | Random Mean | Effect Size (d) | p-value | Interpretation |
|--------|------------|-------------|-----------------|---------|----------------|
| **Velocity** | 14.90 | 15.42 | **-0.22** | 0.019 | Pivots are *slower* |
| Direction Change | 1.37 | 1.38 | -0.05 | 0.55 | No difference |
| Lyapunov | 1.64 | 1.64 | -0.01 | 0.90 | No difference |
| Menger Curvature | 0.111 | 0.109 | 0.17 | 0.08 | Slight trend (NS) |
| **Gaussian Proxy** | 0.757 | 0.768 | **-0.31** | 0.001 | Pivots are *more linear* |

### Surprising Finding

**Our hypothesis was wrong!** We expected:
- Pivots to have *higher* velocity (sharp directional changes)
- Pivots to have *higher* curvature (trajectory bending)

**What we actually found**:
- Pivots have **lower** velocity (d=-0.22) - the model *slows down*
- Pivots have **more linear** trajectories (d=-0.31) - less bending, not more

### Interpretation: Pivots as "Reflection Pauses"

The data suggests that self-correction phrases mark moments where the model:
1. **Pauses** - moves more slowly through activation space
2. **Linearizes** - follows more direct, less curved paths

This makes intuitive sense: when the model says "Wait..." or "But...", it's not making a dramatic turn. Instead, it's *consolidating* and *re-evaluating* - which manifests as smoother, slower movement.

### Example Pivots with Curvature Values

**Example 1**: Sales tax problem
```
Cost = $60 - $18 = $48. Therefore, Joe will have $50 - $48 = $2 left.

Answer: \boxed{2}

Wait, but hold on a second. The problem says "assuming that sales
tax is included." Hmm, so does that mean that the $48 already
includes tax, or is the tax applied after the discount?
```
- **Pattern**: "Wait, but"
- **Menger curvature**: 0.095 (vs random mean 0.109)
- **Gaussian proxy**: 0.730 (vs random mean 0.768) → More linear

**Example 2**: Diaper changes problem
```
So Jordan does the other half: 10 - 5 = 5 changes.

Yes, that seems right. The problem says "Jordan's wife changes half
of the diapers," so it's half of the total changes, not half per
child or something else. So the calculation is straightforward.

Wait, but the problem doesn't specify per child, just total.
```
- **Pattern**: "Wait, but"
- **Menger curvature**: 0.124
- **Gaussian proxy**: 0.795

**Example 3**: Auditorium problem
```
Step 5: Students occupy the rest. So 54 minus 18 is 36. That makes sense.

Wait a second, let me make sure about the fractions. The problem says
"One-third of the remaining seats were occupied by the parents..."
```
- **Pattern**: "Wait"
- **Menger curvature**: 0.115
- **Gaussian proxy**: 0.780

---

## What Makes These "Error-Detection" Signals?

### Experiment A: External Error Detection

The Wynroe-style error direction detects **external errors** - mistakes in the input/context that the model is processing. This is analogous to:
- A human noticing a typo in someone else's writing
- A calculator flagging an impossible result
- A spell-checker highlighting a misspelled word

**The signal is about inconsistency detection**: The model's representation shifts when the context contains something that doesn't fit its learned expectations.

### Experiment B: Internal Uncertainty Signals

Natural pivots mark moments of **internal re-evaluation** - when the model recognizes it should double-check its own reasoning. This is analogous to:
- A human saying "Wait, let me think about that again..."
- Pausing mid-sentence to reconsider a claim
- The feeling of "something's not quite right"

**The signal is about metacognitive pause**: The slower, more linear trajectories suggest the model is in a consolidation state, not a dramatic directional change.

---

## Connection to Error Detection

Both experiments reveal different aspects of error-related processing:

| Aspect | Experiment A | Experiment B |
|--------|--------------|--------------|
| **Error Type** | External (in context) | Internal (in own reasoning) |
| **Timing** | At error position | At self-correction phrase |
| **Signal** | Directional (positive projection) | Dynamic (slower, linear) |
| **Strength** | Very strong (d=1.7) | Moderate (d=-0.3) |

The strong Experiment A result suggests models have robust error-detection capabilities for external inputs. The weaker Experiment B result suggests self-correction phrases are *markers* of internal uncertainty but don't have dramatically different trajectory geometry.

---

## Implications

1. **Error detection is real**: Models can detect calculation errors in context with high reliability (d=1.7).

2. **Pivots ≠ sharp turns**: Self-correction phrases don't represent dramatic trajectory changes. Instead, they mark "slow, linear" consolidation moments.

3. **Different mechanisms**: External error detection (Experiment A) and internal self-correction (Experiment B) appear to be distinct phenomena with different geometric signatures.

4. **Future work**:
   - Can we use the error-detection direction to *improve* model outputs?
   - Are "slow, linear" regions predictive of answer quality?
   - Do pivots in correct vs incorrect solutions differ?

---

## Files

### Experiment A (Wynroe Replication)
- Data: `data/wynroe_replication/wynroe_trajectories.h5` (on box1)
- Results: `results/wynroe/wynroe_analysis.json`
- Plots: `results/wynroe/*.png`

### Experiment B (Natural Pivots)
- Data: `data/pivot_collection/pivot_trajectories.h5` (on eyecog)
- Results:
  - `results/pivot_analysis/pivot_analysis.json` (velocity, direction, Lyapunov)
  - `results/pivot_analysis/curvature_analysis.json` (Menger, Gaussian)
  - `results/pivot_analysis/pivot_examples.json` (text examples)
- Plots: `results/pivot_analysis/*.png`
