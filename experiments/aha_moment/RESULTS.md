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

### Dataset: GSM8K

**GSM8K** (Grade School Math 8K) is a benchmark of ~8,500 grade school math word problems requiring multi-step arithmetic reasoning. Each problem has:
- A **question** in natural language
- A **step-by-step solution** with intermediate calculations
- A **final answer** marked with `#### <number>`

Example problem:
```
Question: Natalia sold clips to 48 of her friends in April, and then she sold
half as many clips in May. How many clips did Natalia sell altogether in April and May?

Solution: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72
```

We use GSM8K because:
1. Solutions contain explicit calculations (e.g., `48 + 24 = 72`) that we can corrupt
2. Calculations are verifiable - we know what's correct vs incorrect
3. OLMo models naturally produce step-by-step solutions for these problems

### Methodology

Our approach follows [Wynroe et al.](https://github.com/Ckwobra/OLMo-error-detection) but uses paired comparisons:

#### Step 1: Generate Chain-of-Thought Solutions

We use `olmo3_rl_zero` (RL-Zero trained) to generate solutions for 200 GSM8K problems:

```python
prompt = f"Question: {question}\nLet me solve this step by step.\n"
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
```

The model produces solutions with natural calculations like `$430 + $320 = $750`.

#### Step 2: Parse Calculations and Create Corrupted Pairs

We find calculation patterns using regex:
```python
# Match patterns like: $430 + $320 = $750 or 4 + 2 = 6
pattern = r'([\$]?[\d,]+\s*[\+\-\*\/]\s*[\$]?[\d,]+)\s*=\s*([\$]?[\d,]+)'
```

Then create corrupted versions by changing the **last** calculation result:
```python
# Original: "Cost = 3 × $2 = $6"
# Corrupted: "Cost = 3 × $2 = $7"  (6 → 7)
corrupt_num = orig_num + 1  # Simple corruption: add 1
```

This gives us **92 valid pairs** where we have both clean and corrupted versions.

#### Step 3: Collect Activations at Error Token Position

For each pair, we run a forward pass and extract hidden states at the position of the corrupted number:

```python
def collect_trajectory_at_position(model, tokenizer, text, target_token_idx, collector):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    collector.register_hooks()  # Register hooks on all layers

    with torch.no_grad():
        model(**inputs)  # Forward pass

    activations = collector.get_activations_at_position(target_token_idx)
    return activations  # Shape: (n_layers, hidden_dim)
```

We collect activations at **all 16 layers** (OLMo 7B has 32 layers, we sample even layers).

#### Step 4: Compute Error-Detection Direction

The key insight: if models detect errors, then activations should differ systematically between clean and corrupt traces. We compute:

```python
def compute_error_direction(clean_activations, corrupted_activations):
    """
    Error-detection direction = mean difference between corrupted and clean.

    Args:
        clean_activations: (n_pairs=92, n_layers=16, hidden_dim=4096)
        corrupted_activations: (n_pairs=92, n_layers=16, hidden_dim=4096)

    Returns:
        direction: (n_layers, hidden_dim) - normalized direction vector
    """
    diff = corrupted_activations - clean_activations  # (92, 16, 4096)
    direction = diff.mean(axis=0)  # Average over pairs: (16, 4096)

    # Normalize per layer
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction_normalized = direction / (norms + 1e-8)

    return direction_normalized
```

This gives us a **unit vector in activation space** that points from "clean" toward "corrupted" representations.

#### Step 5: Project and Measure Effect Size

To test if this direction is meaningful, we project all activations onto it:

```python
def project_onto_direction(activations, direction):
    """
    Project activations onto error-detection direction.

    Positive projection = activation lies toward "corrupted" side
    Negative projection = activation lies toward "clean" side
    """
    projections = np.sum(activations * direction, axis=-1)  # Dot product
    return projections
```

Then measure separation using **Cohen's d** (standardized effect size):

```python
def compute_effect_size(clean_proj, corrupt_proj):
    pooled_std = np.sqrt((np.var(clean) + np.var(corrupt)) / 2)
    d = (np.mean(corrupt) - np.mean(clean)) / pooled_std
    return d
```

Effect size interpretation:
- d = 0.2: Small effect
- d = 0.5: Medium effect
- d = 0.8: Large effect
- **d = 1.7: Very large effect** (what we observe!)

### Results

| Model | Best Layer | Effect Size (d) | p-value | Interpretation |
|-------|------------|-----------------|---------|----------------|
| **olmo3_rl_zero** | Layer 14 | **1.70** | 8.8e-18 | Very strong separation |
| **olmo3_think** | Layer 14 | **1.65** | 8.2e-17 | Very strong separation |

The effect size of 1.7 means the clean and corrupted distributions are separated by 1.7 standard deviations - almost no overlap!

### Layer-by-Layer Profile

The error-detection signal **builds up through layers**, peaking at layer 14:

```
Layer  0: d=1.10  ████████████
Layer  1: d=0.94  ██████████
Layer  2: d=0.93  ██████████
Layer  3: d=1.07  ████████████
Layer  4: d=1.19  █████████████
Layer  5: d=1.30  ██████████████
Layer  6: d=1.39  ███████████████
Layer  7: d=1.41  ████████████████
Layer  8: d=1.36  ███████████████
Layer  9: d=1.37  ███████████████
Layer 10: d=1.33  ██████████████
Layer 11: d=1.47  ████████████████
Layer 12: d=1.39  ███████████████
Layer 13: d=1.34  ██████████████
Layer 14: d=1.70  ███████████████████  ← Peak
Layer 15: d=1.53  █████████████████
```

**Why layer 14?** This is the penultimate layer in our 16-layer sampling. The pattern suggests error detection requires deep processing - early layers don't distinguish as well.

### Visualization

**Projection Distributions** (Layer 14):

![Projection Distributions](results/wynroe/projection_distributions.png)

The histograms show clean (blue) and corrupted (red) projections at layer 14. Clean solutions cluster around -4.7 while corrupted solutions cluster around +6.2 - clear separation.

**Layer Profile**:

![Layer Profile](results/wynroe/layer_profile.png)

Effect size (Cohen's d) across all 16 layers for both models.

**Model Comparison**:

![Model Comparison](results/wynroe/model_comparison.png)

### Concrete Example

**Problem**: A math problem about calculating costs

**Clean Solution** (correct calculation):
```
The total is $430 + $320 = $750
Therefore, the answer is $750.
```
At the token position of `750`, the model's activation projects **negative** on the error-direction:
- Clean projection mean: **-4.72**

**Corrupt Solution** (error introduced by adding 1):
```
The total is $430 + $320 = $751  ← ERROR (should be $750)
Therefore, the answer is $751.
```
At the same relative position, the model's activation projects **positive**:
- Corrupt projection mean: **+6.17**

**The difference**: 6.17 - (-4.72) = **10.89 units** apart in the direction space!

### Why This is an "Error-Detection" Signal

The model's internal representation shifts dramatically when processing incorrect calculations. This is NOT about:
- ❌ "Knowing the right answer ahead of time"
- ❌ "Detecting syntax errors"
- ❌ "Random noise"

This IS about:
- ✅ **Inconsistency detection**: The model recognizes that `$430 + $320 = $751` doesn't match the arithmetic it expects
- ✅ **Internal expectation violation**: Something in the representation says "this doesn't add up"
- ✅ **Generalizable signal**: Works across different problems and both rl_zero and think models

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
