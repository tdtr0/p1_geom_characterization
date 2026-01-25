
# Phase 3 Findings: H1/H2 Hypothesis Testing

**Date**: 2026-01-21
**Models**: olmo3_base, olmo3_sft, olmo3_rl_zero, olmo3_think
**Tasks**: GSM8K (math), HumanEval (code)
**Samples**: 150 per task per model
**Data**: `/data/thanhdo/trajectories_0shot/`

---

## Methodology

### Error-Detection Direction Analysis
- **Method**: Compute difference-in-means between correct and incorrect activations at each layer
- **Direction**: `d = mean(correct) - mean(incorrect)`, normalized
- **Classifier**: Project activations onto direction, threshold at median
- **Transfer test**: Train classifier on task A, test on task B

### Menger Curvature
- **Method**: Compute curvature at each layer transition using 3 consecutive points
- **Formula**: `κ = 4A / (|a||b||c|)` where A = triangle area, a,b,c = side lengths

### Statistical Tests
- Cohen's d for effect size
- t-test for significance (p < 0.05)

---

## Task Accuracy by Model

| Model | Training | GSM8K | HumanEval |
|-------|----------|-------|-----------|
| base | Pre-train | 12.6% | 3.8% |
| sft | SFT | **59.4%** | 4.8% |
| rl_zero | RL-Zero | 14.0% | **13.4%** |
| think | SFT+DPO+RLVR | 39.4% | 5.0% |

---

## H1: Within-Domain Classification

| Model | GSM8K Layer | GSM8K Acc | GSM8K d | HE Layer | HE Acc | HE d |
|-------|-------------|-----------|---------|----------|--------|------|
| base | 28 | **88.0%** | 2.17 | 24 | **70.7%** | 1.06 |
| sft | 26 | 63.3% | 0.79 | 26 | 60.0% | 0.69 |
| rl_zero | 26 | 84.0% | 1.80 | 24 | 60.0% | 0.74 |
| think | **2** | 81.3% | 1.83 | 26 | 56.0% | 0.55 |

**Observations**:
- Base has strongest separation despite lowest task accuracy
- Think localizes GSM8K at layer 2 (very early)
- SFT has weakest separation despite highest task accuracy

---

## H2: Cross-Domain Transfer

### Math ↔ Code (GSM8K ↔ HumanEval)

| Model | GSM8K → HE | HE → GSM8K | Pattern |
|-------|------------|------------|---------|
| base | **85.3%** | 10.0% | Asymmetric |
| sft | **66.0%** | **56.0%** | **Bidirectional** |
| rl_zero | 41.3% | 18.0% | None |
| think | 22.7% | 44.7% | None |

### Cross-Domain Subspace Alignment (2026-01-25)

**Method**: Compute error direction for each task, measure cosine similarity and cross-transfer effect.

**Results by Model (n=300 per task):**

| Model | Task Accuracy | Cosine Sim | GSM8K→LogiQA d | LogiQA→GSM8K d | Pattern |
|-------|---------------|------------|----------------|----------------|---------|
| **base** | 12%/23% | 0.069 | 0.08 (p=0.46) | 0.18 (p=0.20) | Orthogonal |
| **sft** | 60%/36% | **0.355** | **0.48 (p=0.0009)** | **0.45 (p=0.0003)** | **Bidirectional** |
| rl_zero | TBD | TBD | TBD | TBD | TBD |
| think | TBD | TBD | TBD | TBD | TBD |

**Key Finding**: SFT shows **5x higher alignment** (cos=0.355 vs 0.069) and **significant bidirectional transfer** (p<0.001 both ways).

**Interpretation**:
- **Base model**: Task-specific error directions (orthogonal, no transfer)
- **SFT model**: Learns more generalizable error patterns that transfer across domains
- This supports the **SFT distillation hypothesis**: SFT training on CoT data creates domain-general representations

### Token-Position Specificity (2026-01-25)

**Question**: Where in the sequence does the error signal peak?

| Task | Peak Position | Fraction | Signal Location | Early d | Middle d | Late d |
|------|---------------|----------|-----------------|---------|----------|--------|
| GSM8K | 50/512 | 10% | **Early** | 0.201 | 0.003 | 0.000 |
| LogiQA | 85/512 | 17% | **Early** | 0.220 | 0.211 | 0.000 |

**Key Finding**: Signal peaks in **early tokens** (10-17%), NOT in answer tokens.

**Interpretation**:
- If signal was in late tokens → detecting answer format
- Signal is in **early** tokens → detecting initial problem encoding/setup
- This suggests the error direction captures how the model *starts* processing, not how it *finishes*

### Implications for H2

1. **No Universal Reasoning Signature**: Error directions are task-specific (cos=0.145)
2. **Surface Structure Confirmed**: Different tasks have orthogonal error patterns
3. **Early Token Signal**: The discriminative signal is in problem setup, not answer format
4. **Transfer Failure Explained**: Cross-domain transfer fails because error directions don't align

---

## Menger Curvature

| Model | GSM8K d | GSM8K p | HE d | HE p |
|-------|---------|---------|------|------|
| base | 0.145 | 0.595 | 0.200 | 0.428 |
| **sft** | **0.531** | **0.002** | **0.751** | **0.002** |
| rl_zero | 0.162 | 0.448 | 0.272 | 0.103 |
| think | 0.045 | 0.784 | 0.326 | 0.159 |

**Only SFT shows significant curvature differences** (p < 0.01 for both tasks).

---

## Summary Table

| Property | Base | SFT | RL-Zero | Think |
|----------|------|-----|---------|-------|
| Task accuracy | Low | High | Medium | High |
| Within-domain separation | **Best** | Weak | Strong | Strong |
| Cross-domain transfer | Asymmetric | **Bidirectional** | None | None |
| Curvature significant | No | **Yes** | No | No |

### Cross-Domain Alignment Summary

| Model | Cosine Sim | Transfer Pattern | Interpretation |
|-------|------------|------------------|----------------|
| **base** | 0.069 | None | Task-specific (orthogonal) |
| **sft** | **0.355** | **Bidirectional** | Domain-general alignment |
| rl_zero | TBD | TBD | Pending |
| think | TBD | TBD | Pending |

**Key Conclusion**: Training method determines whether error directions generalize:
- **Base**: Task-specific patterns (surface structure)
- **SFT**: Learns transferable error patterns (domain-general)

Rerunning for rl_zero and think models.

---

## Discussion: Why Does SFT Transfer?
Validates h1 - nothing new. 
### The SFT Distillation Hypothesis

SFT models are typically trained on chain-of-thought data generated by larger reasoning models (e.g., GPT-4, Claude). This creates a form of **knowledge distillation** where:

1. **Higher-rank representations**: The SFT training signal comes from a more capable model that has richer, more general representations across domains

2. **Shared reasoning patterns**: CoT data from strong models encodes domain-general reasoning patterns (decomposition, verification, correction) rather than task-specific shortcuts

3. **Geometric implication**: SFT learns to traverse similar manifold regions regardless of task domain, producing bidirectional transfer

### RL-Zero: Domain Specialization

RL-Zero optimizes directly for task-specific rewards without intermediate supervision:

1. **Reward hacking**: The model finds domain-specific shortcuts that maximize reward without generalizable reasoning

2. **Low-rank task circuits**: Creates efficient but narrow pathways for each task type

3. **No transfer**: High within-domain accuracy (84%) but representations don't generalize

### Base Model: Emergent Asymmetry

The pretrained base model shows asymmetric transfer (math→code works, code→math fails):

1. **Math subsumes code**: Mathematical reasoning patterns may be more general, encompassing code-like logical patterns

2. **Code is specialized**: Code generation requires specific syntax/API knowledge not captured by math

### Think Model: Verbose Noise

Despite SFT+DPO+RLVR training, think model shows no transfer:

1. **Early layer localization**: GSM8K signal at layer 2 is anomalous (others: layer 26-28)

2. **Token dilution**: 10K+ reasoning tokens may obscure the underlying geometric signal

3. **Possible interpretation**: Verbose thinking adds noise that masks transferable patterns


This just shows that h1 is true, and validates h1. nothing new. 
---

## Implications

1. **For verification**: SFT models are better candidates for cross-domain verifiers
2. **For training**: RL-Zero creates specialists; SFT creates generalists
3. **For interpretability**: The geometric signature of "correct" depends on training method, not just correctness
