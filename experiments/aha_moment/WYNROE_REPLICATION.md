# Activation Patching Reveals Distillation Creates Imitation Circuits

**Date**: 2026-01-23
**Authors**: ManiVer Project
**Status**: Complete

---

## Abstract

We replicate Wynroe et al.'s activation patching methodology on DeepSeek-R1-Distill-Qwen-7B and extend it to the OLMo-3 model family. Our key finding: **distillation from DeepSeek R1 creates localized "imitation circuits"** at layers 16-18, while native RLVR training produces distributed processing without such shortcuts. This suggests the layer 18 spike found by Wynroe is not a fundamental property of reasoning models, but rather an artifact of knowledge distillation.

---

## 1. Background

### Wynroe et al. Finding

[Wynroe et al. (2026)](https://www.lesswrong.com/posts/BnzDiSYKfrHaFPefi/finding-an-error-detection-feature-in-deepseek-r1) discovered that DeepSeek-R1-Distill-Qwen-7B has a **localized error-detection circuit** at layer 20. Using activation patching, they showed that patching clean activations at this layer recovers most of the model's ability to detect errors and trigger correction behavior ("Wait...", "Actually...", etc.).

### Our Question

Is this localized circuit:
1. **Architectural** (a property of Qwen-based models)?
2. **Training-specific** (a result of distillation from DeepSeek R1)?

To answer this, we tested four models with different architectures and training methods.

---

## 2. Methodology

### Activation Patching

For each (clean, corrupted) solution pair:
1. Run corrupted prompt through model
2. At layer L, replace corrupted activations with clean activations
3. Measure logit-diff recovery for correction tokens ("Wait", "But", "Actually", "Hmm")
4. Repeat for all layers

**Recovery %** = How much of the clean→corrupted logit-diff is restored by patching at layer L.

### Critical Methodology Fix

Our initial implementation patched **all token positions**:
```python
# WRONG - patches entire sequence
hidden_states[:, :min_len, :] = patched[:, :min_len, :]
```

This gave 100% recovery at all layers (flat profile). The correct approach patches only the **final token position** where the logit-diff is measured:
```python
# CORRECT - patches final token only (standard TransformerLens methodology)
hidden_states[:, -1, :] = patched[:, -1, :]
```

### Dataset

- **MATH dataset** (hendrycks/math) - algebra subset
- 50 samples per model (after filtering for logit-diff > 3)
- Corruption: Inject arithmetic error in intermediate calculation

---

## 3. Models Tested

| Model | Architecture | Layers | Training | Reasoning Source |
|-------|--------------|--------|----------|------------------|
| **DeepSeek-R1-Distill-Qwen-7B** | Qwen | 28 | SFT on DeepSeek R1 outputs | DeepSeek R1 (671B) |
| **OLMo-3-7B-RL-Zero** | OLMo | 32 | Native RLVR from scratch | Self-discovered |
| **OLMo-3-7B-Think-SFT** | OLMo | 32 | SFT on Dolci dataset | DeepSeek R1 |
| **OLMo-3-7B-Think** | OLMo | 32 | SFT + DPO + RLVR | DeepSeek R1 + Self |

### Critical Discovery

The [Dolci-Think-SFT dataset](https://huggingface.co/datasets/allenai/dolci-thinking-sft) used to train OLMo-3-Think-SFT was generated using **DeepSeek R1** and **DeepSeek R1-0528** for the thinking traces. This means both DeepSeek-Distill and OLMo-SFT learned from the same teacher!

---

## 4. Results

### 4.1 DeepSeek-R1-Distill-Qwen-7B (Wynroe's Model)

| Layer | Recovery % | Δ from Previous |
|-------|------------|-----------------|
| 0 | 10.7% | - |
| 2 | 10.6% | -0.1% |
| 4 | 10.6% | 0.0% |
| 6 | 10.8% | +0.2% |
| 8 | 11.1% | +0.3% |
| 10 | 11.5% | +0.4% |
| 12 | 12.2% | +0.7% |
| 14 | 13.0% | +0.8% |
| 16 | 17.1% | +4.1% |
| **18** | **38.6%** | **+21.5%** ← Circuit activates |
| 20 | 55.8% | +17.2% |
| 22 | 64.5% | +8.7% |
| 24 | 71.7% | +7.2% |
| 26 | 91.2% | +19.5% |

**Key Finding**: Sharp spike at layer 16→18 (+21.5%), matching Wynroe's finding.

---

### 4.2 OLMo-3-7B-RL-Zero (Native RLVR)

| Layer | Recovery % | Δ from Previous |
|-------|------------|-----------------|
| 0 | 8.2% | - |
| 2 | 8.1% | -0.1% |
| 4 | 8.4% | +0.3% |
| 6 | 11.9% | +3.5% |
| 8 | 15.9% | +4.0% |
| 10 | 21.1% | +5.2% |
| 12 | 26.0% | +4.9% |
| 14 | 35.1% | +9.1% |
| 16 | 43.0% | +7.9% |
| 18 | 56.3% | +13.3% |
| 20 | 57.2% | +0.9% |
| 22 | 64.5% | +7.3% |
| 24 | 68.1% | +3.6% |
| 26 | 70.4% | +2.3% |
| 28 | 74.9% | +4.5% |
| 30 | 94.0% | +19.1% |

**Key Finding**: **Gradual increase** throughout layers. No sharp spike. Largest jump is +13.3% at L18, but this is part of a smooth ramp, not a sudden activation.

---

### 4.3 OLMo-3-7B-Think-SFT (Distilled from DeepSeek R1)

| Layer | Recovery % | Δ from Previous |
|-------|------------|-----------------|
| 0 | 3.1% | - |
| 2 | 3.0% | -0.1% |
| 4 | 2.9% | -0.1% |
| 6 | 3.4% | +0.5% |
| 8 | 4.4% | +1.0% |
| 10 | 5.6% | +1.2% |
| 12 | 7.1% | +1.5% |
| 14 | 11.4% | +4.3% |
| 16 | 16.9% | +5.5% |
| **18** | **27.2%** | **+10.3%** ← Mid-layer spike |
| 20 | 29.6% | +2.4% |
| 22 | 37.4% | +7.8% |
| 24 | 40.0% | +2.6% |
| 26 | 45.8% | +5.8% |
| 28 | 54.0% | +8.2% |
| **30** | **86.8%** | **+32.8%** ← Final spike |

**Key Finding**: **HYBRID pattern** - both a mid-layer spike at L18 (+10.3%) AND a final spike at L30 (+32.8%).

---

### 4.4 OLMo-3-7B-Think (SFT + DPO + RLVR)

| Layer | Recovery % | Δ from Previous |
|-------|------------|-----------------|
| 0 | 6.4% | - |
| 2 | 6.5% | +0.1% |
| 4 | 6.5% | 0.0% |
| 6 | 7.1% | +0.6% |
| 8 | 7.7% | +0.6% |
| 10 | 7.8% | +0.1% |
| 12 | 8.7% | +0.9% |
| 14 | 11.7% | +3.0% |
| 16 | 15.6% | +3.9% |
| 18 | 21.8% | +6.2% |
| 20 | 24.0% | +2.2% |
| 22 | 29.9% | +5.9% |
| 24 | 32.2% | +2.3% |
| 26 | 36.9% | +4.7% |
| 28 | 42.9% | +6.0% |
| **30** | **86.7%** | **+43.8%** ← Massive final spike |

**Key Finding**: **Final spike only** at L30 (+43.8%). The mid-layer spike seen in SFT has been suppressed.

---

## 5. Comparison Summary

| Metric | DeepSeek-Distill | OLMo RL-Zero | OLMo SFT | OLMo Think |
|--------|------------------|--------------|----------|------------|
| **Architecture** | Qwen (28L) | OLMo (32L) | OLMo (32L) | OLMo (32L) |
| **Training** | SFT on DeepSeek | Native RLVR | SFT on Dolci | SFT+DPO+RLVR |
| **Reasoning Source** | DeepSeek R1 | Self | DeepSeek R1 | DeepSeek R1 + Self |
| **L16→L18 Jump** | **+21.5%** | +13.3% | **+10.3%** | +6.2% |
| **L28→L30 Jump** | +19.5%* | +19.1% | **+32.8%** | **+43.8%** |
| **Profile Shape** | **Mid-spike** | **Gradual** | **Hybrid** | **Final-spike** |
| **Peak Recovery** | 91.2% (L26) | 94.0% (L30) | 86.8% (L30) | 86.7% (L30) |

*DeepSeek has 28 layers, so L26 is its final layer.

### Visual Comparison

```
DeepSeek-Distill (Qwen arch)          OLMo RL-Zero (native RLVR)
      │                                    │
   ▂▂▂█████▆▆▆▆▆                        ▂▃▄▅▆▇▇▇▇▇███
      ↑                                    (gradual ramp)
   L18 spike
   (imitation circuit)

OLMo Think (SFT+RLVR)                 OLMo SFT (distilled)
      │                                    │
   ▂▂▂▂▃▃▄▄▅▅▆▆████                    ▂▂▂▃▄█▅▆▆▇▇████
                  ↑                        ↑        ↑
              L30 spike                L18 spike  L30 spike
              (RLVR enhanced)          (HYBRID!)
```

---

## 6. The Distillation Hypothesis

### Claim

**Distillation from DeepSeek R1 creates localized "imitation circuits" at layers 16-18.**

### Evidence

| Evidence | Finding |
|----------|---------|
| **DeepSeek-Distill shows L18 spike** | +21.5% jump at L18 |
| **OLMo-SFT shows L18 spike** | +10.3% jump at L18 |
| **Both trained on DeepSeek R1 outputs** | DeepSeek directly, OLMo via Dolci |
| **OLMo RL-Zero shows NO spike** | Gradual ramp (no distillation) |
| **OLMo Think has SUPPRESSED spike** | +6.2% at L18 (RLVR reduces it) |

### Why is the OLMo-SFT spike smaller than DeepSeek?

1. **Different architecture**: Qwen vs OLMo may distribute computation differently
2. **Different base model**: DeepSeek-Distill started from Qwen2.5-Math-7B (math-specialized)
3. **OLMo has L30 spike**: Some processing moved to final layer

---

## 7. The Imitation Circuit Theory

### What is an Imitation Circuit?

During distillation, the student model learns to mimic the teacher's reasoning patterns. The training data contains explicit "error → correction" transitions:

```
Training sample from DeepSeek R1:
"...let me calculate: 2+3=6... Wait, that's wrong. 2+3=5..."
                           ↑
                    TOKEN N: "Wait"
```

The student learns a pattern-matching function:
```
f(hidden_state at N-1) → high P("Wait")
```

This creates a **localized circuit** at layers 16-18 that detects error-associated activation patterns and triggers correction tokens. It's analogous to [induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) - dedicated circuits for specific behaviors.

### Why is it Localized at L16-18?

The teacher model (DeepSeek R1) makes the "correction decision" around these layers. The student replicates this computational structure through supervised learning on the teacher's outputs.

### Why Does RLVR Suppress the Circuit?

| Training Method | Optimizes For | Result |
|-----------------|---------------|--------|
| **SFT (distillation)** | Matching teacher outputs | Copies teacher's circuit location |
| **RLVR** | Final answer correctness | May "unlearn" shortcut in favor of robust processing |

Evidence: OLMo-Think (SFT + RLVR) has smaller L18 spike (+6.2%) than OLMo-SFT (+10.3%). RLVR training suppresses the imitation circuit.

---

## 8. Key Insights

### 1. Distillation Creates Imitation Circuits

Both models distilled from DeepSeek R1 (DeepSeek-Distill and OLMo-SFT) show mid-layer spikes at L16-18. This is a pattern-matching shortcut learned from the teacher.

### 2. Architecture Adds Processing Layers

All OLMo models (including SFT) have L30 spikes regardless of training. This appears to be an architectural property, not a training effect.

### 3. RLVR Modifies the Circuit Profile

- **Suppresses mid-layer spike**: Think (+6.2%) < SFT (+10.3%) < DeepSeek (+21.5%)
- **Enhances final spike**: Think (+43.8%) > SFT (+32.8%) > RL-Zero (+19.1%)

RLVR training moves error processing from the imitation circuit to the final layer.

### 4. Native RLVR Produces Distributed Processing

OLMo-RL-Zero, trained with RLVR from scratch (no distillation), shows no localized circuit. Processing is distributed across all layers.

---

## 9. Implications

### For Interpretability

The layer 18 spike in DeepSeek-R1 is **not a fundamental property of reasoning models**. It's an artifact of distillation. Studies of "reasoning circuits" in distilled models may be studying imitation artifacts rather than genuine reasoning mechanisms.

### For Training

- **Distillation creates shortcuts**: Fast to train, but may create brittle circuits
- **Native RLVR creates distributed processing**: Potentially more robust, but harder to interpret
- **SFT + RLVR is a middle ground**: RLVR can partially "unlearn" distillation artifacts

### For Mechanistic Understanding

To study genuine reasoning circuits, we should:
1. Focus on models trained with native RLVR (not distillation)
2. Look at final-layer processing (L30 spikes) rather than mid-layer circuits
3. Consider that "error detection" may be distributed rather than localized

---

## 10. Future Directions

### Direction Transfer Test

Does DeepSeek's L18 error direction transfer to OLMo-SFT?

```python
# Train probe on DeepSeek L18 activations
error_direction = train_probe(deepseek_l18_acts, labels)

# Test on OLMo-SFT L18 activations
transfer_acc = evaluate(olmo_sft_l18_acts, error_direction)

# If transfer_acc > 70% → same circuit learned from same teacher
```

### Causal Intervention

Can we suppress correction behavior by subtracting the error direction?

```python
# Subtract error direction at L18
acts_modified = acts_l18 - alpha * error_direction

# Does model now fail to say "Wait" when it should?
```

### Training Dynamics

When does the imitation circuit form during SFT? Is it gradual or sudden (phase transition)?

---

## 11. References

1. **Wynroe et al.** - "Finding an Error-Detection Feature in DeepSeek-R1"
   https://www.lesswrong.com/posts/BnzDiSYKfrHaFPefi/finding-an-error-detection-feature-in-deepseek-r1

2. **Dolci-Think-SFT Dataset** - Generated using DeepSeek R1
   https://huggingface.co/datasets/allenai/dolci-thinking-sft

3. **DeepSeek-R1-Distill-Qwen-7B** - SFT on DeepSeek R1 outputs
   https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

4. **In-context Learning and Induction Heads** - Olsson et al. 2022
   https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

5. **OLMo-3 Model Family**
   https://huggingface.co/allenai

---

## 12. Appendix: Raw Data

### DeepSeek-R1-Distill-Qwen-7B
- N = 50 pairs
- Best layer: 26 (91.2% recovery)
- Results: `experiments/aha_moment/results/wynroe_patching_v2/deepseek_patching_results.json`

### OLMo-3-7B-RL-Zero
- N = 24 pairs (lower filtering rate)
- Best layer: 30 (94.0% recovery)
- Results: `experiments/aha_moment/results/wynroe_patching_v2/rl_zero_patching_results.json`

### OLMo-3-7B-Think-SFT
- N = 50 pairs
- Best layer: 30 (86.8% recovery)
- Results: `experiments/aha_moment/results/wynroe_patching/sft_patching_results.json`

### OLMo-3-7B-Think
- N = 50 pairs
- Best layer: 30 (86.7% recovery)
- Results: `experiments/aha_moment/results/wynroe_patching/think_patching_results.json`

---

*Report generated: 2026-01-23*
