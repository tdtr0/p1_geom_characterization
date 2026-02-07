# RLVR Potential Index Report

**Generated**: 2026-02-06
**Data Source**: `/data/thanhdo/trajectories_0shot/`

## Executive Summary

The RLVR Potential Index framework measures how "ready" a base model is for RL training based on geometric properties of its activations. Key finding: **RL-Zero preserves base geometry (high GAS, high CKA) while SFT dramatically transforms it** - but SFT achieves higher accuracy on GSM8K.

---

## Metrics Explained

| Metric | Description | Formula | Range | Interpretation |
|--------|-------------|---------|-------|----------------|
| **LCS** | Latent Capability Score | probe_AUC(base) | [0.5, 1.0] | How much base model "knows" about task |
| **GAS** | Geometric Alignment Score | \|cos(e_base, e_trained)\| | [0, 1] | Does base geometry predict training direction? |
| **ID** | Intrinsic Dimensionality | effective_rank(activations) | [1, d_model] | Complexity of representation (lower = more compressed) |
| **CKA** | Centered Kernel Alignment | CKA(base, trained) | [0, 1] | Representation similarity base → trained |
| **TER** | Training Efficiency Ratio | Δacc / (1 - CKA) | [0, ∞) | Accuracy gain per unit geometry change |
| **RRS** | RLVR Readiness Score | LCS × GAS × (1/log(ID)) | [0, ∞) | Composite readiness metric |

---

## Results by Task

### GSM8K (Math Reasoning)

**Base Model (OLMo-3-Base):**
- Accuracy: **14.0%** (14/100)
- Latent Capability Score (LCS): **0.647** (moderate)
- Intrinsic Dimensionality (ID): **68.16**

| Model | Accuracy | Δ Acc | GAS | CKA | TER | RRS |
|-------|----------|-------|-----|-----|-----|-----|
| **rl_zero** | 14.0% | +0.0% | **0.660** | 0.995 | 0.00 | **0.101** |
| sft | 64.0% | +50.0% | 0.070 | 0.815 | 2.70 | 0.011 |
| think | 39.0% | +25.0% | 0.183 | 0.818 | 1.37 | 0.028 |

**Key Observations:**
- RL-Zero has **highest GAS (0.66)** but **zero accuracy improvement**
- SFT has **lowest GAS (0.07)** but **highest accuracy (+50%)**
- CKA shows RL-Zero barely changes representations (0.995), while SFT/Think change more (0.81-0.82)

---

### HumanEval (Code Generation)

**Base Model (OLMo-3-Base):**
- Accuracy: **2.0%** (2/100) - very low baseline
- Latent Capability Score (LCS): **NaN** (too few correct samples for probe)
- Intrinsic Dimensionality (ID): **18.94** (much lower than GSM8K)

| Model | Accuracy | Δ Acc | GAS | CKA |
|-------|----------|-------|-----|-----|
| **rl_zero** | 13.0% | +11.0% | **0.797** | 1.000 |
| sft | 6.0% | +4.0% | 0.674 | 0.992 |
| think | 6.0% | +4.0% | 0.620 | 0.992 |

**Key Observations:**
- RL-Zero again has **highest GAS (0.80)** AND **highest accuracy improvement (+11%)**
- CKA is near-perfect for all models (0.99-1.00) - minimal representation change
- Lower intrinsic dimensionality (18.94 vs 68.16) suggests code is represented more compactly

---

## Cross-Task Analysis

### Geometric Alignment Score (GAS) Comparison

| Model | GSM8K GAS | HumanEval GAS | Pattern |
|-------|-----------|---------------|---------|
| rl_zero | 0.660 | 0.797 | Highest on both |
| sft | 0.070 | 0.674 | Low → High |
| think | 0.183 | 0.620 | Medium → High |

**Finding**: RL-Zero consistently preserves base error direction across tasks, while SFT diverges more on GSM8K than HumanEval.

### CKA Similarity Comparison

| Model | GSM8K CKA | HumanEval CKA |
|-------|-----------|---------------|
| rl_zero | 0.995 | 1.000 |
| sft | 0.815 | 0.992 |
| think | 0.818 | 0.992 |

**Finding**: RL-Zero makes almost no representation changes (CKA ≈ 1.0). SFT/Think make larger changes on GSM8K (CKA ≈ 0.81) than HumanEval (CKA ≈ 0.99).

### Intrinsic Dimensionality Comparison

| Task | Base ID | Interpretation |
|------|---------|----------------|
| GSM8K | 68.16 | Higher complexity, more distributed |
| HumanEval | 18.94 | Lower complexity, more compressed |

**Finding**: Code (HumanEval) has ~3.6x lower intrinsic dimensionality than math (GSM8K), suggesting code representations are more structured/compressed.

---

## Key Findings

### 1. RL-Zero Preserves Base Geometry
- **GAS > 0.6** for RL-Zero on both tasks (vs < 0.2 for SFT on GSM8K)
- **CKA ≈ 1.0** for RL-Zero (vs 0.81 for SFT on GSM8K)
- This aligns with Jack Morris's observation that RLVR adds "information" without disrupting structure

### 2. High GAS ≠ High Performance
- RL-Zero has highest GAS but zero improvement on GSM8K
- SFT has lowest GAS but +50% improvement on GSM8K
- **Implication**: Geometric alignment predicts training *path*, not training *success*

### 3. Training Efficiency Varies by Task
- **GSM8K**: SFT achieves 2.70 TER (50% acc gain / 18.5% geometry change)
- **HumanEval**: All models have high CKA (minimal change), so TER is unstable

### 4. The "RLVR Readiness" Paradox
- High RRS (RL-Zero = 0.101) doesn't predict accuracy improvement
- Low RRS (SFT = 0.011) achieves better performance
- **Conclusion**: RRS measures *geometric compatibility*, not *potential for improvement*

---

## Interpretation

### What RLVR Potential Index Actually Measures

The index captures **geometric compatibility** between base model and training method:
- **High LCS**: Base model has latent signal separating correct/incorrect
- **High GAS**: Training preserves the base model's error direction
- **Low ID**: Compact representation (potentially easier to fine-tune)

### Why RL-Zero Has High GAS But Low Improvement (GSM8K)

1. **RL-Zero is trained on outcome reward only** → It optimizes the path to correct answers without explicitly learning CoT
2. **The base model already has the "right" geometry** (LCS = 0.647) but lacks the capability to execute
3. **RL-Zero preserves geometry** but doesn't add new capability
4. **SFT explicitly teaches CoT patterns** from stronger model, changing geometry but adding capability

### Revised Interpretation of Metrics

| If High... | Means... | Predicts... |
|------------|----------|-------------|
| LCS | Base has latent knowledge | Potential for *any* training |
| GAS | Training preserves base direction | Training is "geometry-compatible" |
| CKA | Representation barely changes | Minimal structural disruption |
| TER | High acc per geometry change | Training is "efficient" |

---

## Practical Implications

### For Predicting RLVR Success

1. **LCS is necessary but not sufficient**: High probe AUC means signal exists, but doesn't guarantee RLVR will extract it
2. **GAS predicts training style, not outcome**: High GAS = RL-style (preserves geometry), Low GAS = SFT-style (transforms geometry)
3. **Consider the task**: Code has lower ID, may respond differently to training than math

### Limitations

1. **Small sample sizes** (100 samples) may not capture full distribution
2. **HumanEval has very few correct base samples** (2%), limiting probe accuracy
3. **RRS formula may need revision**: Current formula rewards geometry preservation, but SFT shows that geometry transformation can be beneficial

---

## Future Work

1. **Extend to other models**: Test on Qwen, Llama, etc. to validate patterns
2. **Add capability gap metric**: Measure the "gap" between latent knowledge and actual performance
3. **Revise RRS formula**: Perhaps `RRS = LCS × (1 - GAS) × efficiency_factor` to reward beneficial transformations
4. **Test on more tasks**: Add MBPP, MATH, logic tasks with balanced correct/incorrect

---

## Raw Data (JSON)

```json
{
  "gsm8k": {
    "base": {"accuracy": 0.14, "lcs": 0.647, "id": 68.16},
    "rl_zero": {"accuracy": 0.14, "gas": 0.660, "cka": 0.995, "ter": 0.0, "rrs": 0.101},
    "sft": {"accuracy": 0.64, "gas": 0.070, "cka": 0.815, "ter": 2.70, "rrs": 0.011},
    "think": {"accuracy": 0.39, "gas": 0.183, "cka": 0.818, "ter": 1.37, "rrs": 0.028}
  },
  "humaneval": {
    "base": {"accuracy": 0.02, "lcs": "NaN", "id": 18.94},
    "rl_zero": {"accuracy": 0.13, "gas": 0.797, "cka": 1.000},
    "sft": {"accuracy": 0.06, "gas": 0.674, "cka": 0.992},
    "think": {"accuracy": 0.06, "gas": 0.620, "cka": 0.992}
  }
}
```
