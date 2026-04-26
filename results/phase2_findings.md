# Phase 2 Findings: Geometric Signatures of Reasoning

## Summary

**H1 CONFIRMED**: We can distinguish correct vs incorrect reasoning trajectories using simple geometric features.

**Key Result**: RLVR training preserves 99.9% of base model geometry while achieving the best classification accuracy.

## H1: Within-Domain Classification (8-shot GSM8K)

| Model | Task Accuracy | H1 Classification (RF) | Status |
|-------|---------------|------------------------|--------|
| olmo3_base | 77.5% | 73.0% | ✓ PASS |
| olmo3_rl_zero | 78.5% | **75.0%** | ✓ PASS |
| olmo3_sft | 68.0% | 68.5% | ✓ PASS |
| olmo3_think | 45.5% | 61.0% (LR) | ✓ PASS |

**Success threshold**: >60% accuracy (all models pass)

## Geometric Analysis (8-shot GSM8K)

| Model | Subspace Preservation | Effective Rank |
|-------|----------------------|----------------|
| olmo3_base | 1.000 (reference) | 4.5 |
| olmo3_rl_zero | **0.999** | 4.5 |
| olmo3_sft | 0.840 | 3.9 |
| olmo3_think | 0.839 | 3.9 |

**Statistical Significance**: RLVR vs SFT t=8.99, p<0.0001

## Most Predictive Features

1. **cos_sim** (layer-to-layer cosine similarity) - especially L4, L5, L8, L14
2. **entropy** (activation entropy per layer) - especially early layers L1, L2
3. **max_activation** (peak activation per layer)

## H2: Cross-Domain Transfer (0-shot, olmo3_base only)

| Train → Test | Accuracy |
|--------------|----------|
| GSM8K → LogiQA | **77.5%** ✓ |
| LogiQA → GSM8K | 12.5% (LR) / 87.5%* (RF) |

*RF may be inflated due to class imbalance

## Data Quality Issues

- **HumanEval 0-shot**: All labels = 0 (code execution failed)
- **LogiQA 0-shot**: 3/4 files corrupted (only olmo3_base works)
- **8-shot**: Only GSM8K collected

## Next Steps

1. Collect 8-shot LogiQA and HumanEval trajectories
2. Run H2 with all 4 models on 8-shot data
3. Test H3 (non-verifiable domains) if H2 passes

## Key Insight

RLVR training creates models that:
- Preserve base model geometry almost perfectly (99.9%)
- Achieve highest task accuracy (78.5%)
- Have most distinguishable correct/incorrect trajectories (75% H1)

This suggests RLVR learns to use the existing representational structure more effectively rather than fundamentally altering it.
