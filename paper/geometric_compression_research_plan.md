# Geometric Signatures of Compression Strategy in Post-Trained LLMs

A Research Plan for Characterizing How Training Paradigms Induce Different Representational Structures


## 1. Executive Summary

This research investigates whether different post-training paradigms (RLVR vs SFT) induce detectably different compression strategies in neural network representations, and whether these strategies correlate with cross-domain transfer performance.

Core claim: RLVR and SFT produce different organizational principles for how models compress problem-solution mappings. These principles have geometric signatures measurable in activation space. The signatures correlate with transfer because they reflect encoding structure rather than task-specific content.

What we measure: The geometry of the representation manifold and the distribution of activation trajectories, not "reasoning" per se.

What we predict: RLVR models show lower effective dimensionality, higher cross-domain geometric consistency, and lower trajectory variance across samples compared to SFT models. These properties correlate with OOD transfer.

What we explicitly do not claim: That we are measuring reasoning, that geometry causes transfer, or that we have operationalized fluid intelligence.


## 2. Motivation and Background

The field has documented that RLVR-trained models transfer better than SFT models to novel domains. Jin et al. (2025) showed RLVR preserves singular vector directions while SFT causes representational drift. Huan et al. (2025) introduced a Transferability Index showing RLVR models exhibit smaller PCA shifts across layers. Chu et al. (2025) demonstrated that RL learns generalizable rules while SFT memorizes training data.

However, the literature has significant gaps:

Gap 1: No pre-transfer geometric predictors. All current methods require target task data or training on target tasks. No method extracts features from a source model alone that predict transfer to unseen tasks.

Gap 2: No separation of paradigm effects from data effects. We do not know if RLVR's geometric signatures come from the training paradigm or from the typically more diverse training data RLVR models see.

Gap 3: No mechanistic explanation. Why does RLVR preserve directions better? Jin et al. hypothesize RL's optimization landscape inherently counteracts drift, but this is speculative.

Gap 4: No application of dynamical systems tools. Path signatures and stability analysis have not been applied to characterize transferability in LLMs.

Gap 5: The "reasoning" framing may be misleading. Recent work on CoT faithfulness shows models often determine answers before generating reasoning chains, suggesting trajectory geometry may capture elaboration rather than decision-making.

This research addresses gaps 1, 3, and 4 while being explicit about the limitations imposed by gaps 2 and 5.


## 3. The Compression Strategy Hypothesis

We propose that what transfers across domains is not task-specific knowledge but organizational structure. Different training paradigms induce different compression strategies for mapping inputs to outputs.

RLVR's sparse outcome signal creates pressure to learn structural representations (relationships, transformations, composition rules) because these generalize across the reward landscape. SFT's dense token-level supervision allows memorization of surface patterns because immediate correction makes robust structure unnecessary.

A compression strategy manifests geometrically as:
- How activations are distributed in the representation space (dimensionality)
- How consistent this distribution is across different domains (invariance)  
- How much the distribution varies across different samples of the same input (stability)

The hypothesis: RLVR induces a compression strategy characterized by low dimensionality (rule-like compression), high cross-domain consistency (same organizational structure regardless of surface domain), and low sample variance (single dominant mode rather than many modes).


## 4. Related Work and Anticipated Objections

This section addresses known problems with the research approach and how we handle them.


### 4.1 The Decision-Before-Reasoning Problem

Afzal et al. (2025) showed that probing classifiers can predict CoT success even before a single token is generated. The decision is encoded in early hidden states; reasoning chains are downstream elaboration.

Implications: We are not measuring how models make decisions. We are measuring how models elaborate and justify decisions. This is still meaningful for transfer if elaboration structure reflects underlying compression strategy.

Our response: We explicitly analyze geometry at different layer depths. If early-layer geometry predicts transfer better than late-layer geometry, we have evidence of decision-relevant structure. If late-layer geometry is equally or more predictive, we are characterizing elaboration quality, which may still be a valid transfer correlate.


### 4.2 The Post-Hoc Rationalization Problem

Turpin et al. (2023) demonstrated that CoT explanations can be systematically unfaithful. Models generate plausible reasoning for biased answers without mentioning the bias. The "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful" paper (2025) showed 7-13% of outputs in production models exhibit post-hoc rationalization.

Implications: The trajectory geometry we measure is the geometry of possible justifications, not a single reasoning process. There is no true trajectory; there is a distribution over trajectories conditioned on an already-determined answer.

Our response: We measure trajectory variance explicitly. The distribution of trajectories IS the compression strategy. High variance indicates memorized lookup (many equally-weighted paths). Low variance indicates rule-based compression (single dominant mode). We test whether variance correlates with problem ambiguity (calibration) rather than being random.


### 4.3 The Paradigm vs Data Confound

RLVR training typically uses more diverse data than SFT due to exploration. We cannot distinguish "RLVR paradigm causes different geometry" from "diverse training data causes different geometry" without controlled experiments.

Our response: We acknowledge this confound explicitly. We frame results as "RLVR models show X" rather than "the RLVR paradigm causes X." We use OLMo models which provide the closest available controlled comparison. If we find that geometric signatures correlate with transfer even controlling for training data diversity (by comparing within model families), this strengthens paradigm-specific claims.


### 4.4 The Elicitation vs Learning Problem

Lambert et al. and others argue RLVR is fundamentally about probability concentration, not capability expansion. Pass@k experiments show base models match RL-trained models at high k values. RLVR concentrates probability mass on existing capabilities rather than creating new ones.

Evidence: With Qwen 2.5 models, random rewards improve MATH scores by 15-20 points. This is probability concentration via GRPO's clipping term, not learning.

Implications: If RLVR is elicitation, there may be no special "reasoning geometry." The geometry we find might just correlate with how good the base model is.

Our response: This actually makes the research MORE tractable. Instead of characterizing training dynamics, we characterize what capabilities exist to be elicited. The question becomes: can geometry predict the capability ceiling (pass@k at high k) rather than predicting transfer from training?

We add explicit tests: After RL training, geometry should change minimally if RL only concentrates probability. This is testable with the DeepSeek V3-Base vs V3 vs R1-Zero vs R1 comparison.


### 4.5 The CKA Reliability Problem

Davari et al. (2022) showed CKA can be manipulated and does not reliably predict transfer across all tasks. No single similarity metric works universally.

Our response: We use multiple geometric measures (not just CKA) and test which measures predict transfer in our specific comparison. We do not claim any single metric is universal.


### 4.6 The Negative Results in Faithfulness Transfer

Ley et al. (2024) showed that interventions to improve CoT faithfulness (in-context learning, fine-tuning, activation editing) fail to generalize across diverse benchmarks.

Implications: If faithfulness does not transfer, why would our geometric signatures of "reasoning quality" transfer?

Our response: We are not measuring faithfulness. We are measuring compression strategy, which is conceptually distinct. Faithfulness is about whether the model reports its actual process. Compression strategy is about how the model organizes representations. A model can have a good compression strategy while being unfaithful about its process.


### 4.7 The Format/Noise Artifact Problem

Geometric signatures could reflect surface features (token length, output format, domain style) rather than meaningful structure.

Our response: We control for surface features explicitly. We compare geometry on problems with the same format but different difficulty. We test whether geometry predicts within-format transfer. If signatures only correlate with surface properties, this is a negative result we report.


## 5. Experimental Design


### 5.1 Model Selection

Primary comparison (controlled):
- OLMo 3-Base 7B: Pretrained only
- OLMo 3-Instruct 7B: SFT on instructions
- OLMo 3-RL-Zero 7B: RL directly from base (no SFT intermediate)

This is the cleanest available comparison. Same base, same scale, different post-training, fully open checkpoints and training logs.

Secondary comparison (larger scale, less controlled):
- DeepSeek V3-Base: Pretrained 671B MoE
- DeepSeek V3: V3-Base + RLHF
- DeepSeek R1-Zero: V3-Base + pure RL
- DeepSeek R1: V3-Base + SFT + RL

For compute efficiency, use distilled versions (8B, 14B) with acknowledgment that distillation may introduce artifacts.

Tertiary comparison (validation):
- Llama-3-8B-Base vs Llama-3-8B-Instruct
- Qwen-2.5-7B-Base vs Qwen-2.5-7B-Instruct

Include Qwen specifically to test whether random-reward artifacts appear in geometric signatures.


### 5.2 Task Selection

Activation collection domains:
- Math: GSM8K, MATH
- Code: HumanEval, MBPP  
- Logic: LogiQA, FOLIO
- Commonsense: HellaSwag, PIQA

Transfer evaluation:
- In-domain: Performance on held-out samples from training domains
- Near transfer: Different tasks in same domain (GSM8K train -> MATH test)
- Far transfer: Different domains entirely (math -> code)

Post-training benchmarks (to avoid contamination):
- LiveMathBench 202505
- AIME 2025
- Recently released benchmarks after model training cutoff


### 5.3 Activation Collection Protocol

For each model and task, collect 500 samples with chain-of-thought prompts.

Extract:
- Residual stream activations at every layer (before and after MLP)
- Both per-token and sequence-aggregated (mean-pooled) activations
- Correctness labels for correlation analysis

Storage format: HDF5 with metadata for reproducibility.

For trajectory analysis, store full per-token activations for a subset (100 samples) to manage storage costs.


### 5.4 Geometric Measures

Static measures (computed on activation matrices):

Effective rank: Captures intrinsic dimensionality of activations.
```
effective_rank = exp(entropy(normalized_singular_values))
```

Spectral decay: How quickly singular values fall off.
```
spectral_decay = fit_power_law(singular_values)
```

Subspace preservation: Principal angles between subspaces of different models.
```
angles = principal_angles(V_base[:k], V_finetuned[:k])
preservation_score = mean(cos(angles))
```

Local curvature: Estimated from perturbation analysis.
```
for sample in samples:
    base_act = model(sample)
    perturbed_acts = [model(perturb(sample, eps)) for _ in n_perturbations]
    local_curvature = variance(perturbed_acts - base_act) / eps^2
```

Dynamic measures (computed on trajectories):

Path signature: Computed using signatory library.
```
trajectory = stack([activations[layer] for layer in layers])  # (n_layers, d_model)
sig = signature(trajectory.unsqueeze(0), depth=3)
```

Signature is reparameterization-invariant and captures path geometry including curvature and self-intersections.

Local Jacobian spectral radius: Approximates sensitivity/stability without requiring true dynamical system.
```
# For each layer transition
jacobian = estimate_jacobian_finite_diff(model, layer, activation_point, eps=1e-4)
local_sensitivity = max(svdvals(jacobian))
```

This is NOT a Lyapunov exponent. It measures local sensitivity to perturbations at a single point. We aggregate across many points to characterize the manifold.

Sample variance: For a given input, how much does trajectory geometry vary across samples?
```
trajectories = [get_trajectory(model, input, temperature=1.0) for _ in n_samples]
signature_variance = variance([signature(t) for t in trajectories])
```


### 5.5 Transfer Measurement

Transferability Index following Huan et al.:
```
TI = (accuracy_target - accuracy_baseline) / (accuracy_source - accuracy_baseline)
```

We compute TI for each source-target domain pair and correlate with geometric measures.


## 6. Specific Hypotheses and Tests


### H1: Subspace Preservation (Primary)

Claim: Transferability correlates with preservation of specific singular vector directions from base model. RLVR preserves "transfer-critical" subspaces that SFT disrupts.

Test:
```
base_V = svd(base_activations)[2]
rlvr_V = svd(rlvr_activations)[2]
sft_V = svd(sft_activations)[2]

rlvr_preservation = mean(cos(principal_angles(base_V[:k], rlvr_V[:k])))
sft_preservation = mean(cos(principal_angles(base_V[:k], sft_V[:k])))
```

Prediction: rlvr_preservation > sft_preservation, and preservation score correlates with transfer success.

Falsification: If preservation scores are similar, or do not correlate with transfer, hypothesis fails.


### H2: Representational Flatness

Claim: Transfer correlates with local flatness of the representation manifold.

Test:
```
flatness = 1 / mean(local_curvature across samples)
```

Prediction: RLVR has higher flatness (lower curvature). Flatness correlates with OOD accuracy independent of in-domain accuracy.

Falsification: If curvature does not differ, or does not correlate with transfer controlling for capability, hypothesis fails.


### H3: Trajectory Consistency

Claim: RLVR produces trajectories that are more consistent across domains and across samples.

Test:
```
cross_domain_consistency = mean([cosine_sim(sig_domain_i, sig_domain_j) for all pairs])
sample_variance = mean([variance(signatures for same input) for all inputs])
```

Prediction: RLVR has higher cross_domain_consistency and lower sample_variance than SFT.

Falsification: If consistency/variance measures do not differ between paradigms, hypothesis fails.


### H4: Layer Localization

Claim: Transfer-predictive geometry is localized in specific layers.

Test: Compute all geometric measures per layer. Run regression predicting transfer from per-layer measures.

Prediction: Early/middle layers predict transfer better than late layers (consistent with decision-before-reasoning finding).

Falsification: If late layers are equally or more predictive, we are measuring elaboration rather than decision structure.


### H5: Calibrated Uncertainty

Claim: RLVR produces models where trajectory variance correlates with actual problem ambiguity.

Test:
```
for problem in problems_with_known_ambiguity:
    samples = [model.generate(problem) for _ in 100]
    trajectory_variance = compute_signature_variance(samples)
    # Correlate with ground-truth ambiguity rating
```

Prediction: RLVR shows correlation between variance and ambiguity. SFT shows no correlation (arbitrary confidence).

Falsification: If neither model shows calibration, or both show equal calibration, hypothesis fails.


### H6: Geometry Predicts Capability Ceiling

Claim: Base model geometry predicts pass@k ceiling before any post-training.

Test:
```
for base_model in [olmo_base, llama_base, qwen_base]:
    geometry = compute_geometric_measures(base_model)
    ceiling = measure_pass_at_256(base_model, reasoning_tasks)
    # Regression: can geometry predict ceiling?
```

Prediction: Geometric measures correlate with capability ceiling.

Falsification: If geometry does not predict ceiling, the geometric approach has limited value for characterizing models.


## 7. Timeline and Phases


### Phase 1: Geometric Characterization (Weeks 1-6)

Week 1-2: Setup
- Download models (OLMo family primary, DeepSeek distill secondary)
- Implement activation collection pipeline with TransformerLens
- Verify pipeline on small test set

Week 3-4: Data collection
- Collect activations for all models on all tasks
- Store full trajectories for subset
- Initial quality checks

Week 5-6: Static analysis
- Compute SVD, effective rank, spectral decay for all models/layers
- Compute subspace preservation scores
- Compute local curvature estimates
- Statistical tests for H1, H2

Deliverables:
- Activation dataset (shareable)
- Geometric characterization report
- Decision: Is there signal to continue?


### Phase 2: Trajectory Analysis (Weeks 7-10)

Week 7-8: Signature computation
- Implement path signature pipeline using signatory
- Compute signatures for trajectory subset
- Dimension reduction if needed (random projection)

Week 9-10: Stability analysis
- Implement Jacobian estimation
- Compute local sensitivity measures
- Aggregate into manifold-level statistics

Statistical tests for H3, H4.

Deliverables:
- Trajectory geometry dataset
- Comparison of signature distributions across models
- Layer-wise analysis of where signal lives


### Phase 3: Transfer Correlation (Weeks 11-14)

Week 11-12: Transfer measurement
- Evaluate all models on transfer tasks
- Compute Transferability Index for all domain pairs
- Use published benchmarks where available

Week 13-14: Correlation analysis
- Regress TI on geometric measures
- Determine which measures predict transfer
- Control for capability (in-domain performance)
- Test H5 (calibrated uncertainty) if time permits

Deliverables:
- Transfer matrix with TI scores
- Regression models predicting transfer from geometry
- Feature importance ranking


### Phase 4: Validation and Writeup (Weeks 15-18)

Week 15-16: Validation
- Test geometric predictors on held-out models
- Evaluate generalization to different model families
- Test H6 (capability ceiling prediction)

Week 17-18: Paper writing
- Document methodology
- Present results with appropriate caveats
- Discuss limitations explicitly
- Submit preprint

Deliverables:
- Paper draft
- Code release
- Reproducibility package


## 8. Compute and Cost Estimates

| Component | GPU Hours (A100) | Cost @ $2/hr |
| Activation collection | 100-150 | $300 |
| SVD/static analysis | 30-50 | $100 |
| Path signatures | 100-200 | $400 |
| Jacobian estimation | 100-150 | $300 |
| Transfer evaluation | 50-100 | $200 |
| Buffer for debugging | 100 | $200 |
| Total Phase 1-3 | 500-750 | $1500 |

Optional causal interventions (if Phase 3 shows signal):
| Component | GPU Hours | Cost |
| Fine-tuning runs | 500-1000 | $2000 |
| Intervention experiments | 300-500 | $1000 |

Total with interventions: ~$4500


## 9. Success Criteria


### Minimum Viable Result (Publishable)

Systematic geometric characterization showing RLVR and SFT produce measurably different activation geometry. Even without transfer prediction, this is a contribution to understanding post-training effects.


### Target Result (Strong Contribution)

Above, plus: At least one geometric measure correlates with transfer success (r > 0.3) controlling for in-domain capability. Layer-wise analysis reveals where transfer-relevant structure lives.


### Stretch Goal (Major Contribution)

Above, plus: Geometric measure predicts transfer to held-out domain pairs with useful accuracy. Base model geometry predicts capability ceiling. Provides actionable guidance for practitioners.


## 10. Failure Modes and Pivots


### If no geometric differences between RLVR and SFT

Pivot: Focus on describing what geometry DOES differ between base and fine-tuned models regardless of paradigm. This is still novel.


### If geometric differences exist but do not predict transfer

Pivot: Report as negative result. Investigate whether geometry predicts other properties (in-domain generalization, calibration, robustness).


### If geometry just correlates with capability

Report this finding. It means geometry is a proxy for model quality but not specific to transfer. Still useful for model selection even if mechanistically uninteresting.


### If results are model-specific and do not generalize

Report with appropriate scope. Characterization of OLMo family is still useful. Flag that results may not generalize.


## 11. Ethical Considerations

This research characterizes existing models and does not create new capabilities. The primary risks are:

Dual use: Understanding what makes models transfer could inform training better models. We assess this as low risk since the findings would be incremental over existing knowledge.

Overpromising: We explicitly scope our claims to avoid suggesting we have "solved" transfer or reasoning. All claims are correlational unless causal interventions succeed.

Reproducibility: We commit to releasing code, activation datasets (where license permits), and detailed methodology.

## 12. Critical Analysis: The Missing Mechanism Problem

**Written: 2026-01-11**

### The Core Problem with the Current Approach

The Phase 1-3 plan as written is fundamentally **correlational and descriptive**. We measure geometry, we measure transfer, we compute correlations. Even if we find r=0.5, we cannot:

1. **Predict transfer without running the model** - We still need target task performance to validate
2. **Claim causation** - Geometry might be a proxy for capability, not a cause of transfer
3. **Provide a tool** - Practitioners gain no actionable guidance
4. **Explain mechanism** - Why would subspace preservation → transfer?

### Brutal Honesty: What Phase 3 Will Actually Show

**Best case outcome of current plan:**
- "Subspace preservation correlates with transfer (r=0.4, p<0.01)"
- "Models with higher preservation show 15% better OOD accuracy"

**What this does NOT tell us:**
- Whether preservation *causes* transfer or is confounded with capability
- Whether we can predict transfer to truly novel domains
- How to *improve* transfer by modifying geometry
- Whether intervening on geometry changes behavior

### Why This Matters

The field already knows RLVR transfers better than SFT. Adding "and their geometry differs" is incremental. The valuable contribution would be:

1. A **predictor** that doesn't require target data
2. A **mechanism** explaining why geometry → transfer
3. An **intervention** that modifies transfer by modifying geometry

---

## 13. Alternative Hypotheses: Toward a Mechanism

### Hypothesis A: Optimal Transport as a Metric for "Closeness to Base"

**Core Idea:** Compute Wasserstein distance W₂(base_activations, finetuned_activations). RLVR has lower W₂ (stays closer to base distribution), which predicts transfer because base model representations are more general.

**Formalization:**
```
W₂(P_base, P_ft) = inf_{γ∈Γ(P_base,P_ft)} E_{(x,y)∼γ}[||x-y||²]^{1/2}
```

**Testable Prediction:** W₂(base, RLVR) < W₂(base, SFT), and lower W₂ correlates with better transfer.

**Critique:**
- (+) Principled distance metric with theoretical foundations
- (+) Can be computed efficiently with Sinkhorn algorithm
- (+) Task-agnostic: can compare distributions without task labels
- (-) Still correlational: W₂ might just correlate with less training, not better transfer
- (-) No causal mechanism: why would smaller OT distance → better transfer?
- (-) OT is sensitive to sample size and dimensionality artifacts

**Verdict:** Worth computing, but still descriptive. Adds one more feature to the correlation analysis.

---

### Hypothesis B: Rectified Flow to Reveal Invariants

**Core Idea:** Train a rectified flow model that transforms base activations → fine-tuned activations. The "complexity" of this flow reveals how much the training changed the representation structure.

**Formalization:**
```
dX_t = v(X_t, t)dt,  X_0 ~ P_base,  X_1 ~ P_ft
```

Train velocity field v to straighten the flow (rectified flow). Measure:
- Flow curvature: ||∂²X_t/∂t²||
- Jacobian trace: tr(∂v/∂X)
- Straightness: E[||X_1 - X_0 - v(X_0,0)||²]

**Testable Prediction:**
- RLVR has "straighter" flows from base (simpler transformation)
- Flow complexity predicts transfer (more complex = worse transfer)

**Critique:**
- (+) Theoretically elegant: flow captures the transformation geometry
- (+) Can visualize the transformation path
- (+) Straightness is a natural measure of "how much changed"
- (-) Training rectified flows is computationally expensive
- (-) High-dimensional (d=4096) flows are hard to train well
- (-) Still descriptive: we learn what the flow looks like, not what to do with it
- (-) No obvious intervention: can't use the flow to improve models

**Verdict:** Interesting for understanding, but adds complexity without adding actionability. A paper could be written about flow properties, but it wouldn't help practitioners.

---

### Hypothesis C: Geometric Predictor (Train on Manifold Shape)

**Core Idea:** Extract geometric features from source model activations only. Train a predictor to estimate transfer performance. Test on held-out models.

**Feature Set:**
```python
features = {
    'effective_rank': ...,
    'spectral_decay': ...,
    'preservation_vs_base': ...,
    'signature_variance': ...,
    'cross_domain_consistency': ...,
    'local_curvature': ...,
    'W2_from_base': ...
}
```

**Predictor:** Ridge regression or random forest: features → transfer_accuracy

**Testable Prediction:** Can predict transfer to novel domains (not seen during predictor training) with useful accuracy (r > 0.5).

**Critique:**
- (+) Produces a usable tool: input = model, output = predicted transfer
- (+) Can be evaluated with proper train/test splits
- (+) Low compute cost once features are extracted
- (-) Requires labeled transfer data to train the predictor (chicken-and-egg)
- (-) May not generalize to new model families
- (-) Still correlational: the predictor learns correlations, not causes
- (-) Limited by number of models we can evaluate (small N problem)

**Verdict:** Most practical of the options. Even if correlational, a working predictor is useful. The small N problem is real—we only have ~4-8 models, so cross-validation is limited.

---

### Hypothesis D: Activation Steering for Causal Test (STRONGEST)

**Core Idea:** If RLVR geometry is "better" for transfer, can we modify SFT activations to be more RLVR-like and observe improved transfer? This is a **causal intervention**.

**Method 1: Linear Subspace Projection**
```python
# Project SFT activations onto RLVR's top-k subspace
V_rlvr = svd(rlvr_activations)[2][:k]  # Top-k right singular vectors
x_steered = V_rlvr.T @ V_rlvr @ x_sft  # Project SFT activation onto RLVR subspace
```

**Method 2: Optimal Transport Map**
```python
# Learn OT map from SFT → RLVR distribution
T = learn_ot_map(P_sft, P_rlvr)  # e.g., using neural OT
x_steered = T(x_sft)  # Apply map at inference
```

**Method 3: Activation Addition (Following Anthropic's Work)**
```python
# Find "transfer direction" in activation space
transfer_direction = mean(rlvr_activations) - mean(sft_activations)
x_steered = x_sft + α * transfer_direction
```

**Testable Prediction:** Steering SFT activations toward RLVR-like geometry improves transfer performance on OOD tasks.

**Critique:**
- (+) CAUSAL: We intervene and observe effect
- (+) If it works: strong evidence geometry matters
- (+) Produces a usable technique: activation steering for transfer
- (-) If it fails: either geometry doesn't cause transfer, or our steering is wrong
- (-) Steering might break the model (activations are optimized together)
- (-) Requires careful implementation (when to steer, which layers, how much)
- (-) May only work for specific task pairs

**Verdict:** This is the experiment that would make the paper significant. Even negative results are informative: "Steering geometry doesn't improve transfer, suggesting geometry is correlate not cause."

---

## 14. Honest Assessment: What's Worth Doing?

### Tier 1: Must Do (Current Plan + Extensions)
1. **Phase 2 Data Collection** - Trajectories at even layers, 500 samples × 3 tasks × 4 models
2. **Static Geometry Measures** - Already have these from Phase 1
3. **Path Signatures** - Novel contribution, worth computing
4. **Transfer Correlation** - Required to connect geometry to behavior

### Tier 2: Should Do (Adds Value)
5. **Optimal Transport Distance** - Adds one principled metric, low effort
6. **Geometric Predictor** - Most practical output, even if limited by small N
7. **Layer-wise Analysis** - Where does transfer-predictive signal live?

### Tier 3: High Risk, High Reward
8. **Activation Steering Experiment** - The causal test
   - If positive: major contribution
   - If negative: still publishable as negative result
   - Risk: may fail for technical reasons unrelated to hypothesis

### Tier 4: Probably Not Worth It (For This Paper)
9. **Rectified Flow Training** - High compute, unclear payoff
10. **Jacobian Estimation** - O(d²) compute, may not add beyond signatures
11. **Sample Variance Analysis** - Requires generation, expensive

---

## 15. Revised Research Plan

### Phase 2 (Weeks 1-4): Data Collection + Measures

**Week 1-2: Trajectory Collection**
- Even layers only: [0, 2, 4, ..., 30] = 16 layers
- 500 samples × 3 tasks (GSM8K, HumanEval, LogiQA)
- 4 models (Base, SFT, RL-Zero, Think)
- ~56 GB storage (per PHASE2_PLAN.md)

**Week 3-4: Compute Measures**
- Path signatures (depth 3, proj_dim=64)
- Optimal Transport distances (Sinkhorn, ε=0.1)
- Cross-domain signature consistency
- Aggregate with Phase 1 static measures

### Phase 3A (Weeks 5-8): Transfer Correlation

**Week 5-6: Transfer Evaluation**
- Evaluate all 4 models on: GSM8K, MATH, HumanEval, MBPP, LogiQA, FOLIO
- Compute transfer matrix

**Week 7-8: Correlation Analysis**
- Merge geometry features with transfer outcomes
- Regression: which features predict transfer?
- Partial correlations controlling for in-domain performance

### Phase 3B (Weeks 9-12): Causal Experiment (The Critical Test)

**Week 9-10: Implement Activation Steering**
- Method: Subspace projection (SFT → RLVR subspace)
- Layers: Middle layers (8-24) where Phase 1 showed largest preservation differences
- Steering strength: Grid search α ∈ [0.1, 0.3, 0.5, 0.7, 1.0]

**Week 11-12: Evaluate Steering Effect**
- Compare: SFT (no steering) vs SFT (steered) on transfer tasks
- Hypothesis: Steering toward RLVR geometry improves transfer
- Even if effect is small, direction matters

### Deliverables

1. **Geometric Feature Set** - Standardized features extractable from any model
2. **Transfer Predictor** - Trained on OLMo, test on held-out models
3. **Steering Analysis** - Does modifying geometry change transfer? (Causal evidence)
4. **Paper** - With honest limitations section

---

## 16. What If We're Wrong?

### Failure Mode 1: Geometry Doesn't Predict Transfer
- **What it means:** Geometry is incidental, not causal
- **What to report:** Negative result, but with the geometric characterization itself
- **Pivot:** What DOES predict transfer? Maybe per-sample correctness patterns?

### Failure Mode 2: Steering Doesn't Work
- **What it means:** Either (a) geometry isn't causal, or (b) our intervention is too crude
- **What to report:** "Naive subspace projection insufficient; geometry may require finer intervention"
- **Follow-up:** Try more sophisticated steering (OT map, learned transformation)

### Failure Mode 3: Results Don't Generalize Beyond OLMo
- **What it means:** OLMo-specific findings, not universal principles
- **What to report:** Characterization of OLMo family with caveat
- **Mitigation:** Test on DeepSeek distilled as secondary validation

### Failure Mode 4: Everything Correlates with Capability
- **What it means:** Geometry just reflects how good the model is
- **What to report:** "Geometric measures are proxies for capability, not specific transfer indicators"
- **This is still useful:** Geometry as a cheap capability proxy

---

## 17. The Honest Bottom Line

**What we can probably show:**
- RLVR and SFT have different geometry (Phase 1: done)
- Geometry correlates with transfer (Phase 3A: likely, given prior work)

**What we might show:**
- A geometric predictor with useful accuracy on held-out models
- Which layers contain transfer-predictive signal
- Optimal transport provides a principled metric

**What would be a major contribution:**
- Activation steering improves transfer (causal evidence)
- A practical tool for model selection without running target tasks

**What we almost certainly cannot show:**
- A complete mechanistic explanation
- Universal principles beyond the models we test
- That we have "solved" transfer prediction

This is honest. The research is worth doing, but we should not oversell the expected outcomes.

---

## 18. CRITICAL REVISION: The Missing Core Question

**Written: 2026-01-11**

### The Fundamental Problem with Everything Above

The entire research plan above asks the **wrong question**. It frames this as "domain transfer" - can a model trained on math do well on code? But that's not what we actually care about.

**The real question**: Can we isolate *reasoning itself* as a geometric process, learn its signature from verifiable domains, and use that signature on non-verifiable domains?

### Why "Transfer" is the Wrong Frame

1. **Transfer is about capability**: Does math training help with code? This is just measuring if training on A helps with B.

2. **Reasoning is about process**: We don't care if the model "transfers" - we care if we can **detect and characterize correct reasoning** geometrically.

3. **Verifiable vs non-verifiable is the key distinction**:
   - Math, code: We KNOW if the answer is correct (ground truth exists)
   - Open-ended reasoning, ethics, strategy: We can't easily verify correctness
   - The valuable tool: Learn "what correct reasoning looks like" on verifiable domains, apply that detector to non-verifiable domains

4. **Linear steering is naive**: Reasoning is a *process* - a trajectory through activation space across layers. It's not a static direction we can add/subtract.

### What We're Actually Trying to Do

**Core Claim (Revised)**: Correct reasoning has a characteristic geometric signature in the flow of activations through transformer layers. This signature can be learned from verifiable domains (where we know correct vs incorrect) and applied as a detector/predictor on non-verifiable domains.

**If true, this gives us**:
1. A geometric detector of "bad reasoning" without ground truth
2. A way to steer generations toward "correct reasoning" trajectories
3. Understanding of what distinguishes correct from incorrect reasoning at the representation level

---

## 19. Revised Hypotheses: Reasoning as Flow

### H1: Correct vs Incorrect Reasoning Have Distinguishable Trajectories

**Core Idea**: On verifiable tasks, compare the layer-by-layer trajectory of activations for problems the model solves correctly vs incorrectly. The trajectories should be distinguishable.

**Test**:
```python
# On GSM8K with ground truth labels
correct_trajectories = [get_trajectory(model, problem) for problem in problems_solved_correctly]
incorrect_trajectories = [get_trajectory(model, problem) for problem in problems_solved_incorrectly]

# Compute path signatures
correct_sigs = [path_signature(t) for t in correct_trajectories]
incorrect_sigs = [path_signature(t) for t in incorrect_trajectories]

# Train classifier
classifier = train_classifier(correct_sigs, incorrect_sigs)
accuracy = cross_validate(classifier)
```

**Prediction**: We can classify correct vs incorrect with >70% accuracy (significantly above 50% baseline).

**Falsification**: If correct and incorrect trajectories are indistinguishable, the geometric approach fails at the first hurdle.

**Critique**:
- (+) Directly tests whether geometry captures reasoning quality
- (+) Uses ground truth we have access to
- (-) May just learn surface features (problem difficulty, length, format)
- (-) "Correct" doesn't mean "good reasoning" - model could guess right
- (-) Only works if model gets some problems right and some wrong

---

### H2: The Correct Reasoning Signature is Domain-Invariant

**Core Idea**: Train a classifier to distinguish correct vs incorrect reasoning on math. Test it on code. If it transfers, correct reasoning has a universal geometric signature across domains.

**Test**:
```python
# Train on math
math_correct_sigs = get_signatures(gsm8k_correct)
math_incorrect_sigs = get_signatures(gsm8k_incorrect)
classifier = train(math_correct_sigs, math_incorrect_sigs)

# Test on code (zero-shot transfer)
code_correct_sigs = get_signatures(humaneval_correct)
code_incorrect_sigs = get_signatures(humaneval_incorrect)
transfer_accuracy = classifier.evaluate(code_correct_sigs, code_incorrect_sigs)
```

**Prediction**: Classifier trained on math correct/incorrect achieves >60% accuracy on code correct/incorrect (above chance).

**Falsification**: If the classifier doesn't transfer, "correct reasoning" may be domain-specific, or geometry doesn't capture it.

**Critique**:
- (+) This is the key test - does reasoning have universal geometry?
- (+) Novel contribution if it works
- (-) Math and code reasoning may just be different
- (-) Sample size issues - need enough correct/incorrect in each domain
- (-) Confounded by model capability - hard problems may just look different

---

### H3: We Can Detect Anomalous Reasoning on Non-Verifiable Domains

**Core Idea**: Use the correct-reasoning classifier as an anomaly detector. On non-verifiable domains (where we don't know the right answer), flag samples that look geometrically "incorrect."

**Test**:
```python
# Train detector on verifiable domains
detector = train_correct_reasoning_detector(math_sigs, code_sigs)

# Apply to non-verifiable domain
philosophical_sigs = get_signatures(philosophical_questions)
reasoning_quality_scores = detector.score(philosophical_sigs)

# Validate with human judgments
human_ratings = collect_human_quality_ratings(philosophical_answers)
correlation = spearman(reasoning_quality_scores, human_ratings)
```

**Prediction**: Geometric "reasoning quality" scores correlate with human judgments (r > 0.3).

**Falsification**: If no correlation with human judgments, the detector doesn't generalize.

**Critique**:
- (+) This is the practical payoff - a detector for non-verifiable domains
- (+) Would be genuinely useful
- (-) Human judgments are noisy and expensive
- (-) May just correlate with fluency or confidence, not reasoning quality
- (-) Non-verifiable domains may have fundamentally different dynamics

---

### H4: Reasoning Trajectories Can Be Steered (Interventional)

**Core Idea**: If we know the geometry of correct reasoning, can we constrain/guide the model's trajectory to stay in that manifold?

**Method**:
```python
# Learn the "correct reasoning" manifold from training data
correct_manifold = fit_manifold(correct_trajectories)  # e.g., PCA, UMAP, or learned embedding

# At inference, project intermediate activations back onto manifold
def guided_forward(model, x, layers_to_steer):
    for layer in model.layers:
        x = layer(x)
        if layer.idx in layers_to_steer:
            x = project_onto_manifold(x, correct_manifold)
    return x
```

**Prediction**: Manifold-constrained generation improves accuracy on held-out verifiable problems.

**Falsification**: If steering degrades performance, either:
(a) The manifold doesn't capture reasoning, or
(b) Steering disrupts other necessary computations

**Critique**:
- (+) CAUSAL test - we intervene and observe effect
- (+) If it works, we have a tool to improve reasoning
- (-) Steering might break generation (activations are jointly optimized)
- (-) Which layers to steer? How strongly? Many hyperparameters
- (-) Computationally expensive at inference
- (-) May just be regression to the mean (push toward average = easier problems)

---

### H5: Correct Reasoning Flow is Simpler (Lower Curvature)

**Core Idea**: Correct reasoning follows straighter, lower-curvature paths through activation space. Incorrect reasoning wanders or oscillates.

**Test**:
```python
# Compute path curvature for correct vs incorrect trajectories
correct_curvatures = [compute_curvature(t) for t in correct_trajectories]
incorrect_curvatures = [compute_curvature(t) for t in incorrect_trajectories]

t_stat, p_value = ttest(correct_curvatures, incorrect_curvatures)
```

**Prediction**: Correct trajectories have significantly lower curvature.

**Interpretation if true**: Correct reasoning is "direct" - the model knows where it's going. Incorrect reasoning is "searching" - the model is uncertain.

**Critique**:
- (+) Simple, interpretable measure
- (+) Connects to intuition about "confident" vs "confused" reasoning
- (-) Curvature might just correlate with problem difficulty
- (-) What if correct reasoning on hard problems requires exploration?
- (-) Curvature definition depends on metric choice

---

## 20. Why Flow/Trajectory, Not Static Geometry?

The Phase 1 results showed static geometry differences (subspace preservation). But reasoning is a **process**, not a snapshot.

**Key insight**: The flow of activations through layers IS the computation. Each layer transforms the representation. The trajectory captures:
- How information is processed
- Where the model "decides"
- How it elaborates and refines
- When it goes wrong

**Static geometry tells us**: "This model's activations live in a different subspace"

**Flow geometry tells us**: "This model's reasoning process takes a different path"

For understanding reasoning, flow is what matters. A model could have identical static geometry but completely different reasoning dynamics.

---

## 21. Revised Experimental Design

### Phase 2A: Data Collection (Weeks 1-2)

**Collect trajectories with correctness labels**:
- 500 samples × 3 tasks (GSM8K, HumanEval, LogiQA)
- 4 models (Base, SFT, RL-Zero, Think)
- Even layers: [0, 2, 4, ..., 30]
- **Critically**: Record correctness (model answer matches ground truth)

**Storage**: ~56 GB for trajectories

### Phase 2B: Correct/Incorrect Classification (Weeks 3-4)

**H1 Test**: Can we distinguish correct from incorrect trajectories?

```python
for model in models:
    for task in tasks:
        correct_sigs = get_signatures(trajectories[model][task][correct])
        incorrect_sigs = get_signatures(trajectories[model][task][incorrect])

        # 5-fold cross-validation
        clf = RandomForestClassifier()
        accuracy = cross_val_score(clf, all_sigs, labels, cv=5).mean()

        results[model][task] = accuracy
```

**Success criterion**: Average accuracy > 65% (significantly above 50%)

### Phase 2C: Cross-Domain Transfer (Weeks 5-6)

**H2 Test**: Does the correct/incorrect classifier transfer across domains?

```python
# Train on math
clf = train(math_correct_sigs, math_incorrect_sigs)

# Test on code (no training on code labels)
code_accuracy = clf.evaluate(code_correct_sigs, code_incorrect_sigs)

# Test on logic
logic_accuracy = clf.evaluate(logic_correct_sigs, logic_incorrect_sigs)
```

**Success criterion**: Transfer accuracy > 55% (above chance)

### Phase 3: Intervention (Weeks 7-10)

**H4 Test**: Can we steer trajectories to improve reasoning?

If Phase 2 shows signal (geometry distinguishes correct/incorrect), attempt steering:

```python
# Learn manifold from correct trajectories
correct_manifold = PCA(n_components=k).fit(correct_trajectories)

# Modify model forward pass
def steered_layer(x, layer, manifold, alpha):
    out = layer(x)
    # Project back toward correct manifold
    projected = manifold.inverse_transform(manifold.transform(out))
    return (1 - alpha) * out + alpha * projected

# Evaluate on held-out problems
steered_accuracy = evaluate_with_steering(model, test_problems, manifold)
baseline_accuracy = evaluate(model, test_problems)
```

**Success criterion**: Steering improves accuracy by >2% on held-out problems

### Phase 4: Non-Verifiable Domain Application (Weeks 11-12)

**H3 Test**: Does the detector generalize to non-verifiable domains?

This requires human evaluation:
1. Generate responses on non-verifiable tasks (philosophy, ethics, strategy)
2. Compute geometric "reasoning quality" scores
3. Collect human quality ratings
4. Compute correlation

**Success criterion**: Correlation r > 0.25 with human ratings

---

## 22. Brutally Honest Assessment

### What's Likely to Work
- H1 (correct vs incorrect distinguishable): Probably yes, but might just capture problem difficulty
- Computing trajectories and signatures: Technically straightforward

### What's Uncertain
- H2 (cross-domain transfer): The critical test. If this fails, the whole premise fails.
- H5 (curvature differences): Simple to compute, unclear if meaningful

### What's Risky
- H4 (steering): Many ways to fail for technical reasons
- H3 (non-verifiable detection): Requires human eval, expensive, noisy

### The Core Risk
**The "correct reasoning" geometry might just be the "easy problem" geometry.**

If easy problems have certain trajectory signatures and hard problems have others, we'll learn a difficulty detector, not a reasoning quality detector. This is confounded.

**Mitigation**: Analyze within difficulty strata. Compare correct vs incorrect on problems of similar difficulty.

### Why This is Worth Doing Despite Risks

1. **If H2 succeeds**: We have evidence that "correct reasoning" has domain-invariant geometry. This is a fundamental claim about how reasoning works.

2. **If H4 succeeds**: We have a tool to improve reasoning without retraining. Major practical value.

3. **Even if we fail**: We learn that geometry doesn't capture reasoning in a transferable way. This is important negative result that saves others from going down this path.

---

## 23. The Optimal Transport / Rectified Flow Ideas - Revisited

The user suggested training a rectified flow model to show invariants. Let me be honest about this:

### Rectified Flow for Trajectory Alignment

**Idea**: Learn a flow that maps "incorrect reasoning trajectories" to "correct reasoning trajectories." If this flow is simple (nearly linear), the difference is superficial. If complex, there's fundamental structural difference.

**Problem**: We don't have matched pairs. We can't say "this incorrect trajectory corresponds to this correct trajectory for the same problem" because the model either gets it right or wrong.

**Alternative**: Learn flow between distributions (correct trajectory distribution → incorrect trajectory distribution) using optimal transport.

```python
# Sinkhorn-regularized OT between correct and incorrect trajectory distributions
P_correct = trajectory_distribution(correct_trajectories)
P_incorrect = trajectory_distribution(incorrect_trajectories)

# Wasserstein distance
W2 = compute_wasserstein(P_correct, P_incorrect)

# Learn transport map
T = learn_ot_map(P_incorrect, P_correct)

# At inference: can we "correct" a trajectory?
corrected_trajectory = T(original_trajectory)
```

**Honest assessment**:
- Training high-dimensional flows is hard
- Unclear what to do with the learned map
- May not generalize to non-verifiable domains
- Adds complexity without clear payoff

**Verdict**: Skip for initial phases. Revisit if H1-H2 show strong signal.

---

## 24. Revised Timeline

### Week 1-2: Data Collection
- Even-layer trajectories for all models/tasks
- Record correctness labels

### Week 3-4: H1 Test
- Correct vs incorrect classification within domain
- Report accuracy and feature importance

### Week 5-6: H2 Test (Critical Decision Point)
- Cross-domain transfer of classifier
- If transfer fails: pivot to understanding WHY
- If transfer succeeds: proceed to intervention

### Week 7-10: H4 Test (If H2 Succeeds)
- Implement trajectory steering
- Evaluate on held-out verifiable problems

### Week 11-12: H3 Test (If Resources Allow)
- Apply to non-verifiable domains
- Human evaluation

### Week 13-14: Write-up
- Document results honestly
- Include negative results

---

## 25. The Real Bottom Line (Revised)

**Old question**: Does geometry predict domain transfer?
**New question**: Can we learn the geometry of correct reasoning from verifiable domains and detect it on non-verifiable domains?

**Why this matters**:
- Verifiable domains (math, code) provide ground truth
- Non-verifiable domains (open-ended reasoning) are where we NEED help
- If geometry captures reasoning, we get a tool for the hard cases

**Key test**: H2 - Does the correct/incorrect classifier transfer across domains?

**If H2 fails**: Geometry is domain-specific, not about "reasoning" in general
**If H2 succeeds**: Reasoning has universal geometric signature → major finding

This is a high-risk, high-reward research direction. The current Phase 1-3 plan is safe but may only produce incremental correlational results. The revised plan risks failure but has a chance at fundamental insights about reasoning

---

## 18. References

Key papers forming the foundation:

Jin et al. (2025). Analyzing singular vector structure in RLVR vs SFT.
Huan et al. (2025). Transferability Index and PCA shift analysis.
Chu et al. (2025). SFT Memorizes, RL Generalizes.
Afzal et al. (2025). Knowing Before Saying: Decision encoding before generation.
Turpin et al. (2023). Unfaithful Explanations in Chain-of-Thought.
Ley et al. (2024). On the Hardness of Faithful Chain-of-Thought Reasoning.
Huh et al. (2024). Platonic Representation Hypothesis.
Fermanian et al. (2023). Path signatures for machine learning.
Storm et al. (2024). Lyapunov ridges in neural networks.
Davari et al. (2022). Reliability of CKA.
Tang et al. (2025). CAST: Cross-task activation steering.

Negative results to engage:
- Papers showing CKA limitations
- RLVR random reward results for Qwen
- Baseline audit papers showing RLVR artifacts
- Faithfulness intervention failures


## 13. Appendix: Implementation Details


### A1. Jacobian Estimation

We estimate the Jacobian of the layer-to-layer transformation using finite differences.

```python
def estimate_jacobian(model, layer_idx, activation, eps=1e-4):
    """
    Estimate Jacobian of transformation from layer_idx to layer_idx+1.
    
    Args:
        model: The transformer model
        layer_idx: Which layer transition to analyze
        activation: The activation vector at layer_idx (shape: d_model)
        eps: Perturbation size for finite differences
    
    Returns:
        jacobian: Estimated Jacobian matrix (shape: d_model x d_model)
    """
    d = activation.shape[-1]
    jacobian = torch.zeros(d, d)
    
    # Base output
    with torch.no_grad():
        base_output = model.run_from_layer(activation, layer_idx, layer_idx + 1)
    
    # Perturb each dimension
    for i in range(d):
        perturbed = activation.clone()
        perturbed[i] += eps
        with torch.no_grad():
            perturbed_output = model.run_from_layer(perturbed, layer_idx, layer_idx + 1)
        jacobian[:, i] = (perturbed_output - base_output) / eps
    
    return jacobian


def compute_local_sensitivity(model, activations, layer_idx, eps=1e-4):
    """
    Compute local sensitivity (max singular value of Jacobian) for many points.
    """
    sensitivities = []
    for act in activations:
        J = estimate_jacobian(model, layer_idx, act, eps)
        max_sv = torch.linalg.svdvals(J)[0].item()
        sensitivities.append(max_sv)
    return sensitivities
```

This is computationally expensive (O(d) forward passes per point). For d=4096, we subsample dimensions or use random projections.


### A2. Path Signature Computation

```python
import signatory

def compute_trajectory_signature(activations_by_layer, depth=3):
    """
    Compute path signature for layer trajectory.
    
    Args:
        activations_by_layer: List of activations, one per layer (each shape: d_model)
        depth: Signature truncation depth
    
    Returns:
        signature: Path signature tensor
    """
    # Stack into path: (n_layers, d_model)
    path = torch.stack(activations_by_layer, dim=0)
    
    # Add batch dimension: (1, n_layers, d_model)
    path = path.unsqueeze(0)
    
    # Compute signature
    sig = signatory.signature(path, depth=depth)
    
    return sig.squeeze(0)
```

For d_model=4096, depth=3 signature has O(d^3) = 10^11 features. We must reduce dimensionality first:

```python
def compute_signature_with_projection(activations_by_layer, depth=3, proj_dim=64):
    """
    Project to lower dimension before computing signature.
    """
    # Random projection matrix (fixed for reproducibility)
    proj_matrix = get_random_projection(d_model=4096, proj_dim=proj_dim, seed=42)
    
    # Project each layer's activations
    projected = [proj_matrix @ act for act in activations_by_layer]
    
    # Now compute signature in projected space
    path = torch.stack(projected, dim=0).unsqueeze(0)
    sig = signatory.signature(path, depth=depth)
    
    return sig.squeeze(0)
```


### A3. Sample Variance Computation

```python
def compute_trajectory_variance(model, input_text, n_samples=100, temperature=1.0):
    """
    Measure how much trajectory geometry varies across samples.
    """
    signatures = []
    
    for _ in range(n_samples):
        # Generate with sampling
        output, activations = model.generate_with_activations(
            input_text, 
            temperature=temperature,
            do_sample=True
        )
        
        # Compute signature
        sig = compute_trajectory_signature(activations)
        signatures.append(sig)
    
    # Compute variance of signature vectors
    signatures = torch.stack(signatures)
    variance = signatures.var(dim=0).mean().item()
    
    return variance
```


### A4. Subspace Preservation Score

```python
def compute_preservation_score(base_activations, finetuned_activations, k=100):
    """
    Measure how much of the base model's top-k subspace is preserved.
    """
    # SVD of both
    _, _, V_base = torch.linalg.svd(base_activations, full_matrices=False)
    _, _, V_ft = torch.linalg.svd(finetuned_activations, full_matrices=False)
    
    # Top-k subspaces
    V_base_k = V_base[:k, :]
    V_ft_k = V_ft[:k, :]
    
    # Principal angles via SVD of product
    M = V_base_k @ V_ft_k.T
    singular_values = torch.linalg.svdvals(M)
    
    # Clamp for numerical stability
    singular_values = torch.clamp(singular_values, -1, 1)
    angles = torch.arccos(singular_values)
    
    # Preservation score: mean cosine of angles
    preservation = torch.cos(angles).mean().item()
    
    return preservation, angles
```
