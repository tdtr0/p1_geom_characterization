# Geometric Signatures of Correct Computation in LLMs

**Project: ManiVer (Manifold Verification)**

## Executive Summary

This research investigates whether *correct* task completion has distinguishable geometric signatures in activation trajectories, and whether these signatures transfer across domains. We take an explicitly **interpolation-centric view** (Allen-Zhu & Li, 2024): transformers compute smooth functions over their representation manifold, and what we observe as "reasoning" is traversal through this space.

**We deliberately avoid cognitive framing.** We do not claim to detect "reasoning" vs "recall" — these may not be separable computational modes. Instead, we ask:

1. **Can we characterize** the geometry of correct vs incorrect solutions within verifiable domains?
2. **Do these geometric signatures share structure** across domains (math → code → logic)?
3. **Can interventions on trajectory geometry** improve task performance?

**Core framework**: The residual stream is the locus of computation. Each layer transforms representations via interpolation through the learned manifold. Correct solutions may traverse this manifold differently than incorrect ones — not because of different "cognitive modes," but because of different dynamical properties (attractor basins, stability, curvature regimes).

---

## Background: What Phase 1 Established

Phase 1 (complete) showed RLVR and SFT models have different static geometry:

| Model | Subspace Preservation vs Base |
|-------|-------------------------------|
| RL-Zero | 98.6% ± 1.1% |
| SFT | 52.4% ± 12.6% |
| Think (RLVR) | 50.4% ± 12.9% |

**Key finding**: RL-Zero preserves base model subspace structure. SFT variants do not.

**Limitation**: This tells us models *differ*, but not whether geometry captures *task performance quality*.

---

## Motivating Result: Linear Methods Don't Capture Reasoning

### SVD Analysis: Is Reasoning Linearly Separable?

Before pursuing dynamical analysis, we tested a simpler hypothesis: **Does RLVR create separable "reasoning subspaces" in top eigenvectors?**

**Method**: Compared SVD of activation matrices between `olmo3_base` and `olmo3_rl_zero`:
- Computed eigenvector alignment: `delta_k = 1 - |cos(v_base_k, v_rlvr_k)|`
- Compared top-10 vs tail-50 eigenvectors across 16 layers

**Prediction if reasoning is separable**:
- Top eigenvectors should change MORE (reasoning directions refined)
- Tail eigenvectors should change LESS (knowledge preserved)

**Results** (2026-01-19):

| Task | Top-10 Delta | Tail-50 Delta | Ratio | Interpretation |
|------|-------------|---------------|-------|----------------|
| HumanEval | 0.008 (0.8%) | 0.073 (7.3%) | 0.12 | **TAIL-HEAVY** |
| GSM8K | 0.022 (2.2%) | 0.067 (6.7%) | 0.33 | **TAIL-HEAVY** |

**Finding: OPPOSITE of separable reasoning hypothesis.**

Tail eigenvectors change 3-8x MORE than top eigenvectors. RLVR:
1. **Preserves top eigenvectors** — core representational structure unchanged
2. **Refines tail eigenvectors** — adjusts low-variance, fine-grained directions
3. **Does NOT create distinct "reasoning subspaces"**

### Why This Motivates Dynamical Analysis

This negative result supports the **interpolation view** (Allen-Zhu & Li, 2024):
- There is no separate "reasoning mode" or "reasoning subspace"
- Reasoning is not captured by linear projections onto principal components
- RLVR refinement is distributed, not localized to specific directions

**Implication**: If reasoning isn't in the **space** (static subspace), it may be in the **flow** (trajectory dynamics).

This motivates our focus on:
- **Vector field analysis**: How activations flow through layers
- **Lyapunov stability**: Whether correct solutions are more stable
- **Attractor dynamics**: Which basins correct vs incorrect solutions converge to
- **Path signatures**: Trajectory shape features (reparameterization-invariant)

*Full analysis: `experiments/svd_reasoning_separability/` and `notebooks/working_notes/SVD_LINEAR_SEPARABILITY_FINDINGS.md`*

---

## Theoretical Framework: Interpolation, Attractors, and Belief Dynamics

### The Allen-Zhu View: Everything is Interpolation

Following Allen-Zhu & Li (2024), we treat the transformer as computing a smooth function:

```
f: input_embedding → output_logits
```

There is no "reasoning mode" vs "recall mode." The forward pass is identical regardless of task difficulty. What differs is the *region* of the representation manifold being traversed:

- **Dense regions** (familiar inputs): Short, direct paths to outputs
- **Sparse regions** (novel inputs): Longer, more complex traversals

**Implication**: Apparent "reasoning" signatures might reflect input regime differences, not computational mode differences. Our controls must address this.

### Curvature Regimes (Merullo et al., 2025)

Recent work on loss curvature shows that weight directions separate by function:

| Curvature | Interpretation | Example |
|-----------|----------------|---------|
| **High** | Used by many examples; general computation | Shared reasoning circuits |
| **Low** | Used by few examples; memorization-like | Lookup tables, specific facts |

**Critical finding**: Math/arithmetic relies heavily on low-curvature (memorization-like) structure. The model can reason about problem structure but looks up arithmetic facts.

**Implication for us**: If correct solutions use more high-curvature (general) computation, trajectory geometry might capture this. We can validate by checking curvature regime activation.

### Attractor Dynamics (Ren & Liu, 2026)

The Hierarchical Reasoning Model analysis shows that even explicit iterative reasoning involves:

- **Multiple fixed points** in latent space (some correct, some wrong)
- **Grokking transitions**: Loss stays flat then drops suddenly (finding right attractor)
- **Attractor trapping**: Incorrect solutions get stuck in wrong basins

**Implication**: Correct vs incorrect solutions may differ in attractor dynamics — not "quality of reasoning" but "which basin you land in." Our Lyapunov analysis can test this.

### Belief State Geometry (Shai et al., 2024; Bigelow et al., 2025)

The residual stream linearly represents belief states over latent generative structure:

- **In-context learning**: Bayesian evidence accumulation
- **Activation steering**: Prior modification
- Both operate on the same belief representation

**Implication**: Correct solutions might show different belief update trajectories — evidence-responsive updates leading to correct posteriors vs prior-dominated updates leading to errors.

### Synthesis: What We're Actually Testing

We're not detecting "reasoning" as a cognitive category. We're testing:

1. **Do correct solutions have different dynamical signatures?** (stability, attractor basins, curvature profiles)
2. **Are these signatures domain-general?** (shared structure across math, code, logic)
3. **Are they causal?** (can interventions improve performance)

This is pure function approximation analysis dressed in geometric language.

---

## Novelty and Prior Work

### What Already Exists

**Hidden states encode correctness** (Zhang et al. 2025, Afzal et al. 2025, Azaria & Mitchell 2023):
- Probes can predict whether a model will answer correctly
- This works even before the answer is generated
- **Limitation**: Tested only within single domains (math, or QA, or code)

**Truth has geometric structure** (Marks & Tegmark 2023):
- Linear truth directions transfer across factual datasets
- **Limitation**: All tests are on factual recall, not reasoning processes

**Activation steering works** (Turner et al. 2023, Meng et al. 2022):
- Can modify model behavior by editing activations
- **Limitation**: Tested on simple attributes (honesty, toxicity), not complex reasoning

### What's Novel in This Work

**1. Cross-domain transfer of correctness signatures (H2)**
- **New**: Test if geometric features that predict correctness on math also work on code and logic
- **Why it matters**: Prior work only tested within-domain; we test whether signatures are domain-general
- **Honest expectation**: This may fail. Different domains may occupy different manifold regions with different dynamics.

**2. Dynamical systems analysis of trajectories**
- **New**: Apply MARBLE-style vector field decomposition, Lyapunov stability analysis, and attractor characterization
- **Method**: Treat layer transitions as a discrete dynamical system; analyze its structure
- **Why it matters**: Connects to theoretical framework (interpolation, attractors, belief dynamics)

**3. Curvature regime analysis**
- **New**: Test whether correct solutions activate more high-curvature (general) vs low-curvature (memorization) weight directions
- **Method**: Inspired by Merullo et al. (2025); project trajectories onto weight singular vectors
- **Why it matters**: Directly tests the "reasoning vs memorization" distinction at the weight level

**4. Causal intervention (H4)**
- **New**: Steer trajectories toward "correct solution" manifold regions
- **Why it matters**: Tests whether geometry is causal, not just correlational
- **Honest expectation**: May not work; geometry might be epiphenomenal

**5. Systematic confound controls**
- **New**: Difficulty stratification, length matching, input regime controls
- **Why it matters**: The main critique is that we might detect "easy vs hard" not "correct vs incorrect"

### What We're NOT Claiming

- ❌ That we detect "reasoning" as a cognitive category
- ❌ That there's a separable "reasoning mode" vs "recall mode"
- ❌ That geometry *causes* correct solutions (only that it correlates, unless H4 succeeds)
- ❌ That this works on all models or all tasks
- ✅ That correct and incorrect solutions may have different dynamical signatures
- ✅ That these signatures might share structure across domains (testable, may fail)

---

## The Research Question (Reframed)

> **Do correct solutions have distinguishable dynamical signatures in activation trajectories, and do these signatures share structure across verifiable domains?**

We explicitly drop the cognitive framing. We're not asking "what is reasoning?" We're asking:

1. **Descriptive**: Can we characterize geometric differences between correct and incorrect solution trajectories?
2. **Transferable**: Do these differences generalize across domains (math, code, logic)?
3. **Causal**: Can we improve task performance by intervening on trajectory geometry?

**Why this matters**: If signatures transfer, we have a domain-general correctness predictor. If they don't, we learn that task performance is geometrically domain-specific — also valuable.

---

## Hypotheses (Reframed)

### H1: Correct vs Incorrect Solutions Have Distinguishable Trajectory Dynamics

On verifiable tasks, activation trajectories for correct vs incorrect solutions should be geometrically distinguishable via dynamical features.

**What we measure**:
- Vector field structure (MARBLE decomposition): potential vs rotational flow
- Stability (Lyapunov exponents): convergence vs divergence
- Path signatures: reparameterization-invariant trajectory features
- Frenet-Serret curvature: trajectory bending

**Test**: Train classifier on dynamical features (correct vs incorrect). Cross-validate within domain.

**Success criterion**: >65% balanced accuracy (significantly above 50% chance)

**Controls**:
- Difficulty-matched pairs (to avoid detecting "easy vs hard")
- Length-matched pairs (to avoid detecting output length)
- Random label baseline (must fail, ≈50%)

**Risk**: May learn input regime (dense vs sparse manifold regions) rather than solution quality.

### H2: Correctness Signatures Share Structure Across Domains (Critical Test)

If correct solutions have domain-general dynamical properties, features learned on one domain should partially transfer to others.

**Reframed question**: Does successful interpolation through the representation manifold have shared geometric properties regardless of what's being computed?

**Test**: Train on GSM8K correct/incorrect, test on HumanEval and LogiQA (zero-shot).

**Success criterion**: >55% transfer accuracy (above 50% chance)

**What would success mean**:
- Correct solutions across domains share dynamical properties (e.g., similar stability profiles, curvature patterns)
- The "correct solution" manifold region has domain-general structure

**What would failure mean**:
- Task performance is geometrically domain-specific
- Each domain has its own "correct solution" signature
- This is still a valuable negative result

**This is the critical test.** We expect partial success at best.

### H3: Signatures Correlate with Human Judgments on Non-Verifiable Domains

Apply trained detector to domains without ground truth. Validate against human quality judgments.

**Test**: Compute geometric scores on open-ended responses. Correlate with human ratings.

**Success criterion**: r > 0.25 correlation with human judgments

**Honest assessment**: This is exploratory. Human judgments on non-verifiable domains are noisy.

### H4: Trajectory Interventions Can Improve Task Performance (Causality Test)

If geometry correlates with correctness, can we improve performance by steering trajectories?

**Test**: Project activations toward "correct solution" manifold region during inference.

**Success criterion**: >2% accuracy improvement on held-out verifiable problems

**What would success mean**: Geometry is causally relevant, not just epiphenomenal

**What would failure mean**: Geometry reflects correctness but doesn't cause it; read-only signal

### H5: Correct Solutions Have More Stable Dynamics (Exploratory)

Hypothesis: Correct solutions show lower Lyapunov exponents (more stable attractor convergence).

**Rationale**: From HRM analysis — correct solutions find the right attractor quickly; incorrect solutions wander or get trapped in wrong basins.

**Test**: Compare Lyapunov spectra for correct vs incorrect trajectories.

**Alternative hypothesis**: Correct solutions might show *controlled* instability (exploration) followed by convergence (exploitation).

---

## Experimental Design

### Data Collection (Phase 2A)

**What we collect**:
- Trajectories at even layers: [0, 2, 4, ..., 30] = 16 layers
- 500 samples per task (GSM8K, HumanEval, LogiQA)
- 4 models (Base, SFT, RL-Zero, Think)
- **Critically**: Record model outputs and correctness labels

**Storage**: ~56 GB total on eyecog

### Geometric and Dynamical Features

#### Original Features

**Path Signatures** (via signatory library):
- Reparameterization-invariant trajectory features
- Captures curvature, winding, self-intersection
- Project to 64 dims before computing (d=4096 too large)

**Frenet-Serret Curvature**:
- Local turning angles between consecutive layers
- Aggregate statistics (mean, variance, max)

#### New Dynamical Systems Features (Phase 3)

**1. MARBLE-style Vector Field Decomposition**

Treat layer transitions as a dynamical system and decompose via Helmholtz:

```python
# Layer transition dynamics
v(x) = x_{l+1} - x_l  # Velocity field

# Helmholtz decomposition
v = ∇φ + ∇×A
  = potential (gradient) + rotational (curl)
```

- **Potential component**: Gradient-following flow toward attractors
- **Rotational component**: Cycling, oscillatory dynamics
- **Hypothesis**: Correct solutions have higher potential/rotational ratio (more direct paths)

**2. Lyapunov Exponent Analysis**

Measure trajectory stability/instability:

```python
# Local Jacobian of layer transition
J_l = ∂x_{l+1}/∂x_l

# Lyapunov exponent
λ = lim (1/L) Σ_l log(σ_max(J_l))
```

- λ > 0: Divergent (chaotic, sensitive)
- λ < 0: Convergent (stable, attractor)
- λ ≈ 0: Neutral

**Hypothesis**: Correct solutions show more stable dynamics (λ < 0) or controlled instability-then-convergence.

**3. Attractor Analysis**

Characterize fixed points and basins:

- **Fixed point detection**: Where does v(x) ≈ 0?
- **Basin estimation**: Which initial conditions converge to which fixed points?
- **Hypothesis**: Correct/incorrect solutions converge to different attractor basins

**4. Activation Regime Analysis (Proxy for Curvature)**

Inspired by Merullo et al. (2025), but using structural proxies (we lack gradient statistics for true K-FAC):

- Project trajectories onto weight singular value regimes
- Measure activation of high-singular-value (distributed) vs low-singular-value (localized) directions
- Compute effective dimensionality of activations
- **Hypothesis**: Correct solutions use more distributed (high-SV) computation
- **Caveat**: This is NOT true curvature analysis — it's a structural proxy

#### Feature Summary

| Feature | What it Captures | Hypothesis for Correct Solutions |
|---------|------------------|----------------------------------|
| Path signature | Trajectory shape (invariant) | More structured, less chaotic |
| Frenet-Serret curvature | Local bending | Lower curvature (straighter paths) |
| Vector field potential/curl | Flow structure | Higher potential ratio (direct paths) |
| Lyapunov exponent | Stability | More stable (λ < 0) |
| Attractor basin | Convergence target | Different basin than incorrect |
| Activation regime (proxy) | Weight direction usage (structural proxy) | More distributed (high-SV) activation |

### Classification Pipeline

```python
# H1 Test: Within-domain classification
for model in models:
    for task in tasks:
        correct_sigs = get_signatures(trajectories[correct])
        incorrect_sigs = get_signatures(trajectories[incorrect])

        clf = RandomForestClassifier()
        accuracy = cross_val_score(clf, all_sigs, labels, cv=5).mean()

# H2 Test: Cross-domain transfer
clf = train(math_correct, math_incorrect)
code_accuracy = clf.evaluate(code_correct, code_incorrect)
logic_accuracy = clf.evaluate(logic_correct, logic_incorrect)
```

### Confound Controls

**Critical confounds to address**:

**1. Problem Difficulty**
- Easy problems may have shorter, more direct trajectories
- Hard problems may exhibit more "wandering"
- **Control**: Stratify by difficulty proxy (problem length, model perplexity)
- **Test**: Train separate classifiers for easy/medium/hard problems
- **Success**: Accuracy remains >60% within each difficulty stratum

**2. Output Length**
- Correct solutions may systematically differ in length
- **Control**: Match correct/incorrect pairs by token count (±10%)
- **Test**: Classification accuracy on length-matched pairs

**3. Surface Format**
- Different domains have different output formats
- **Control**: Test within-format transfer (GSM8K → MATH, both math)
- **Test**: If within-format transfer succeeds but cross-format fails, it's format not reasoning

**4. Random Label Control** (Hewitt & Liang 2019)
- Train classifier on shuffled labels
- **Success criterion**: Control accuracy ≈50% (geometry encodes meaningful signal)
- **Failure criterion**: Control accuracy >55% (classifier learns spurious features)

### Baseline Comparisons

Geometry-based methods should outperform or complement simpler baselines:

**Baseline 1: Model Confidence**
- Use model's own probability estimates: P(correct | logits)
- Expected performance: 60-70% (Kadavath et al. 2022)

**Baseline 2: Output Length**
- Classify based on token count alone
- Expected performance: 55-60%

**Baseline 3: Semantic Entropy** (if compute allows)
- Sample multiple outputs, measure semantic consistency
- Expected performance: 70-75% (Farquhar et al. 2024)

**Success criterion**: Trajectory geometry achieves >65% AND adds value beyond baselines (e.g., ensemble with confidence improves over either alone)

---

## Timeline

### Weeks 1-2: Data Collection
- Run cleanup on eyecog (`./scripts/cleanup_smallworld.sh`)
- Collect trajectories with correctness labels
- 500 samples × 3 tasks × 4 models

### Weeks 3-4: H1 Test
- Compute path signatures
- Train correct/incorrect classifiers per domain
- Report accuracy and feature importance

### Weeks 5-6: H2 Test (Decision Point)
- Cross-domain transfer of classifiers
- **If fails**: Pivot to understanding WHY (what's domain-specific?)
- **If succeeds**: Proceed to intervention

### Weeks 7-10: H4 Test (If H2 Succeeds)
- Implement trajectory steering
- Test on held-out verifiable problems

### Weeks 11-12: Write-up
- Document results (including negative results)
- Prepare paper

---

## Anticipated Challenges and Mitigation Strategies

### Challenge 1: The Decision-Before-Reasoning Problem

**Issue**: Recent work (Afzal et al. 2025, David 2025) shows models commit to answers early in CoT generation. Trajectory geometry may capture elaboration quality, not decision-making.

**Mitigation**:
- Analyze layer-wise: Test if early layers (0-15) are more predictive than late layers (16-31)
- If early layers dominate, we're capturing decision structure (valuable)
- If late layers dominate, we're capturing elaboration (still potentially useful)

**Implication**: Reframe findings accordingly—be explicit about what we're measuring.

### Challenge 2: Confounds May Dominate Signal

**Issue**: Geometry may distinguish difficulty/length/format rather than reasoning quality.

**Mitigation**:
- **Difficulty stratification**: Mandatory for all analyses
- **Length matching**: Compare correct/incorrect of similar token count
- **Within-format transfer**: Test GSM8K → MATH before GSM8K → HumanEval
- **Control tasks**: Random label baseline must fail (≈50% accuracy)

**Decision point**: If controls show confounds dominate, pivot to understanding what geometry actually captures.

### Challenge 3: H2 May Fail (Critical Risk)

**Issue**: Cross-domain transfer is the linchpin hypothesis. If it fails, the "universal geometry" premise fails.

**Mitigation**:
- **Hierarchical transfer**: Test closer domains first (GSM8K → MATH)
- **Asymmetric transfer**: Check if math → code works better than code → math
- **Feature decomposition**: Identify which geometric features transfer vs which don't

**Pivot strategies** (if H2 fails):
1. Characterize domain-specific vs shared geometric features
2. Measure transfer as function of domain similarity
3. Focus on within-domain applications (still useful)

### Challenge 4: Sample Size for Transfer Tests

**Issue**: 500 samples per task may yield imbalanced classes (e.g., 350 correct, 150 incorrect).

**Mitigation**:
- Monitor class balance during collection
- If model is too accurate (>80%), sample harder problems
- Ensure minimum 100 samples per class for robust classification

### Challenge 5: HumanEval Correctness is Expensive

**Issue**: Running test cases for every sample is slow and potentially unsafe (code execution).

**Mitigation**:
- Use sandboxed execution environment (Docker container with timeouts)
- Collect HumanEval last (after GSM8K and LogiQA validate the approach)
- Consider using pass@1 from existing benchmarks if available

### Challenge 6: Path Signatures May Be Brittle

**Issue**: Signatures require dimensionality reduction (4096 → 64 dims). Choice of projection may affect results.

**Mitigation**:
- Test multiple projection methods (PCA, random projection, UMAP)
- Test multiple signature depths (2, 3, 4)
- Report robustness across choices

---

## Success Criteria (Reframed)

| Outcome | Implication | Next Step |
|---------|-------------|-----------|
| H1 success, H2 success | Correct solutions have domain-general dynamical signatures | Proceed to H4 (intervention) |
| H1 success, H2 fails | Correct solution signatures are domain-specific | Characterize per-domain structure (still valuable) |
| H1 fails | Trajectory dynamics don't distinguish correct/incorrect | Major pivot: try attention patterns, gradients |
| H4 success | Trajectory geometry is causally relevant | Major contribution: geometry enables intervention |
| H4 fails | Geometry is correlational, not causal | Geometry is read-only signal (still useful for detection) |

---

## Honest Assessment of Risks

### The Core Confound

**"Correct reasoning" geometry might just be "easy problem" geometry.**

If easy problems have certain signatures and hard problems have others, we learn a difficulty detector, not a reasoning quality detector.

**Mitigation**: Analyze within difficulty strata. Compare correct vs incorrect on problems of similar difficulty.

### What We're Likely to Show
- H1 will probably succeed (geometry distinguishes *something*)
- The question is whether it's reasoning vs. difficulty vs. length vs. other confounds

### What's Uncertain
- H2 (cross-domain transfer) - this is the critical unknown
- Whether any intervention (H4) will work

### Even if We Fail
- We learn geometry doesn't capture transferable reasoning
- This is valuable negative result that saves others from this path

---

## File Structure

```
ManiVer/
├── RESEARCH_PLAN.md              # This file (main plan)
├── PHASE2_PLAN.md                # Data collection details
├── phase1_implementation_plan.md # Phase 1 (complete)
├── archive_transfer_correlation_plan.md  # Old approach (archived)
├── paper/
│   └── geometric_compression_research_plan.md  # Full technical details
├── scripts/
│   ├── collect_trajectories_half_layers.py  # Collection script
│   ├── cleanup_smallworld.sh                # Disk cleanup
│   └── run_analysis.py                      # Analysis pipeline
├── src/
│   ├── activation_collector.py
│   ├── task_data.py
│   └── geometric_measures.py
└── data/
    ├── activations/    # Phase 1 data
    └── trajectories/   # Phase 2 data (to be collected)
```

---

## Compute Resources

- **Server**: eyecog (2x RTX 3090 24GB)
- **Storage**: ~160 GB available after cleanup
- **Estimated GPU hours**: 40-60 for collection + analysis

---

## Key Decisions Made

1. **Even layers only**: Layer smoothness analysis shows negligible difference between consecutive layers
2. **500 samples**: Need enough for correct/incorrect split with sufficient N
3. **Correctness labels**: Critical addition - without these, we can't test H1/H2
4. **Path signatures**: Primary trajectory feature (reparameterization-invariant)
5. **Focus on verifiable domains first**: Math, code, logic have ground truth

---

## What's NOT in This Plan

1. **Optimal transport / rectified flow**: Deferred until H1-H2 show signal
2. **Human evaluation (H3)**: Only if H2 succeeds and resources allow
3. **Fine-grained layer analysis**: Start with aggregate, refine if needed
4. **Multiple temperature sampling**: Focus on greedy first

These can be added if initial results warrant.

---

## Key References

### Theoretical Framework

**Interpolation View**:
- Allen-Zhu, Z., & Li, Y. (2024). Physics of Language Models (Parts 1-3). *ICML 2024 Tutorial*.
  - Key insight: Transformers compute smooth functions; "reasoning" is interpolation through representation space.

**Curvature Regimes**:
- Merullo, J., Vatsavaya, S., Bushnaq, L., & Lewis, O. (2025). Understanding Memorization via Loss Curvature. *arXiv:2510.24256*.
  - Key insight: High-curvature weights = general computation; low-curvature = memorization. Math uses memorization-like circuits.

**Attractor Dynamics**:
- Ren, Z., & Liu, Z. (2026). Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models. *arXiv:2601.10679*.
  - Key insight: Iterative reasoning shows attractor dynamics with "grokking" transitions, not gradual refinement.

**Belief State Geometry**:
- Shai, M., et al. (2024). Transformers Represent Belief State Geometry in their Residual Stream. *NeurIPS 2024*. arXiv:2405.15943.
  - Key insight: Residual stream linearly represents belief states, even with fractal geometry.
- Bigelow, E., et al. (2025). Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering. *arXiv:2511.00617*.
  - Key insight: ICL = evidence accumulation; steering = prior modification; same belief representation.

**Vector Field Analysis**:
- Gosztolai, A., & Bhattacharyya, R. (MARBLE papers). Manifold-based Analysis of Neural Population Dynamics.
  - Key insight: Neural dynamics can be decomposed into potential (gradient) and rotational (curl) components.

### Supporting Evidence

**Hidden states encode correctness**:
- Zhang et al. (2025), Afzal et al. (2025), Azaria & Mitchell (2023)
- Limitation: Only tested within single domains.

**Truth has geometric structure**:
- Marks, S., & Tegmark, M. (2023). The Geometry of Truth.
- Limitation: Tested on factual recall, not reasoning processes.

**Activation steering works**:
- Turner, A., et al. (2023). Activation Addition.
- Meng, K., et al. (2022). ROME.
- Limitation: Tested on simple attributes, not complex reasoning.

**Trajectory structure**:
- Hosseini & Fedorenko (2023): Trajectories straighten with task success.

### Critical Challenges

**Decision before reasoning**:
- Afzal et al. (2025), David (2025): Models commit to answers early in CoT.

**CoT unfaithfulness**:
- Turpin, M., et al. (2023). Language Models Don't Always Say What They Think.

**Probe controls**:
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks.

**Transfer failures**:
- Ley, D., et al. (2024). Faithfulness interventions fail to transfer.

### Methodological

**Path signatures**:
- Kidger, P., & Lyons, T. (2020). Signatory: differentiable computations of the signature and logsignature transforms.

**Lyapunov analysis**:
- Standard dynamical systems texts; adapted for discrete layer transitions.

**Helmholtz decomposition**:
- Standard vector calculus; applied to neural activation flows.
