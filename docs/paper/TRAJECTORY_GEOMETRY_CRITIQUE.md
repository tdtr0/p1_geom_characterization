# Trajectory Geometry: Intuition, Critique, and Alternatives

**Question**: Is analyzing activation trajectories through layers the right approach for understanding reasoning quality?

---

## 🧠 Intuition: Why Trajectory Geometry?

### The Core Idea

**Reasoning is a process, not a state**. When a transformer solves a problem, it doesn't just "have" the answer—it *computes* it through a sequence of transformations across layers.

**Analogy**: Think of reasoning like navigation
- **Static analysis** (Phase 1): Taking a photo of where the car ended up
- **Trajectory analysis** (Phase 2-4): Recording the path the car took to get there

**Key insight**: Two cars can end up at the same destination via very different routes. Similarly, two models might have similar final activations but very different computational paths.

### What Trajectories Capture

**Layer-by-layer transformation**:
```
Input: "What is 15 × 23?"
Layer 0:  [raw token embeddings]
Layer 5:  [recognizes multiplication problem]
Layer 10: [retrieves multiplication facts]
Layer 15: [performs computation]
Layer 20: [formats answer]
Layer 30: [final output representation]
```

**Geometric properties of this path**:
- **Curvature**: How much does the path bend? (Straight = confident, curved = exploring)
- **Length**: How far does the representation travel? (Short = efficient, long = inefficient)
- **Winding**: Does the path loop back on itself? (Smooth = coherent, tangled = confused)

### Why This Might Work

**Hypothesis 1: Correct reasoning is more direct**
- Model "knows where it's going" → straighter path
- Incorrect reasoning involves exploration, backtracking → curved path
- **Evidence**: Hosseini & Fedorenko (2023) showed successful processing straightens trajectories

**Hypothesis 2: Correct reasoning has universal structure**
- Math, code, and logic all involve step-by-step inference
- The *process* of correct reasoning may be similar across domains
- Even if content differs, the computational flow may be universal

**Hypothesis 3: Trajectories enable intervention**
- If we know what "correct" paths look like, we can steer generation to stay on those paths
- This is more principled than just "add a vector" (Turner et al.)
- Trajectory steering = process-level intervention

---

## ⚠️ Critique: Why This Might NOT Work

### Problem 1: Decision-Before-Reasoning

**Evidence** (Afzal et al. 2025, David 2025):
- Models commit to answers in early layers (0-15)
- Later layers (16-31) are elaboration, not decision-making
- The "trajectory" we measure is mostly post-hoc rationalization

**Implication**: 
- Trajectory geometry may capture **elaboration quality**, not **reasoning quality**
- Early layers matter more than late layers
- The "flow" is an illusion—the decision is already made

**Counter-argument**:
- Even if elaboration, it may still transfer (good elaboration is universal)
- We can test this: Compare early-layer vs late-layer predictiveness

### Problem 2: Trajectories Are High-Dimensional and Noisy

**Challenge**: Activations are 4096-dimensional
- Path signatures require dimensionality reduction (4096 → 64)
- This introduces artifacts (what if the "real" signal is in the discarded dimensions?)
- Different projection methods may yield different results

**Implication**:
- Results may be brittle (sensitive to projection choice)
- Hard to interpret (what does a "signature coefficient" mean?)
- May not generalize across model architectures

### Problem 3: Trajectories Conflate Multiple Processes

**What's happening in a trajectory**:
1. Feature extraction (early layers)
2. Decision-making (middle layers)
3. Elaboration (late layers)
4. Output formatting (final layers)

**Problem**: We measure the *whole* trajectory, but only care about decision-making.

**Implication**:
- Trajectory geometry may be dominated by irrelevant processes (formatting, elaboration)
- The signal we want (decision quality) may be buried in noise

### Problem 4: Confounds May Be Insurmountable

**Difficulty confound**:
- Easy problems: Short, direct trajectories (low curvature)
- Hard problems: Long, exploratory trajectories (high curvature)
- This is true *regardless* of correctness

**Length confound**:
- Correct solutions may be systematically longer (more detailed)
- Trajectory length correlates with output length
- Geometry may just be measuring verbosity

**Format confound**:
- Math outputs: "#### 42"
- Code outputs: Function definitions with indentation
- These format differences may dominate geometric differences

**Implication**: Even with controls, confounds may be too strong to isolate the "reasoning quality" signal.

---

## 🔄 Alternative Methods (If Trajectory Geometry Fails)

### Alternative 1: Early-Layer Decision Probing

**Concept**: Focus on layers 0-15 where the decision is made (Afzal et al. 2025).

**Method**:
```python
# Extract activations at layer 10 (middle of decision phase)
decision_activations = get_activations(model, prompt, layer=10)

# Train probe: decision_activation → correctness
probe = LogisticRegression()
probe.fit(decision_activations, correctness_labels)

# Test cross-domain transfer
transfer_accuracy = probe.score(code_decision_activations, code_labels)
```

**Advantages**:
- Simpler (single layer, not full trajectory)
- Targets decision-making directly
- Less confounded by elaboration

**Disadvantages**:
- Loses temporal dynamics
- May miss multi-step reasoning

**When to use**: If trajectory analysis shows late layers don't matter.

### Alternative 2: Attention Pattern Analysis

**Concept**: Analyze attention weights, not activation values.

**Hypothesis**: Correct reasoning has characteristic attention patterns
- Attends to relevant tokens
- Builds coherent context
- Avoids distraction

**Method**:
```python
# Extract attention patterns
attention_patterns = get_attention_weights(model, prompt)  # (n_layers, n_heads, seq_len, seq_len)

# Compute attention features
features = {
    'entropy': attention_entropy(attention_patterns),  # How focused?
    'locality': attention_locality(attention_patterns),  # Local vs global?
    'consistency': attention_consistency(attention_patterns),  # Stable across layers?
}

# Classify correct vs incorrect
clf.fit(features, correctness_labels)
```

**Advantages**:
- Attention is interpretable (can visualize which tokens matter)
- May capture reasoning structure more directly
- Complementary to activation analysis

**Disadvantages**:
- Attention may not reflect information flow (see: "Attention is not Explanation")
- High-dimensional (n_layers × n_heads × seq_len²)
- May be noisy

**When to use**: If activation trajectories are too noisy or uninterpretable.

### Alternative 3: Gradient-Based Attribution

**Concept**: Use gradients to identify which activations causally affect correctness.

**Method**:
```python
# For each sample, compute gradient of correctness w.r.t. activations
model.eval()
activations = get_activations_with_grad(model, prompt)

# Gradient of output probability w.r.t. activations
grad = torch.autograd.grad(output_prob, activations)

# Activation importance = gradient magnitude
importance = grad.abs().mean(dim=0)  # (n_layers, d_model)

# Compare correct vs incorrect importance patterns
```

**Advantages**:
- Directly measures causal relevance (gradient = sensitivity)
- Can identify which layers/dimensions matter
- More principled than correlation-based probing

**Disadvantages**:
- Computationally expensive (requires backward pass)
- Gradients can be noisy
- Requires differentiable correctness metric

**When to use**: If we want causal evidence without intervention experiments.

### Alternative 4: Semantic Entropy (Farquhar et al. 2024)

**Concept**: Sample multiple outputs, measure semantic consistency.

**Method**:
```python
# Generate multiple outputs for same input
outputs = [model.generate(prompt, temperature=1.0) for _ in range(10)]

# Cluster by semantic meaning (using NLI)
clusters = cluster_by_meaning(outputs)

# Compute entropy over clusters
semantic_entropy = entropy(cluster_distribution)

# High entropy = uncertain = likely incorrect
```

**Advantages**:
- Already validated (Nature paper)
- Doesn't require ground truth
- Works across domains

**Disadvantages**:
- Expensive (10× generation cost)
- Requires NLI model for clustering
- Doesn't explain *why* model is uncertain

**When to use**: As a baseline to beat, or if geometry fails.

### Alternative 5: Mechanistic Interpretability (Circuit Analysis)

**Concept**: Identify specific circuits (attention heads + MLPs) responsible for reasoning.

**Method** (following Anthropic's work):
- Use activation patching to identify which components matter
- Trace information flow through specific circuits
- Analyze circuit behavior on correct vs incorrect samples

**Advantages**:
- Most mechanistic (understands *how* reasoning works)
- Can identify failure modes precisely
- Highly interpretable

**Disadvantages**:
- Extremely labor-intensive
- Model-specific (circuits may differ across architectures)
- Requires deep expertise

**When to use**: If we want deep mechanistic understanding, not just prediction.

---

## 🎯 Recommended Approach

### Primary: Trajectory Geometry (Current Plan)

**Proceed with trajectory analysis** because:
1. Novel contribution (not done before for reasoning)
2. Captures temporal dynamics (reasoning is a process)
3. Enables steering (causal intervention)
4. Mathematically principled (path signatures)

**But add safeguards**:
1. **Layer-wise analysis**: Test if early layers are more predictive (decision-before-reasoning)
2. **Confound controls**: Mandatory difficulty/length/format stratification
3. **Baseline comparisons**: Must beat simpler methods
4. **Robustness checks**: Test multiple projection methods and signature depths

### Backup: Early-Layer Probing

**If trajectory analysis fails** (H1 < 60% or H2 < 52%):
- Pivot to early-layer decision probing
- Simpler, more direct
- May be more robust

### Complementary: Attention Analysis

**Run in parallel** (low cost):
- Collect attention patterns during trajectory collection
- Analyze as secondary feature set
- May provide interpretability even if trajectories don't transfer

---

## 📊 Anticipated Challenges (Detailed)

### Challenge 1: Path Signatures May Not Capture Reasoning

**Why signatures might fail**:
- Designed for continuous paths (time series), not discrete layer transitions
- May be sensitive to reparameterization (even though theoretically invariant)
- High-dimensional signatures (exponential in depth) require aggressive truncation

**Mitigation**:
- Test multiple signature depths (2, 3, 4)
- Compare to simpler trajectory features (curvature, length, variance)
- If signatures fail, fall back to hand-crafted features

### Challenge 2: Projection Artifacts

**Problem**: Reducing 4096 → 64 dims loses information.

**Questions**:
- What if the "reasoning signal" is in the discarded 4032 dimensions?
- Different projection methods (PCA, random, UMAP) may give different results

**Mitigation**:
- Test multiple projection methods
- Try different target dimensions (32, 64, 128, 256)
- Report robustness across choices
- If results are brittle, acknowledge limitation

### Challenge 3: Trajectory Variance Within Correct/Incorrect

**Problem**: "Correct" trajectories may be heterogeneous
- Correct via method A (algebraic)
- Correct via method B (numerical)
- Correct via method C (pattern matching)

**Implication**: Averaging over all "correct" trajectories may wash out structure.

**Mitigation**:
- Cluster trajectories within correct/incorrect classes
- Analyze clusters separately
- May find multiple "correct reasoning" modes

### Challenge 4: Steering May Fail for Technical Reasons

**Ways steering can fail**:
1. **Manifold is wrong**: Learned from insufficient data
2. **Layers are wrong**: Steering early/late instead of middle
3. **Strength is wrong**: α too high (breaks model) or too low (no effect)
4. **Method is wrong**: Projection is too crude; need learned map

**Implication**: Negative H4 result is ambiguous (geometry doesn't matter? or steering is wrong?).

**Mitigation**:
- Extensive hyperparameter search
- Multiple steering methods (projection, addition, OT map)
- Sanity checks (steering on training set should work)
- Ablations (correct vs random vs incorrect manifold)

---

## 🔬 Is This Approach Meaningful?

### Arguments FOR Trajectory Geometry

**1. Reasoning IS a process**
- Each layer performs a computation
- The sequence of computations IS the reasoning
- Trajectories capture this directly

**2. Path signatures are principled**
- Reparameterization-invariant (don't depend on layer indexing)
- Capture geometric properties (curvature, winding)
- Used successfully in time series analysis

**3. Enables causal intervention**
- Can steer trajectories during inference
- Tests whether geometry matters (not just correlates)
- Provides actionable insights

### Arguments AGAINST Trajectory Geometry

**1. Decision is made early**
- Afzal et al. (2025): Correctness predictable before generation
- Later trajectory is elaboration, not decision
- May be measuring the wrong thing

**2. Post-hoc rationalization**
- Turpin et al. (2023): CoT can be unfaithful
- Trajectory may reflect plausible explanation, not true reasoning
- Geometry of rationalization ≠ geometry of reasoning

**3. Confounds may dominate**
- Difficulty, length, format may be stronger signals than reasoning quality
- Even with controls, may not isolate the target construct
- Risk: Measure something, but not what we intended

### Verdict: Proceed with Caution

**The trajectory geometry approach is worth trying** because:
- Novel (not done before for reasoning)
- Theoretically motivated (reasoning is a process)
- Enables causal tests (steering)

**But be prepared for failure** because:
- Decision-before-reasoning problem is real
- Confounds are serious
- Prior work on CoT faithfulness is pessimistic

**Key**: Frame results honestly
- If it works: "Trajectory geometry captures elaboration quality, which transfers"
- If it fails: "Trajectory geometry is confounded by difficulty/format"

---

## 🚀 Recommended Execution Strategy

### Phase 2-3: Test Trajectory Approach

**Do**:
- Collect trajectories as planned
- Compute path signatures
- Test H1 and H2
- Implement all confound controls

**Decide** (after Phase 3):
- If H2 succeeds (>55% transfer): Proceed to H4 (steering)
- If H2 fails (<55% transfer): Pivot to alternatives

### If Trajectory Approach Fails

**Pivot 1: Early-Layer Probing** (simplest)
- Extract layer 10-15 activations only
- Train probes on these
- Test transfer
- Faster, simpler, more direct

**Pivot 2: Attention Analysis** (complementary)
- Analyze attention patterns
- May be more interpretable
- Can run in parallel with trajectory analysis

**Pivot 3: Hybrid Approach** (most robust)
- Combine trajectory features + attention features + static features
- Ensemble classifier
- May capture multiple aspects of reasoning

---

## 📈 Expected Outcomes

### Optimistic Scenario (60% probability)

**H1 succeeds** (>65% within-domain):
- Geometry distinguishes correct/incorrect
- But may be confounded by difficulty

**H2 partially succeeds** (52-58% transfer):
- Weak signal, but above chance
- Some geometric features transfer, others don't

**H4 fails** (no improvement):
- Geometry correlates but doesn't cause
- Steering is too crude

**Paper**: "Geometric Signatures of Reasoning: Correlates Without Causation"
**Venue**: ACL, EMNLP, or NeurIPS workshop

### Realistic Scenario (30% probability)

**H1 succeeds** (>65%):
- Strong within-domain signal

**H2 succeeds** (>55% transfer):
- Universal geometry exists!

**H4 partially succeeds** (1-3% improvement):
- Small but significant effect
- Proof of concept for steering

**Paper**: "Geometric Signatures of Reasoning Transfer Across Domains"
**Venue**: NeurIPS, ICML, ICLR (main conference)

### Pessimistic Scenario (10% probability)

**H1 fails** (<60%):
- Geometry doesn't distinguish correct/incorrect
- Confounds dominate

**Pivot**: 
- Analyze why (what does geometry actually capture?)
- Try alternative methods (early-layer probing, attention)

**Paper**: "Why Trajectory Geometry Doesn't Capture Reasoning Quality"
**Venue**: ICBINB workshop, negative results workshop

---

## 🎯 Bottom Line

**The trajectory geometry approach is a calculated risk**:

**Upside**: If it works, major contribution (novel method + causal evidence + practical application)

**Downside**: If it fails, we learn that geometry doesn't capture reasoning (still publishable, but lower impact)

**Mitigation**: 
- Strong controls (difficulty, length, format)
- Multiple methods (signatures, curvature, attention)
- Causal tests (steering)
- Honest reporting (don't overclaim)

**Recommendation**: **Proceed as planned**, but be ready to pivot if H2 fails.

---

## 📚 Key Papers to Guide Approach

**If trajectory approach works**:
- Cite: Hosseini & Fedorenko (2023), Kidger & Lyons (2021 - signatory)
- Frame: "We extend trajectory straightening to reasoning quality"

**If trajectory approach fails**:
- Cite: Afzal et al. (2025), Turpin et al. (2023)
- Frame: "Decision-before-reasoning limits trajectory-based analysis"

**Either way**:
- Cite: Hewitt & Liang (2019) for control tasks
- Cite: Marks & Tegmark (2023) for geometric structure of truth
- Cite: Turner et al. (2023) for activation steering

---

## 🔮 Future Directions (Beyond This Project)

**If trajectory geometry works**:
1. Apply to more models (Llama 3, GPT-4, Claude)
2. Test on more tasks (MMLU, ARC, reasoning benchmarks)
3. Improve steering (learned maps, RL-based optimization)
4. Investigate mechanistic causes (why does geometry transfer?)

**If trajectory geometry fails**:
1. Try alternative methods (early-layer probing, attention, circuits)
2. Characterize when/why geometry works vs fails
3. Develop better methods for reasoning quality detection

**Either way**:
- Contribute to understanding of how LLMs reason
- Provide tools for detecting/improving reasoning
- Advance interpretability research
