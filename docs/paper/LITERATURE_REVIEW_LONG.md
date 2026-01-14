# Literature Review: Geometric Signatures of Reasoning in LLMs (Extended)

**Research Question**: Can we learn the geometry of correct reasoning from verifiable domains (where we know the right answer) and use it on non-verifiable domains (where we don't)?

This extended review provides comprehensive coverage of the five core hypotheses (H1–H5), including detailed evidence, methodological considerations, and connections to broader literature.

---

## Executive Summary

The research plan proposes that **correct reasoning has a characteristic geometric signature** in the flow of activations through transformer layers. If such signatures exist and transfer across domains, they could enable:
1. Detection of bad reasoning on non-verifiable domains
2. Steering models toward correct reasoning trajectories  
3. Understanding what distinguishes correct from incorrect reasoning at the representation level

This review finds:
- **H1 (distinguishable trajectories)**: Strong supporting evidence, but significant confounds (difficulty, length, format)
- **H2 (domain-invariant signature)**: Mixed evidence; this is the critical test that determines project viability
- **H3 (non-verifiable detection)**: Weak evidence; requires H2 success and faces human judgment challenges
- **H4 (steering)**: Moderate evidence from related work, but high technical risk
- **H5 (lower curvature)**: Weak evidence; exploratory hypothesis with definition ambiguities

---

## Part I: Foundational Context

### 1.1 The Interpretability Landscape

**Probing and Linear Representations**

The research builds on a rich tradition of probing neural network representations to understand what information they encode.

**Hewitt & Liang (2019)** - "Designing and Interpreting Probes with Control Tasks"
- Introduced the concept of **control tasks** to validate probe findings
- Key insight: High probe accuracy doesn't guarantee the model "uses" that information
- Probes can learn to extract information that merely correlates with labels on training data
- **Selectivity** (performance on control task) matters as much as accuracy

**Pimentel et al. (2020)** - "Information-Theoretic Probing for Linguistic Structure"  
- Formalized probing through an information-theoretic lens
- Showed that arbitrarily powerful probes can extract spurious correlations
- Proposed measuring **mutual information** between representations and linguistic properties
- Critique: Even mutual information doesn't prove causal use of information

**Belinkov (2022)** - "Probing Classifiers: Promises, Shortcomings, and Advances" (Computational Linguistics)
- Comprehensive survey of probing methodology
- Documents that probes often identify features that correlate with but don't cause model behavior
- Recommends **intervention experiments** (like activation steering) to establish causality

**Implication for H1**: Simply showing that trajectories are distinguishable doesn't prove the model uses trajectory geometry for reasoning. Need to show the geometry is **causally relevant** (hence H4's importance).

### 1.2 Truth and Correctness in LLM Representations

**Marks & Tegmark (2023)** - "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations"
- Identified **linear truth directions** in LLM activation space
- Truth probes trained on one dataset (e.g., cities) transfer to others (e.g., facts about animals)
- Key finding: "Larger models have a more abstract notion of truth"
- Method: Train linear probes on true/false statement pairs, test cross-dataset generalization

**Strengths**:
- Demonstrates that truth has geometric structure
- Shows some degree of universality (cross-dataset transfer)
- Linear structure is computationally tractable

**Limitations**:
- Truth ≠ reasoning quality (a statement can be true by coincidence)
- Tested on factual statements, not reasoning processes
- Transfer within similar domains (all factual recall), not math→code→logic

**"Truth is Universal" (NeurIPS 2024)** - Extension of Marks & Tegmark
- Refined the truth representation to **two-dimensional subspace**:
  - One "general truth direction"
  - One "polarity-sensitive truth direction" (for negations)
- Found that classifiers trained only on affirmative statements fail on negated statements
- **Critical limitation**: Truth representation is context-dependent

**Implication for H2**: Even if "truth" has universal geometry, "correct reasoning" may not. Reasoning involves process, not just final truth value.

### 1.3 The Decision-Before-Reasoning Problem

**Afzal et al. (2025)** - "Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion"  
- Probing classifiers predict CoT success **even before a single token is generated**
- Achieves 70-80% accuracy using only the hidden state after processing the question
- Suggests the "decision" is encoded early; CoT is elaboration

**Experimental setup**:
- Collect hidden states at various points: after question, after first CoT token, after full CoT
- Train probes to predict whether the final answer will be correct
- Find that early states are highly predictive

**David (2025)** - "Temporal Predictors of Outcome in Reasoning Language Models"
- Models commit to answers early in the CoT generation process
- Analyzed when the "decision" crystallizes by tracking prediction confidence over time
- Found that by 20-30% through the CoT, the outcome is largely determined

**Implication for H1-H5**: 
- Trajectory geometry may capture **elaboration quality**, not decision-making
- Early layers may be more important than late layers (testable with H4's layer-wise analysis)
- This doesn't invalidate the research—elaboration quality may still transfer—but reframes the interpretation

---

## Part II: Hypothesis-by-Hypothesis Analysis

## H1: Correct vs Incorrect Reasoning Have Distinguishable Trajectories

**Full Hypothesis Statement**: On verifiable tasks, activation trajectories for problems solved correctly vs incorrectly should be geometrically distinguishable. A classifier trained on trajectory signatures should achieve >65% accuracy (significantly above 50% chance).

### Supporting Evidence (Detailed)

**Zhang et al. (2025)** - "Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification"

**Key contributions**:
- Trained probes on hidden states to predict correctness of intermediate reasoning steps
- Achieved 85-90% accuracy on MATH and GSM8K datasets
- Found that models encode correctness information **before** the answer is fully formulated
- Probes generalize across different problem types within the same domain

**Method**:
- Extract hidden states at each "chunk" (intermediate reasoning step)
- Label each chunk as correct/incorrect based on ground truth
- Train lightweight probe (linear or 2-layer MLP)
- Test on held-out problems

**Results**:
- Early prediction: Can predict final answer correctness from states 50% through the CoT
- Layer analysis: Middle layers (15-25 for 32-layer models) are most informative
- Model comparison: Stronger reasoners (o1, DeepSeek-R1) have clearer correctness signals

**Limitations**:
- Tested only on math reasoning
- Correctness labels are binary (right/wrong answer), not reasoning quality
- May be detecting problem difficulty rather than reasoning process

**Sun et al. (2025)** - "Probing for Arithmetic Errors in Language Models"
- Focused specifically on arithmetic errors in multi-step reasoning
- Found that error-prone steps have distinct geometric signatures
- Probes can identify where in a calculation the model will make a mistake

**Azaria & Mitchell (2023)** - "The Internal State of an LLM Knows When It's Lying"
- Broader than reasoning: tested on factual statements
- LLM hidden states encode truthfulness even when the model generates false statements
- Trained classifiers outperform probability-based methods (model's own confidence)

**Method**:
- Generate true and false statements on various topics
- Extract hidden states from multiple layers
- Train classifier to predict truthfulness
- Compare to baseline: model's output probability

**Results**:
- Hidden state classifiers: 75-85% accuracy
- Probability baseline: 60-70% accuracy  
- Best performance: middle-to-late layers

**Implication**: Internal representations encode correctness signals that aren't fully reflected in output probabilities.

### Critical Evidence and Confounds (Detailed)

**Confound 1: Problem Difficulty**

Easy problems may have:
- Shorter trajectories (fewer reasoning steps)
- More direct paths (less exploration)
- Higher activation magnitudes (more confident)

Hard problems may have:
- Longer trajectories
- More "wandering" (exploring solution space)
- Lower activation magnitudes (uncertainty)

If the model gets easy problems right and hard problems wrong, a classifier could learn to distinguish **difficulty** rather than **reasoning quality**.

**Mitigation strategies**:
1. **Stratify by difficulty**: Use problem length, human solve time, or model perplexity as difficulty proxies
2. **Matched comparisons**: For each correct solution, find an incorrect solution of similar difficulty
3. **Within-difficulty classification**: Train separate classifiers for easy/medium/hard problems

**Confound 2: Output Length**

Correct solutions may systematically differ in length from incorrect ones:
- Correct: Complete, well-structured reasoning
- Incorrect: May be truncated (model gives up) or verbose (model is confused)

Trajectory geometry may simply reflect sequence length.

**Mitigation**: Control for output length explicitly; compare correct vs incorrect solutions of similar token count.

**Confound 3: Surface Format**

Different tasks have different output formats:
- Math: "#### 42" format
- Code: Function definitions, indentation
- Logic: "The answer is A" format

Geometry may distinguish **format** rather than reasoning.

**Mitigation**: Test within-format transfer (e.g., GSM8K → MATH, both math but different difficulty).

**Construct Validity: "Correct" ≠ "Good Reasoning"**

A model can be correct via:
- **Guessing**: Random chance on multiple choice
- **Memorization**: Recalling training examples
- **Spurious correlations**: Pattern matching without understanding
- **Partial reasoning**: Sound reasoning with a lucky computational error cancellation

Conversely, a model can be incorrect via:
- **Sound reasoning + arithmetic error**: All logic correct, one calculation wrong
- **Correct method, wrong execution**: Right approach, implementation bug

**Implication**: H1 may succeed (geometry distinguishes correct/incorrect) while still not capturing "reasoning quality" as intended.

**Recommendation**: Supplement binary correctness with **process supervision** (label intermediate steps) or **explanation quality** ratings.

---

## H2: The Correct Reasoning Signature is Domain-Invariant

**Full Hypothesis Statement**: If correct reasoning has universal geometry, a classifier trained on math (GSM8K correct/incorrect) should work on code (HumanEval) and logic (LogiQA) with >55% transfer accuracy (zero-shot, no training on target domain).

**This is the critical test**. If H2 fails, the entire "universal geometry of reasoning" premise fails.

### Supporting Evidence (Detailed)

**Marks & Tegmark (2023)** - "The Geometry of Truth"

**Cross-dataset transfer experiments**:
- Train truth probe on cities dataset ("Paris is the capital of France" vs "Paris is the capital of Germany")
- Test on animals dataset ("Dogs are mammals" vs "Dogs are reptiles")
- Achieve 65-75% transfer accuracy (vs 50% chance)

**Scale effects**:
- Smaller models (GPT-2): 55-60% transfer
- Medium models (GPT-J 6B): 65-70% transfer  
- Large models (GPT-3 175B): 70-75% transfer
- **Interpretation**: Larger models develop more abstract, transferable truth representations

**Layer analysis**:
- Early layers (0-10): Poor transfer (55%)
- Middle layers (10-25): Best transfer (70-75%)
- Late layers (25-40): Moderate transfer (65%)
- **Interpretation**: Middle layers encode abstract semantic content

**Limitations for H2**:
- All tested domains are **factual recall** (cities, animals, historical events)
- No test of **reasoning process** transfer (math → code → logic)
- Truth is binary (true/false), reasoning has intermediate steps

**Hosseini & Fedorenko (2023)** - "Large Language Models Implicitly Learn to Straighten Neural Sentence Trajectories"

**Key finding**: Transformers learn to produce progressively straighter activation trajectories through layers, and this "straightening" is **domain-general**.

**Method**:
- Measure trajectory curvature across layers for various tasks (language modeling, QA, reasoning)
- Compute: curvature = angle between consecutive layer transitions
- Compare across domains and model scales

**Results**:
- All successful processing shows trajectory straightening
- Effect is stronger in larger models
- Straightening occurs regardless of domain (language, math, code)

**Implication for H2**: If straightening is universal, and correct reasoning produces straighter paths (H5), then correct reasoning geometry may transfer.

**Neural Collapse Theory** (Papyan et al., 2020; Galanti et al., 2022)

**Core prediction**: Classification tasks converge to maximally simple, symmetric geometry:
- Features collapse to vertices of a **Simplex Equiangular Tight Frame**
- Within-class variability → 0
- Between-class separation → maximal
- Classifier weights align with class means

**Galanti et al. (2022)** - "On the Implicit Bias of Initialization: How Infinitely Wide and Deep Networks Learn"
- When neural collapse emerges, **few-shot transfer** becomes trivial
- A linear classifier on new classes requires very few samples
- **Mechanism**: Collapsed features are maximally separated, making linear classification easy

**Application to H2**:
- If verification is binary classification (correct/incorrect)
- And if neural collapse occurs in LLM reasoning
- Then correct/incorrect features should be maximally separated
- And this separation should be **universal** (same geometry across domains)

**Critical caveat**: Neural collapse theory assumes:
1. Balanced classes (equal correct/incorrect samples)
2. Sufficient training to convergence
3. Classification as the terminal objective

LLM reasoning may violate these assumptions.

### Critical Evidence (Detailed)

**Turpin et al. (2023)** - "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting"

**Experimental design**:
- Introduce biasing features in prompts (e.g., "(a) is the correct answer")
- Ask model to generate CoT reasoning
- Check if CoT mentions the biasing feature

**Results**:
- Models generate plausible reasoning that ignores the bias
- Final answers are influenced by the bias (showing the model "saw" it)
- CoT explanations are **unfaithful**: they don't reflect the actual decision process

**Implication for H2**:
- Trajectory geometry may reflect **post-hoc rationalization**, not true reasoning
- The "correct reasoning" signature may be "plausible explanation" signature
- This could be domain-specific (math explanations differ from code explanations)

**Quantitative findings**:
- 75-85% of biased answers have CoT that doesn't mention the bias
- Effect is stronger in larger models (more sophisticated rationalization)
- Persists across domains (math, commonsense, factual QA)

**Ley et al. (2024)** - "On the Hardness of Faithful Chain-of-Thought Reasoning in Large Language Models"

**Tested interventions to improve faithfulness**:
1. **In-context learning**: Provide examples of faithful reasoning
2. **Fine-tuning**: Train on datasets with faithful CoT
3. **Activation editing**: Modify hidden states to reduce bias

**Results**: All interventions **fail to generalize** across diverse benchmarks.
- ICL: Works on training distribution, fails on new domains
- Fine-tuning: Overfits to specific task format
- Activation editing: Breaks model on out-of-distribution inputs

**Implication**: If faithfulness interventions don't transfer, why would geometric signatures of "reasoning quality" transfer?

**Counter-argument**: Faithfulness ≠ reasoning quality. A model can have good reasoning structure while being unfaithful about its process.

**Domain-Specific Reasoning Patterns**

**Math reasoning**:
- Symbolic manipulation (algebraic steps)
- Numerical computation (arithmetic)
- Formula application
- **Geometric signature**: May involve discrete state transitions (equation → simplified equation)

**Code reasoning**:
- Control flow analysis (if/else, loops)
- Type checking and inference
- API/library knowledge
- **Geometric signature**: May involve hierarchical structure (function calls, scope)

**Logic reasoning**:
- Formal inference rules (modus ponens, etc.)
- Quantifier handling (∀, ∃)
- Contradiction detection
- **Geometric signature**: May involve constraint satisfaction

**Hypothesis**: These different reasoning patterns may have **fundamentally different geometric signatures**, preventing transfer.

**Empirical evidence**: Limited direct evidence, but:
- **Chu et al. (2025)** - "SFT Memorizes, RL Generalizes": Different training paradigms produce different reasoning strategies
- **Jin et al. (2025)** - "The Path Not Taken": RLVR learns different representational structure than SFT
- These differences may be domain-specific

### Testable Predictions for H2

**Strong transfer (supports H2)**:
- Math → Code: >60% accuracy
- Math → Logic: >60% accuracy
- Code → Math: >60% accuracy
- **Interpretation**: Universal reasoning signature exists

**Weak transfer (challenges H2)**:
- All transfers: 52-58% accuracy (barely above chance)
- **Interpretation**: Some weak signal, but mostly domain-specific

**No transfer (falsifies H2)**:
- All transfers: ≤52% accuracy
- **Interpretation**: No universal reasoning signature; geometry is domain-specific

**Asymmetric transfer (interesting finding)**:
- Math → Code: 65% (good)
- Code → Math: 53% (poor)
- **Interpretation**: Math reasoning may be more general, or code reasoning more specialized

---

## H3: Detector Works on Non-Verifiable Domains

**Full Hypothesis Statement**: A detector trained on verifiable domains can predict reasoning quality on non-verifiable tasks (philosophy, ethics, strategy), validated by correlation with human judgments (r > 0.25).

**Dependency**: Requires H2 success. If geometry doesn't transfer across verifiable domains, it won't transfer to non-verifiable ones.

### Supporting Evidence

**Farquhar et al. (2024)** - "Detecting Hallucinations in Large Language Models Using Semantic Entropy" (Nature)

**Method**: Semantic entropy—measure uncertainty by clustering sampled outputs
- Generate multiple outputs for the same input
- Cluster outputs by semantic meaning (using NLI model)
- Compute entropy over cluster distribution
- High entropy = high uncertainty = likely hallucination

**Results**:
- Detects confabulations with 75-85% AUROC
- Correlates with human judgments (r = 0.6-0.7)
- Works across domains (QA, biography generation, medical advice)

**Relevance to H3**: Shows that **internal uncertainty signals** can be extracted and validated against human judgment, even without ground truth.

**Limitation**: Semantic entropy requires sampling (expensive); geometric detector would be cheaper.

**Kossen et al. (2024)** - "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs"

**Contribution**: Train probes on hidden states to predict semantic entropy
- Cheaper than sampling (single forward pass)
- Achieves 70-80% of sampling-based performance
- Generalizes across tasks with minimal fine-tuning

**Relevance**: Demonstrates that **geometric features** (hidden states) can predict uncertainty/quality.

### Critical Evidence

**Human Judgment is Noisy and Biased**

**Lin et al. (2021)** - "TruthfulQA: Measuring How Models Mimic Human Falsehoods"
- Models often generate false statements that humans rate as plausible
- Human raters fail to catch falsehoods 30-40% of the time
- Correlation with human ratings may reflect **fluency** rather than truth

**Implication**: If humans can't reliably judge reasoning quality, how can we validate the detector?

**Min et al. (2023)** - "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation"
- Even on partially-verifiable long-form generation, automated metrics struggle
- Human evaluation is expensive (\$0.50-\$2 per sample) and inconsistent
- Inter-annotator agreement: κ = 0.4-0.6 (moderate)

**Non-Verifiable Domains Have Different Dynamics**

**Philosophical reasoning**:
- No single "correct" answer
- Multiple valid perspectives
- Quality = coherence, depth, consideration of counterarguments

**Ethical reasoning**:
- Value-laden (different moral frameworks)
- Context-dependent
- Quality = consistency, consideration of stakeholders

**Strategic reasoning**:
- Outcome-dependent (can't verify until executed)
- Counterfactual ("what if" scenarios)
- Quality = consideration of alternatives, risk assessment

**Hypothesis**: The geometry of "good reasoning" on these tasks may be fundamentally different from "correct answers" on verifiable tasks.

---

## H4: Trajectories Can Be Steered Toward Correct Reasoning

**Full Hypothesis Statement**: Projecting activations onto a "correct reasoning" manifold during inference improves accuracy on held-out verifiable problems by >2%.

**This is the causal test**: If geometry matters, intervening on it should change behavior.

### Supporting Evidence

**Turner et al. (2023)** - "Steering Language Models With Activation Engineering"

**Method**: Activation addition (ActAdd)
1. Collect activations for contrasting prompts (e.g., "Be honest" vs "Be deceptive")
2. Compute steering vector: mean(honest) - mean(deceptive)
3. Add steering vector to activations during inference
4. Measure behavior change

**Results**:
- Honesty steering: Increases truthful responses by 15-20%
- Toxicity steering: Reduces toxic outputs by 30-40%
- Works with as few as 2 contrasting examples
- Effect is dose-dependent (stronger vector = stronger effect)

**Relevance to H4**: Demonstrates that **linear interventions** in activation space can causally affect behavior.

**Key insight**: Steering vectors are **task-agnostic**—a honesty vector trained on one domain works on others.

**Meng et al. (2022)** - "Locating and Editing Factual Associations in GPT" (ROME)

**Method**: Causal tracing + rank-one model editing
- Identify which layers store factual associations
- Modify weights at those layers to change facts
- Measure: Does the edit change outputs? Does it preserve other knowledge?

**Results**:
- Factual associations are stored in mid-layer MLPs (layers 10-20 for GPT-J)
- Editing these layers changes outputs predictably
- Edits generalize to paraphrases

**Relevance**: Shows that **targeted interventions** at specific layers can modify behavior without breaking the model.

**Belrose et al. (2023)** - "Eliciting Latent Predictions from Transformers with the Tuned Lens"

**Method**: Train affine probes to decode predictions from intermediate layers
- For each layer, train probe: hidden_state → vocabulary distribution
- Measure: How early can we predict the final output?

**Results**:
- Correct predictions are often latent in middle layers (15-25)
- Tuned lens reveals "what the model is thinking" before the final layer
- Suggests that steering toward these latent predictions could improve accuracy

### Critical Evidence

**Steering May Break the Model**

**Razzhigaev et al. (2024)** - "Your Transformer is Secretly Linear"
- Transformers exhibit **local linearity** (small perturbations have linear effects)
- But **global nonlinearity** (large perturbations can break the model)
- Naive projection may push activations out of the valid manifold

**Implication**: Steering must be carefully calibrated; too strong = model collapse.

**Technical Challenges**

1. **Which layers to steer?**
   - Early layers: May disrupt feature extraction
   - Middle layers: Most promising (where decisions form)
   - Late layers: May be too late to change reasoning

2. **How strongly to steer?**
   - Too weak: No effect
   - Too strong: Model breaks
   - Need grid search over steering strength α ∈ [0.1, 0.3, 0.5, 0.7, 1.0]

3. **Regression to the mean**
   - Steering toward "average correct trajectory" may just make problems easier
   - May not improve on hard problems (where we need it most)

**Zhang & Zhou (2024)** - "Understanding Transformer Architecture through Continuous Dynamics: A Partial Differential Equation Perspective"
- Transformers have complex, nonlinear dynamics
- Simple linear steering may be insufficient for complex reasoning
- May need learned, nonlinear steering functions

---

## H5: Correct Reasoning Has Lower Curvature

**Full Hypothesis Statement**: Correct reasoning follows straighter paths through activation space (lower curvature); incorrect reasoning wanders (higher curvature).

### Supporting Evidence

**Hosseini & Fedorenko (2023)** - "Large Language Models Implicitly Learn to Straighten Neural Sentence Trajectories"

**Quantitative findings**:
- Curvature decreases by 40-60% from early to late layers
- Larger models show stronger straightening
- Straightening correlates with task performance

**Interpretation**: Successful processing = confident, direct computation.

**Intuition from Optimization**
- Correct reasoning: Model "knows where it's going" → direct path
- Incorrect reasoning: Model explores, backtracks → curved path
- Analogy: Gradient descent on smooth vs rugged loss landscape

### Critical Evidence

**Curvature May Reflect Difficulty, Not Correctness**

**Scenario 1**: Easy problem, correct solution
- Short, direct path → low curvature

**Scenario 2**: Easy problem, incorrect solution (model guesses)
- Also short, direct path → low curvature

**Scenario 3**: Hard problem, correct solution (model explores)
- Long, exploratory path → high curvature

**Scenario 4**: Hard problem, incorrect solution
- Long, confused path → high curvature

**Implication**: Curvature may distinguish difficulty (scenarios 1-2 vs 3-4) rather than correctness (1,3 vs 2,4).

**Definition Ambiguity**

Curvature can be defined multiple ways:
1. **Euclidean**: Angle between consecutive layer transitions in ambient space
2. **Geodesic**: Curvature along the manifold (requires manifold estimation)
3. **Discrete**: Sum of turning angles
4. **Continuous**: Second derivative of trajectory

Different definitions may yield different conclusions.

**Jin et al. (2025)** - "The Path Not Taken: RLVR Provably Learns Off the Principals"

**Finding**: RLVR training operates in low-curvature subspaces
- RLVR updates avoid principal directions (high curvature)
- Instead, updates occur in low-curvature, spectrum-preserving subspaces

**Implication**: Lower curvature in RLVR models is a **training artifact**, not a property of "correct reasoning."

**Counter-argument**: If RLVR produces better reasoning AND lower curvature, maybe the training artifact IS the mechanism.

---

## Part III: Methodological Recommendations

### Sample Size and Statistical Power

**Current plan**: 500 samples per task
- Need balanced correct/incorrect split
- If model accuracy is 70%, expect ~350 correct, ~150 incorrect
- For cross-validation (5-fold), each fold has ~30 incorrect samples
- **Concern**: May be underpowered for H2 (cross-domain transfer)

**Recommendation**: 
- Ensure at least 100 samples per class (correct/incorrect) per task
- If model is too accurate, sample harder problems to balance classes

### Baseline Comparisons

**Kadavath et al. (2022)** - "Language Models (Mostly) Know What They Know"
- Model's own probability estimates predict correctness well
- Calibration baseline: Use P(correct | model confidence)

**Wang et al. (2022)** - "Self-Consistency Improves Chain of Thought Reasoning"
- Sample multiple outputs, take majority vote
- Variance across samples predicts correctness

**Recommendation**: Compare trajectory geometry against:
1. Model confidence (logit-based)
2. Semantic entropy (sampling-based)
3. Self-consistency (majority vote)
4. Output length (simple baseline)

Geometry should **add value** beyond these simpler signals.

### Control Tasks (Hewitt & Liang, 2019)

**Proposed control**: Train classifier on **random labels**
- Shuffle correct/incorrect labels
- Train classifier on trajectory signatures
- Measure accuracy

**Interpretation**:
- If control accuracy ≈ 50%: Geometry encodes meaningful signal
- If control accuracy > 55%: Classifier may be learning spurious features

---

## Part IV: Synthesis and Strategic Assessment

### What the Literature Tells Us

**Strong consensus**:
1. Hidden states encode information about correctness (H1 likely succeeds)
2. Truth has some geometric structure (Marks & Tegmark)
3. Activation steering can modify behavior (H4 is technically feasible)

**Weak consensus**:
1. Whether correctness geometry transfers across domains (H2 is uncertain)
2. Whether geometry reflects reasoning vs post-hoc rationalization
3. Whether curvature is meaningful (H5 is speculative)

**No consensus**:
1. Whether non-verifiable reasoning has similar geometry (H3 is unexplored)

### The Critical Unknown: H2

H2 is the **linchpin**. If it fails:
- H3 becomes irrelevant (no transfer to non-verifiable domains)
- H4 becomes domain-specific (steering works within-domain only)
- H5 becomes a curiosity (curvature is domain-specific)

If H2 succeeds:
- Opens path to universal reasoning quality detector
- Enables cross-domain steering
- Provides evidence for abstract reasoning representations

### Publishable Outcomes

**Positive results (H1 + H2 succeed)**:
- "Geometric Signatures of Reasoning Transfer Across Domains"
- Venue: NeurIPS, ICML, ICLR (main conference)
- Impact: High (novel finding, practical applications)

**Mixed results (H1 succeeds, H2 fails)**:
- "Domain-Specific Geometric Signatures of Reasoning in LLMs"
- Venue: ACL, EMNLP, NeurIPS workshop
- Impact: Medium (characterizes limitations, informs future work)

**Negative results (H1 fails)**:
- "Why Trajectory Geometry Doesn't Capture Reasoning Quality"
- Venue: ICBINB workshop, ICLR workshop
- Impact: Low-Medium (negative result, but informative)

### Recommended Pivots if H2 Fails

**Pivot 1**: Characterize **what differs** across domains
- Use H1 classifiers to identify domain-specific vs universal features
- Decompose geometry into shared + domain-specific components

**Pivot 2**: Test **hierarchical transfer**
- Math → Math (different dataset): Should work well
- Math → Physics: Closer domains
- Math → Code: Farther domains
- Measure transfer as function of domain distance

**Pivot 3**: Focus on **within-domain** applications
- Even if geometry doesn't transfer, within-domain detection is useful
- Can still improve reasoning via domain-specific steering

---

## Conclusion

The proposed research sits at the intersection of **interpretability**, **reasoning**, and **transfer learning**. The literature provides:

**Strong foundations**:
- Hidden states encode correctness (multiple papers)
- Geometric methods are tractable (path signatures, subspace analysis)
- Interventions are possible (activation steering)

**Open questions**:
- Does correctness geometry transfer? (H2)
- Is it reasoning or rationalization? (Decision-before-reasoning problem)
- Can we steer effectively? (H4)

**Key risks**:
- Confounds (difficulty, length, format)
- Small sample size for transfer tests
- Technical challenges in steering

**Recommendations**:
1. **Prioritize H2**: This is the critical test
2. **Add strong controls**: Difficulty stratification, baseline comparisons
3. **Prepare for negative results**: Have pivot strategies ready
4. **Focus on causal tests**: H4 (steering) is more valuable than H5 (curvature)

The research is **high risk, high reward**. Success would be a significant contribution; failure would still yield publishable insights about the limits of geometric interpretability.
