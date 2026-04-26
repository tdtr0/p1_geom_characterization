# Literature Review: Geometric Signatures of Reasoning in LLMs

**Research Question**: Can we learn the geometry of correct reasoning from verifiable domains and apply it to non-verifiable domains?

This review covers the five core hypotheses (H1–H5) from the research plan, presenting both supporting and critical evidence for each.

---

## H1: Correct vs Incorrect Reasoning Have Distinguishable Trajectories

**Hypothesis**: On verifiable tasks, activation trajectories for correctly vs incorrectly solved problems should be geometrically distinguishable.

### Supporting Evidence

**Zhang et al. (2025)** - "Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification"
- Probes trained on hidden states can predict correctness of intermediate reasoning steps with high accuracy
- Models encode information about answer correctness *before* the answer is fully formulated
- Suggests internal representations differ systematically between correct and incorrect reasoning paths

**Afzal et al. (2025)** - "Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion"
- Lightweight probes predict CoT success even before a single token is generated
- Achieves strong classification performance using only early hidden states
- Indicates that "decision" is encoded early in the forward pass, before elaboration

**Azaria & Mitchell (2023)** - "The Internal State of an LLM Knows When It's Lying"
- LLM hidden states can be used to detect truthfulness of statements
- Trained classifiers outperform probability-based methods for detecting false outputs
- Demonstrates that internal representations encode correctness signals

### Critical Evidence

**Construct validity concern**: "Correct" ≠ "good reasoning"
- A model can be correct by guessing, memorization, or spurious correlations
- Conversely, a model can follow sound reasoning but make a computational error
- **Hewitt & Liang (2019)** - "Designing and Interpreting Probes with Control Tasks": Probes may learn task-specific shortcuts rather than the underlying concept

**Confound: Problem difficulty**
- Easy problems may have shorter, more direct trajectories regardless of reasoning quality
- Hard problems may exhibit more "wandering" even when reasoning is sound
- Geometry may distinguish difficulty rather than correctness

**Pimentel et al. (2020)** - "Information-Theoretic Probing for Linguistic Structure"
- Arbitrarily powerful probes can extract information that merely correlates with labels on training data
- High probe accuracy doesn't guarantee the model "uses" that information for its decisions

---

## H2: The Correct Reasoning Signature is Domain-Invariant

**Hypothesis**: A classifier trained to distinguish correct vs incorrect trajectories on math should transfer to code and logic tasks.

### Supporting Evidence

**Marks & Tegmark (2023)** - "The Geometry of Truth"
- Linear truth directions in LLM representations generalize across different datasets
- Truth probes trained on one domain transfer to others with above-chance accuracy
- Larger models show more abstract, transferable truth representations

**Hosseini & Fedorenko (2023)** - "Large Language Models Implicitly Learn to Straighten Neural Sentence Trajectories"
- Transformers learn to produce progressively straighter activation trajectories through layers
- This "straightening" is domain-general and emerges with scale
- Suggests a universal geometric property of successful processing

**Neural Collapse theory** (Papyan et al., 2020)
- Classification tasks converge to maximally simple, symmetric geometry (Simplex Equiangular Tight Frame)
- If verification is binary classification (correct/incorrect), representations should collapse to low-dimensional, transferable structure
- **Galanti et al. (2022)**: Neural collapse enables few-shot transfer with linear classifiers

### Critical Evidence

**Turpin et al. (2023)** - "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting"
- CoT explanations can be systematically unfaithful to the model's actual decision process
- Models generate plausible reasoning for biased answers without revealing the bias
- Trajectory geometry may reflect *post-hoc rationalization* rather than true reasoning

**Ley et al. (2024)** - "On the Hardness of Faithful Chain-of-Thought Reasoning in Large Language Models"
- Interventions to improve CoT faithfulness (ICL, fine-tuning, activation editing) fail to generalize across benchmarks
- If faithfulness doesn't transfer, why would geometric signatures of "reasoning quality" transfer?

**Domain-specific reasoning patterns**
- Math reasoning may involve symbolic manipulation (algebraic steps)
- Code reasoning may involve control flow and type checking
- Logic reasoning may involve formal inference rules
- These may have fundamentally different geometric signatures

---

## H3: Detector Works on Non-Verifiable Domains

**Hypothesis**: A detector trained on verifiable domains can predict reasoning quality on non-verifiable tasks, validated by human judgment.

### Supporting Evidence

**Farquhar et al. (2024)** - "Detecting Hallucinations in Large Language Models Using Semantic Entropy"
- Semantic entropy (clustering of sampled outputs) detects confabulations without ground truth
- Correlates with human judgments of factual accuracy
- Demonstrates that internal uncertainty signals can be extracted and validated

**Kossen et al. (2024)** - "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs"
- Probes on hidden states detect hallucinations more cheaply than sampling-based methods
- Generalizes across tasks with minimal fine-tuning
- Shows promise for detecting unreliable outputs without verification

### Critical Evidence

**Human judgment is noisy and biased**
- Humans often rate coherent, confident outputs highly even when factually wrong
- **Lin et al. (2021)** - "TruthfulQA": Models mimic human falsehoods; human raters may not catch this
- Correlation with human ratings may reflect fluency/confidence rather than reasoning quality

**Non-verifiable domains have different dynamics**
- Philosophical reasoning may not have a "correct" answer
- Ethical reasoning involves values, not verifiable facts
- The geometry of "good reasoning" may be fundamentally different from "correct answers"

**Min et al. (2023)** - "FActScore: Fine-grained Atomic Evaluation of Factual Precision"
- Even on partially-verifiable long-form generation, automated metrics struggle
- Human evaluation remains expensive and inconsistent

---

## H4: Trajectories Can Be Steered Toward Correct Reasoning

**Hypothesis**: Projecting activations onto a "correct reasoning" manifold during inference improves accuracy on held-out problems.

### Supporting Evidence

**Turner et al. (2023)** - "Steering Language Models With Activation Engineering"
- Adding steering vectors to activations can control model behavior (e.g., increase honesty, reduce toxicity)
- Activation addition is lightweight and works with minimal data
- Demonstrates that linear directions in activation space causally affect outputs

**Meng et al. (2022)** - "Locating and Editing Factual Associations in GPT" (ROME)
- Causal tracing identifies specific layers where factual associations are stored
- Editing activations at those layers changes model outputs predictably
- Shows that targeted interventions can modify behavior

**Belrose et al. (2023)** - "Eliciting Latent Predictions from Transformers with the Tuned Lens"
- Affine probes can decode predictions from intermediate layers
- Suggests that "correct" outputs are latent in earlier representations
- Steering toward these latent predictions could improve accuracy

### Critical Evidence

**Steering may break the model**
- Activations are jointly optimized; modifying one component may disrupt others
- **Razzhigaev et al. (2024)** - "Your Transformer is Secretly Linear": While transformers exhibit local linearity, global interventions may fail
- Naive projection may collapse diversity or introduce artifacts

**Technical challenges**
- Which layers to steer? How strongly? Many hyperparameters to tune
- Steering might just regress to the mean (push toward average = easier problems)
- May only work for specific task pairs or model architectures

**Zhang & Zhou (2024)** - "Understanding Transformer Architecture through Continuous Dynamics: A Partial Differential Equation Perspective"
- Transformers have complex, nonlinear dynamics
- Simple linear steering may be insufficient for complex reasoning tasks

---

## H5: Correct Reasoning Has Lower Curvature

**Hypothesis**: Correct reasoning follows straighter paths through activation space; incorrect reasoning wanders.

### Supporting Evidence

**Hosseini & Fedorenko (2023)** - "Large Language Models Implicitly Learn to Straighten Neural Sentence Trajectories"
- Successful language processing produces progressively straighter trajectories
- Curvature decreases through layers as the model refines predictions
- Lower curvature correlates with better performance

**Intuition from optimization**
- Correct reasoning may reflect confident, direct computation ("the model knows where it's going")
- Incorrect reasoning may involve exploration, backtracking, or uncertainty
- Straighter paths = more efficient computation

### Critical Evidence

**Curvature may reflect difficulty, not correctness**
- Hard problems may require exploration even when reasoning is sound
- Easy problems may have straight paths even with poor reasoning (e.g., pattern matching)
- Curvature could be a proxy for problem complexity rather than reasoning quality

**Definition ambiguity**
- Curvature depends on the metric used (Euclidean, geodesic, etc.)
- Different choices may yield different conclusions
- No consensus on the "right" way to measure trajectory curvature in high-dimensional spaces

**Jin et al. (2025)** - "The Path Not Taken: RLVR Provably Learns Off the Principals"
- RLVR training preserves spectral structure but operates in low-curvature subspaces
- This is a training artifact, not necessarily a property of "correct reasoning"
- Curvature differences may reflect training paradigm rather than reasoning quality

---

## Cross-Cutting Concerns

### The Decision-Before-Reasoning Problem

**David (2025)** - "Temporal Predictors of Outcome in Reasoning Language Models"
- Models commit to answers early in the CoT generation process
- Later tokens are elaboration, not decision-making
- Trajectory geometry may capture elaboration quality, not reasoning process

**Implication**: H1–H5 may succeed while still not isolating the reasoning *process* itself.

### Sample Size and Statistical Power

- Most cited studies use large datasets (thousands of samples)
- The research plan proposes 500 samples per task
- Cross-domain transfer (H2) with limited data may yield noisy results
- **Recommendation**: Ensure sufficient samples in both correct/incorrect classes for robust classification

### Baseline Comparisons

**Kadavath et al. (2022)** - "Language Models (Mostly) Know What They Know"
- Simple calibration (model's own probability estimates) predicts correctness well
- Geometry-based methods should outperform or complement this baseline

**Wang et al. (2022)** - "Self-Consistency Improves Chain of Thought Reasoning"
- Sampling multiple outputs and taking majority vote improves accuracy
- Trajectory variance across samples may be a simpler signal than geometry

---

## Summary Assessment

| Hypothesis | Supporting Strength | Critical Concerns | Likelihood of Success |
|------------|-------------------|-------------------|---------------------|
| **H1** | Strong (multiple papers show hidden states encode correctness) | Confounds (difficulty, length, format) | **High** (with controls) |
| **H2** | Moderate (truth directions transfer; neural collapse theory) | Domain-specific reasoning; unfaithful CoT | **Medium** (critical test) |
| **H3** | Weak (limited evidence for non-verifiable transfer) | Human judgment noise; different dynamics | **Low** (requires H2 success) |
| **H4** | Moderate (activation steering works in other contexts) | Technical challenges; may break model | **Medium** (high risk, high reward) |
| **H5** | Weak (one main supporting paper; intuitive but unproven) | Confounds with difficulty; definition ambiguity | **Low** (exploratory) |

**Key Recommendations**:
1. **Control for confounds**: Stratify by difficulty, length, and format
2. **Use strong baselines**: Compare against calibration, self-consistency, semantic entropy
3. **H2 is the critical test**: If it fails, the entire "universal geometry" premise fails
4. **Prepare for negative results**: Even failures are publishable if well-characterized

