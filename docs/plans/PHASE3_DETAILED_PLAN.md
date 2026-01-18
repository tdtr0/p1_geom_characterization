# Phase 3: Cross-Domain Transfer and Dynamical Analysis (H2)

**Status**: ⏳ Pending (depends on Phase 2 success)
**Duration**: Weeks 5-8 (4 weeks)
**Objective**: Test H2 - whether geometric/dynamical signatures of correct solutions transfer across domains

---

## Overview

Phase 3 tests whether correct solutions share dynamical signatures across domains. We adopt an **interpolation-centric view** (Allen-Zhu & Li, 2024): we're not detecting "reasoning" as a cognitive category, but characterizing the geometry of successful task completion.

**Reframed question**: Do correct solutions across math, code, and logic share dynamical properties (attractor dynamics, stability, curvature profiles)?

**What would success mean**: Correct solutions traverse similar manifold regions regardless of domain.

**What would failure mean**: Task performance is geometrically domain-specific (still valuable to characterize).

**Success criterion**: >55% transfer accuracy (above 50% chance)

---

## Theoretical Framework (New)

### Why Dynamical Systems Analysis?

Recent work suggests:

1. **Everything is interpolation** (Allen-Zhu): No "reasoning mode" vs "recall mode"
2. **Attractor dynamics matter** (Ren & Liu, 2026): Correct solutions find right attractors; incorrect get trapped
3. **Curvature separates mechanisms** (Merullo et al., 2025): High-curvature = general; low-curvature = memorization
4. **Belief state geometry** (Shai et al., 2024): Residual stream represents belief states
5. **Menger curvature captures logic** (Zhou et al., Oct 2025): Curvature profiles reveal logical structure beyond surface semantics

Our analysis suite connects directly to these insights:

| Analysis | Connects To | Hypothesis |
|----------|-------------|------------|
| Vector field decomposition | Interpolation structure | Correct = more potential (direct) flow |
| Lyapunov exponents | Attractor dynamics | Correct = stable convergence |
| Menger curvature | Zhou et al. (logic vs semantics) | Curvature captures correctness better than position |
| Curvature regime activation | Memorization vs generalization | Correct = more high-curvature |
| Path signatures | Trajectory shape | Correct = more structured paths |

---

## Experimental Design

### Transfer Matrix

We test all pairwise domain transfers:

| Train Domain | Test Domain | Expected Difficulty |
|--------------|-------------|-------------------|
| GSM8K (math) | HumanEval (code) | Hard (different reasoning types) |
| GSM8K (math) | LogiQA (logic) | Medium (both symbolic) |
| HumanEval (code) | GSM8K (math) | Hard (different reasoning types) |
| HumanEval (code) | LogiQA (logic) | Medium (both structured) |
| LogiQA (logic) | GSM8K (math) | Medium (both symbolic) |
| LogiQA (logic) | HumanEval (code) | Hard (different reasoning types) |

**Total**: 6 transfer tests per model = 24 tests (4 models × 6 transfers)

### Within-Format Control

To distinguish domain transfer from format transfer, also test:

| Train | Test | Purpose |
|-------|------|---------|
| GSM8K | MATH | Within-format (both math, different difficulty) |
| HumanEval | MBPP | Within-format (both code, different style) |

**Prediction**: Within-format should work better than cross-format if format is a confound.

---

## Week-by-Week Breakdown

### Week 5: Cross-Domain Classification

**Day 1-2: Prepare Transfer Pipeline**

Create `scripts/test_h2_transfer.py`:

```python
def test_transfer(train_model, train_task, test_task):
    # Load signatures
    train_sigs = load_signatures(train_model, train_task)
    test_sigs = load_signatures(train_model, test_task)
    
    # Load labels
    train_labels = load_labels(train_model, train_task)
    test_labels = load_labels(train_model, test_task)
    
    # Train classifier on source domain
    clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    clf.fit(train_sigs, train_labels)
    
    # Test on target domain (zero-shot)
    test_accuracy = clf.score(test_sigs, test_labels)
    
    return {
        'train_model': train_model,
        'train_task': train_task,
        'test_task': test_task,
        'accuracy': test_accuracy,
        'n_test_samples': len(test_labels),
        'test_class_balance': test_labels.mean()
    }
```

**Day 3-5: Run All Transfer Tests**

For each model (Base, SFT, RL-Zero, Think):
- Train on GSM8K, test on HumanEval and LogiQA
- Train on HumanEval, test on GSM8K and LogiQA
- Train on LogiQA, test on GSM8K and HumanEval

**Day 6-7: Analyze Results**

Compute:
- Mean transfer accuracy across all pairs
- Best/worst transfer pairs
- Model comparison (does RL-Zero transfer better than SFT?)
- Asymmetry analysis (is math → code easier than code → math?)

**Deliverable**: `results/h2_transfer_matrix.csv`

### Week 6: Deep Dive and Controls

**Day 1-2: Difficulty Stratification**

Test if transfer works within difficulty strata:
- Bin problems by difficulty (easy/medium/hard)
- Train on easy math, test on easy code
- Train on hard math, test on hard code

**Hypothesis**: If transfer fails overall but succeeds within strata, difficulty is a confound.

**Day 3-4: Feature Decomposition**

Identify which geometric features transfer:
- Train separate classifiers on different feature subsets:
  - Signature coefficients only
  - Curvature measures only
  - Trajectory length only
- Test which features transfer best

**Day 5: Baseline Comparisons**

Compare transfer performance to baselines:
- **Baseline 1**: Train on random labels (should be ~50%)
- **Baseline 2**: Transfer model confidence (logits) instead of geometry
- **Baseline 3**: Transfer output length

**Success criterion**: Geometry transfer > all baselines

**Day 6-7: Write Transfer Report**

Document:
- Transfer matrix with confidence intervals
- Best/worst performing transfers
- Comparison to baselines
- Feature importance for transfer
- Decision: Does H2 succeed or fail?

**Deliverable**: `results/phase3_transfer_report.md`

---

## Analysis Details

### Statistical Testing

For each transfer test, compute:
- **Accuracy**: Fraction correct
- **Confidence interval**: Bootstrap 95% CI (1000 samples)
- **Significance**: Binomial test against 50% chance
- **Effect size**: Cohen's h for proportion difference

**Report format**:
```
GSM8K → HumanEval (olmo3_rl_zero):
  Accuracy: 58.3% [54.1%, 62.5%]
  p-value: 0.003 (vs 50% chance)
  Effect size: h = 0.17 (small)
  Interpretation: Weak but significant transfer
```

### Transfer Success Criteria

**Strong transfer** (supports H2):
- Mean accuracy > 60% across all transfers
- At least 4/6 transfers significantly above chance (p < 0.05)
- Beats all baselines

**Weak transfer** (challenges H2):
- Mean accuracy 52-58%
- Only 2-3/6 transfers significant
- Comparable to some baselines

**No transfer** (falsifies H2):
- Mean accuracy ≤52%
- No transfers significantly above chance
- Baselines perform equally well

### Model Comparison

**Hypothesis**: RL-Zero may transfer better than SFT (if RLVR produces more domain-general computation)

---

## Dynamical Systems Analysis (New)

This section adds three new analysis methods that connect to the theoretical framework.

### 1. MARBLE-style Vector Field Decomposition

**Background**: The MARBLE framework (Gosztolai & Bhattacharyya) treats neural dynamics as vector fields on manifolds and decomposes them via Helmholtz decomposition.

**Method**:

```python
def compute_vector_field(trajectories):
    """
    trajectories: (n_samples, seq_len, n_layers, d_model)

    Treat layer transitions as discrete dynamics:
    v_l(x) = x_{l+1} - x_l
    """
    # Layer-to-layer velocity field
    velocities = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]
    # Shape: (n_samples, seq_len, n_layers-1, d_model)
    return velocities

def helmholtz_decomposition(velocities, positions):
    """
    Decompose velocity field into potential + rotational components.

    v = ∇φ + ∇×A

    Potential (∇φ): Gradient-following, converging to attractors
    Rotational (∇×A): Cycling, oscillatory dynamics
    """
    # Approach 1: Fit neural network f(x) to predict v(x)
    # Then compute divergence (∇·v) and curl (∇×v)

    # Approach 2: Use Hodge decomposition on discretized manifold
    # Requires building graph Laplacian from trajectory points

    # Approach 3: PCA on velocity directions
    # High-variance directions ~ potential flow
    # Orthogonal directions ~ rotational flow

    # We'll use Approach 3 (most tractable for high-dim data)
    ...
```

**What we measure**:
- **Potential ratio**: ||∇φ|| / ||v|| — fraction of flow that's gradient-following
- **Rotational ratio**: ||∇×A|| / ||v|| — fraction of flow that's cycling
- **Directional consistency**: How aligned are velocities with mean flow direction?

**Hypothesis**: Correct solutions have higher potential ratio (more direct paths to answer).

**Implementation**:

```python
def analyze_flow_structure(trajectories, labels):
    """
    Compare flow structure between correct and incorrect solutions.
    """
    velocities = compute_vector_field(trajectories)

    # Compute per-sample metrics
    results = []
    for i, (v, label) in enumerate(zip(velocities, labels)):
        # v: (seq_len, n_layers-1, d_model)

        # 1. Mean velocity magnitude per layer
        v_mag = np.linalg.norm(v, axis=-1).mean(axis=0)  # (n_layers-1,)

        # 2. Velocity direction consistency (how aligned across tokens)
        v_normalized = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
        consistency = np.abs(v_normalized.mean(axis=0)).mean(axis=-1)  # (n_layers-1,)

        # 3. PCA-based potential ratio
        for layer in range(v.shape[1]):
            v_layer = v[:, layer, :]  # (seq_len, d_model)
            _, s, _ = np.linalg.svd(v_layer, full_matrices=False)
            # Top singular value captures "main flow direction" (potential-like)
            potential_ratio = s[0] / s.sum()

        results.append({
            'is_correct': label,
            'velocity_magnitude': v_mag,
            'direction_consistency': consistency,
            'potential_ratio': potential_ratio
        })

    return pd.DataFrame(results)
```

### 2. Lyapunov Exponent Analysis

**Background**: Lyapunov exponents measure the rate of trajectory divergence/convergence. From the HRM paper (Ren & Liu, 2026), correct solutions should show stable dynamics (finding right attractor), while incorrect solutions may show instability or convergence to wrong attractors.

**Method**:

```python
def compute_lyapunov_exponents(trajectories):
    """
    Compute local Lyapunov exponents along trajectories.

    For discrete dynamics x_{l+1} = f(x_l), the Lyapunov exponent is:
    λ = lim (1/L) Σ_l log(||J_l||)

    where J_l = ∂f/∂x is the local Jacobian.

    Since we don't have explicit f, we estimate from nearby trajectories.
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    lyapunov_per_sample = []

    for i in range(n_samples):
        traj = trajectories[i]  # (seq_len, n_layers, d_model)

        # Estimate local expansion/contraction
        layer_lyapunov = []
        for l in range(n_layers - 1):
            x_l = traj[:, l, :]      # (seq_len, d_model)
            x_l1 = traj[:, l+1, :]   # (seq_len, d_model)

            # Method 1: Singular value of transition
            # (crude approximation - assumes linear dynamics)
            delta_x = x_l1 - x_l
            _, s, _ = np.linalg.svd(delta_x, full_matrices=False)

            # Use ratio of singular values as expansion indicator
            expansion = np.log(s[0] / (s[-1] + 1e-8))
            layer_lyapunov.append(expansion)

        lyapunov_per_sample.append({
            'layer_lyapunov': layer_lyapunov,
            'mean_lyapunov': np.mean(layer_lyapunov),
            'max_lyapunov': np.max(layer_lyapunov),
            'lyapunov_trend': np.polyfit(range(len(layer_lyapunov)), layer_lyapunov, 1)[0]
        })

    return pd.DataFrame(lyapunov_per_sample)

def estimate_jacobian_spectrum(trajectories, n_neighbors=10):
    """
    More sophisticated: estimate Jacobian from local neighborhood.

    For each point x, find k nearest neighbors and fit linear map
    from their layer-l positions to layer-(l+1) positions.
    """
    from sklearn.neighbors import NearestNeighbors

    # Flatten trajectories for neighbor search
    # Group by layer, find neighbors, estimate local Jacobian
    ...
```

**What we measure**:
- **Mean Lyapunov exponent**: Overall stability
- **Max Lyapunov exponent**: Worst-case instability
- **Lyapunov spectrum**: Per-layer stability profile
- **Lyapunov trend**: Does stability increase through layers? (convergence)

**Hypothesis**: Correct solutions have more negative Lyapunov exponents (stable convergence to attractor).

**Alternative hypothesis**: Correct solutions show controlled instability (exploration) in early layers, then convergence (exploitation) in late layers.

### 3. Attractor Analysis

**Background**: The HRM analysis shows that iterative reasoning involves multiple fixed points. Correct solutions find the right attractor; incorrect solutions get trapped in wrong basins.

**Method**:

```python
def analyze_attractors(trajectories, labels):
    """
    Characterize attractor structure in trajectory space.
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    # 1. Final layer activations as "attractor proxies"
    final_states = trajectories[:, :, -1, :]  # (n_samples, seq_len, d_model)
    final_mean = final_states.mean(axis=1)    # (n_samples, d_model)

    # 2. Cluster final states
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10)  # Assume ~10 attractor basins
    cluster_labels = kmeans.fit_predict(final_mean)

    # 3. Analyze cluster composition
    for cluster_id in range(10):
        mask = cluster_labels == cluster_id
        n_correct = labels[mask].sum()
        n_total = mask.sum()
        purity = max(n_correct, n_total - n_correct) / n_total
        print(f"Cluster {cluster_id}: {n_total} samples, {purity:.1%} purity")

    # 4. Convergence rate to final state
    convergence_rates = []
    for i in range(n_samples):
        traj = trajectories[i]  # (seq_len, n_layers, d_model)
        final = traj[:, -1, :]  # (seq_len, d_model)

        # Distance to final state at each layer
        distances = []
        for l in range(n_layers):
            dist = np.linalg.norm(traj[:, l, :] - final, axis=-1).mean()
            distances.append(dist)

        # Fit exponential decay
        # d(l) = d_0 * exp(-λ * l)
        log_dist = np.log(np.array(distances) + 1e-8)
        decay_rate, _ = np.polyfit(range(n_layers), log_dist, 1)

        convergence_rates.append({
            'decay_rate': -decay_rate,  # Positive = converging
            'final_distance': distances[-1],
            'is_correct': labels[i]
        })

    return pd.DataFrame(convergence_rates)
```

**What we measure**:
- **Cluster purity**: Do correct/incorrect solutions occupy different attractor basins?
- **Convergence rate**: How quickly does trajectory approach final state?
- **Basin separation**: Distance between correct and incorrect cluster centroids

**Hypothesis**: Correct and incorrect solutions converge to different attractor basins, with correct solutions showing faster, more stable convergence.

### 4. Curvature Regime Analysis (Proxy Only)

**Background**: Following Merullo et al. (2025), high-curvature weight directions correspond to general computation, low-curvature to memorization.

**IMPORTANT LIMITATION**: The Goodfire K-FAC method requires:
1. Gradient statistics computed during training
2. Fisher information matrix approximation
3. Eigendecomposition of the curvature matrix

**We cannot directly replicate this** because we only have:
- Stored activation trajectories
- Model weights (can load separately)
- No gradient/curvature information

**What we CAN do (proxy approaches)**:

**Approach 1: Singular Value Regime Projection**
- SVD of weight matrices gives us directions ordered by "importance"
- High singular value directions are used more broadly (proxy for high-curvature)
- Low singular value directions are more specialized (proxy for memorization)
- **Caveat**: This is NOT the same as K-FAC curvature — it's a structural proxy

**Approach 2: Activation Effective Dimensionality**
- Compute covariance of activations for correct vs incorrect
- Measure effective rank (how distributed is the representation)
- **Hypothesis**: Correct solutions might use higher effective dimensionality (more distributed)

**Approach 3: Direction Selectivity**
- Measure how selective activations are to specific weight directions
- Low selectivity = distributed (general)
- High selectivity = localized (memorization-like)

```python
def analyze_curvature_regime_proxy(trajectories, model, labels):
    """
    PROXY for curvature regime analysis.
    NOT equivalent to Goodfire K-FAC — this uses structural proxies.
    """
    results = []

    for layer_idx in range(n_layers):
        # Get weight matrix
        W = model.layers[layer_idx].mlp.weight  # or appropriate weight

        # SVD decomposition
        U, S, Vh = np.linalg.svd(W, full_matrices=False)

        # Split into high/low singular value regimes
        n_total = len(S)
        k = n_total // 4  # Top/bottom 25%

        for i, (traj, label) in enumerate(zip(trajectories, labels)):
            act = traj[:, layer_idx, :]  # (seq_len, d_model)

            # Project onto singular value regimes
            high_sv_proj = act @ Vh[:k, :].T   # Top-k directions
            low_sv_proj = act @ Vh[-k:, :].T   # Bottom-k directions

            # Compute activation magnitude in each regime
            high_sv_act = np.linalg.norm(high_sv_proj, axis=-1).mean()
            low_sv_act = np.linalg.norm(low_sv_proj, axis=-1).mean()

            # Effective dimensionality of activations
            act_cov = np.cov(act.T)
            eigvals = np.linalg.eigvalsh(act_cov)
            eff_dim = np.exp(entropy(eigvals / eigvals.sum()))

            results.append({
                'layer': layer_idx,
                'is_correct': label,
                'high_sv_activation': high_sv_act,
                'low_sv_activation': low_sv_act,
                'sv_ratio': high_sv_act / (low_sv_act + 1e-8),
                'effective_dimensionality': eff_dim
            })

    return pd.DataFrame(results)
```

**Interpretation caveats**:
1. High singular values ≠ high curvature (they're related but not identical)
2. Our analysis shows structural properties, not true curvature regimes
3. Any findings should be framed as "activation regime analysis" not "curvature analysis"

**Honest framing**: We test whether correct solutions preferentially activate distributed (high-singular-value) vs localized (low-singular-value) weight directions. This is *inspired by* but not equivalent to Goodfire's curvature analysis.

**Note**: Marked as "proxy only" to be clear about what we can and cannot claim.

### 5. Error-Detection Direction Analysis (Wynroe-style)

**Background**: Keith Wynroe (2025) demonstrated that DeepSeek-R1 contains a linear "error-detection" feature that activates strongly when the model generates errors. At tokens where the model writes incorrect content, this direction shows discontinuous activation spikes — potential signatures of internal error recognition.

**Key insight**: This directly connects to our Phase 3 goals. If correct vs incorrect solutions differ geometrically, one strong signal should be a linear direction that separates them.

**Our contribution**: Wynroe's analysis was on DeepSeek-R1. We apply this methodology to the OLMo 3 model family to test:
1. Does the error-detection direction exist in non-R1 models?
2. Is it present in Base models or only after SFT/RLVR?
3. Does the direction transfer across domains (connects to H2)?

**Method**:

```python
def extract_error_detection_direction(correct_trajectories, incorrect_trajectories, layer_idx=-1):
    """
    Extract error-detection direction via difference-in-means.

    This is a simpler version of Wynroe's approach, using our Phase 2 data
    where we have full trajectories labeled as correct/incorrect.

    Args:
        correct_trajectories: (n_correct, seq_len, n_layers, d_model)
        incorrect_trajectories: (n_incorrect, seq_len, n_layers, d_model)
        layer_idx: Which layer to analyze (-1 = last layer)

    Returns:
        direction: (d_model,) - the error-detection direction
        statistics: dict with significance tests
    """
    # Use mean activation across sequence (or specific token positions)
    correct_acts = correct_trajectories[:, :, layer_idx, :].mean(axis=1)  # (n_correct, d_model)
    incorrect_acts = incorrect_trajectories[:, :, layer_idx, :].mean(axis=1)  # (n_incorrect, d_model)

    # Difference-in-means direction
    mean_correct = correct_acts.mean(axis=0)  # (d_model,)
    mean_incorrect = incorrect_acts.mean(axis=0)  # (d_model,)

    direction = mean_incorrect - mean_correct  # Points toward "error" region
    direction = direction / np.linalg.norm(direction)  # Normalize

    # Project all samples onto direction
    correct_proj = correct_acts @ direction
    incorrect_proj = incorrect_acts @ direction

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(incorrect_proj, correct_proj)
    effect_size = (incorrect_proj.mean() - correct_proj.mean()) / np.std(np.concatenate([correct_proj, incorrect_proj]))

    # Classification accuracy using this single direction
    threshold = (correct_proj.mean() + incorrect_proj.mean()) / 2
    correct_pred = (correct_proj < threshold).mean()
    incorrect_pred = (incorrect_proj > threshold).mean()
    accuracy = (correct_pred * len(correct_proj) + incorrect_pred * len(incorrect_proj)) / (len(correct_proj) + len(incorrect_proj))

    return direction, {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size_d': effect_size,
        'classification_accuracy': accuracy,
        'correct_mean': correct_proj.mean(),
        'incorrect_mean': incorrect_proj.mean(),
        'separation': incorrect_proj.mean() - correct_proj.mean()
    }


def analyze_error_direction_per_layer(trajectories, labels):
    """
    Find the best layer for error detection.

    Applies difference-in-means at each layer and reports which layer
    shows strongest separation.
    """
    n_layers = trajectories.shape[2]

    correct_traj = trajectories[labels == True]
    incorrect_traj = trajectories[labels == False]

    results = []
    for layer_idx in range(n_layers):
        direction, stats = extract_error_detection_direction(
            correct_traj, incorrect_traj, layer_idx=layer_idx
        )
        results.append({
            'layer': layer_idx,
            **stats
        })

    return pd.DataFrame(results)


def test_direction_transfer(train_trajectories, train_labels, test_trajectories, test_labels):
    """
    Test if error-detection direction transfers across domains.

    1. Extract direction from train domain (e.g., GSM8K)
    2. Apply to test domain (e.g., HumanEval)
    3. Measure classification accuracy

    This directly tests H2 using a single interpretable feature.
    """
    # Extract direction from train domain
    correct_train = train_trajectories[train_labels == True]
    incorrect_train = train_trajectories[train_labels == False]

    direction, train_stats = extract_error_detection_direction(
        correct_train, incorrect_train, layer_idx=-1  # Best layer (determined earlier)
    )

    # Apply to test domain
    test_acts = test_trajectories[:, :, -1, :].mean(axis=1)  # (n_test, d_model)
    test_proj = test_acts @ direction

    # Compute accuracy on test domain
    correct_test = test_proj[test_labels == True]
    incorrect_test = test_proj[test_labels == False]

    # Use train threshold
    threshold = train_stats['correct_mean'] + (train_stats['separation'] / 2)

    correct_pred = (correct_test < threshold).mean()
    incorrect_pred = (incorrect_test > threshold).mean()
    test_accuracy = (correct_pred * len(correct_test) + incorrect_pred * len(incorrect_test)) / len(test_labels)

    return {
        'train_accuracy': train_stats['classification_accuracy'],
        'test_accuracy': test_accuracy,
        'transfer_ratio': test_accuracy / train_stats['classification_accuracy']
    }
```

**What we measure**:
- **Direction existence**: Does a significant error-detection direction exist? (p < 0.05, d > 0.5)
- **Layer profile**: Which layers show strongest separation?
- **Model comparison**: Is direction stronger in RL-Zero vs Base vs SFT?
- **Transfer**: Does direction from GSM8K work on HumanEval/LogiQA?

**Hypotheses**:
1. RL-Zero will show stronger error-detection direction than Base (RLVR learns error awareness)
2. SFT may show weak or no direction (no outcome-based training signal)
3. Direction will partially transfer across domains (connects to H2)

**Connection to Wynroe's work**:
- Wynroe used clean/corrupted pairs (same problem, artificially changed answer)
- We use correct/incorrect solutions (different problems, natural variation)
- Our approach is simpler but tests the same underlying hypothesis
- If we find similar results on OLMo, it validates cross-model generalization

**Advantages of using Phase 2 data**:
- No additional data collection needed (zero GPU time)
- Already have correctness labels
- Can compare across all 4 models and 3 tasks
- Directly integrates with other Phase 3 analyses

**Limitations**:
- Our correct/incorrect distinction may be noisier than clean/corrupted
- We measure at final-answer level, not token-level errors
- Cannot detect within-CoT error recognition (future work with aha_moment experiment)

### 6. Menger Curvature Analysis

**Background**: Zhou et al. (2025) "The Geometry of Reasoning: Flowing Logics in Representation Space" (arXiv:2510.09782) found that different orders of trajectory derivatives reveal different structure:
- **0th order (positions)**: Clusters by surface semantics (topic, language)
- **1st order (velocities)**: Begins to show logical structure
- **2nd order (curvature)**: Dominated by logical structure — flows with same logical skeleton show correlated curvature even across different topics/languages

**Key insight**: Menger curvature captures how "sharply" a trajectory turns at each point, computed from just 3 consecutive states.

**Method**:

```python
def compute_menger_curvature(p1, p2, p3):
    """
    Compute Menger curvature for three consecutive points.

    Menger curvature κ = 1/R where R is the circumcircle radius.

    Formula: κ = 4 * Area(triangle) / (|p1-p2| * |p2-p3| * |p3-p1|)

    For high-dimensional points, this generalizes naturally.
    """
    # Side lengths
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)

    # Avoid division by zero
    if a < 1e-10 or b < 1e-10 or c < 1e-10:
        return 0.0

    # Area via cross product (works in high dimensions)
    v1 = p2 - p1
    v2 = p3 - p1

    # For high-dim: use singular values of [v1, v2] matrix
    # The 2D "area" is sqrt(det(G)) where G = [v1, v2]^T @ [v1, v2]
    gram = np.array([[np.dot(v1, v1), np.dot(v1, v2)],
                     [np.dot(v1, v2), np.dot(v2, v2)]])
    area = 0.5 * np.sqrt(max(0, np.linalg.det(gram)))

    # Menger curvature
    curvature = 4 * area / (a * b * c)

    return curvature


def compute_trajectory_curvature_profile(trajectory):
    """
    Compute Menger curvature at each point along a layer trajectory.

    Args:
        trajectory: (n_layers, d_model) - single sample's layer trajectory

    Returns:
        curvatures: (n_layers-2,) - curvature at each interior point
    """
    n_layers = trajectory.shape[0]
    curvatures = []

    for i in range(1, n_layers - 1):
        p1 = trajectory[i-1]
        p2 = trajectory[i]
        p3 = trajectory[i+1]

        kappa = compute_menger_curvature(p1, p2, p3)
        curvatures.append(kappa)

    return np.array(curvatures)


def analyze_menger_curvature(trajectories, labels):
    """
    Compare curvature profiles between correct and incorrect solutions.

    Based on Zhou et al.'s finding that curvature captures logical structure
    better than raw positions.
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    results = []

    for i in range(n_samples):
        # Average trajectory across sequence positions
        mean_traj = trajectories[i].mean(axis=0)  # (n_layers, d_model)

        # Compute curvature profile
        curvatures = compute_trajectory_curvature_profile(mean_traj)

        results.append({
            'is_correct': labels[i],
            'mean_curvature': curvatures.mean(),
            'max_curvature': curvatures.max(),
            'curvature_variance': curvatures.var(),
            'early_curvature': curvatures[:len(curvatures)//2].mean(),  # First half of layers
            'late_curvature': curvatures[len(curvatures)//2:].mean(),   # Second half of layers
            'curvature_profile': curvatures.tolist()
        })

    return pd.DataFrame(results)


def compute_curvature_correlation(trajectories1, trajectories2, labels1, labels2):
    """
    Test Zhou et al.'s key finding: do correct solutions show
    correlated curvature profiles across different domains?

    If curvature captures logical structure (not surface semantics),
    correct solutions in math and code should have similar curvature patterns.
    """
    # Get correct solutions from each domain
    correct1 = trajectories1[labels1 == True]
    correct2 = trajectories2[labels2 == True]

    # Compute mean curvature profiles
    profiles1 = []
    for traj in correct1:
        mean_traj = traj.mean(axis=0)
        profiles1.append(compute_trajectory_curvature_profile(mean_traj))

    profiles2 = []
    for traj in correct2:
        mean_traj = traj.mean(axis=0)
        profiles2.append(compute_trajectory_curvature_profile(mean_traj))

    # Average curvature profile per domain
    mean_profile1 = np.mean(profiles1, axis=0)
    mean_profile2 = np.mean(profiles2, axis=0)

    # Correlation between domain curvature profiles
    from scipy.stats import pearsonr, spearmanr

    pearson_r, pearson_p = pearsonr(mean_profile1, mean_profile2)
    spearman_r, spearman_p = spearmanr(mean_profile1, mean_profile2)

    return {
        'pearson_correlation': pearson_r,
        'pearson_pvalue': pearson_p,
        'spearman_correlation': spearman_r,
        'spearman_pvalue': spearman_p,
        'profile1': mean_profile1.tolist(),
        'profile2': mean_profile2.tolist()
    }
```

**What we measure**:
- **Mean curvature**: Overall trajectory "sharpness"
- **Curvature variance**: Consistency of turns
- **Early vs late curvature**: Where does sharp turning happen?
- **Cross-domain correlation**: Do correct solutions in different domains have similar curvature profiles?

**Hypotheses** (following Zhou et al.):
1. Curvature profiles will be more consistent across correct solutions than positions
2. Correct solutions may show distinct curvature signatures (e.g., higher early curvature = more "setup", lower late curvature = "convergence")
3. Cross-domain curvature correlation will be higher for correct solutions (logical structure transfer)
4. RLVR models may show more domain-invariant curvature patterns

**Connection to Zhou et al.**:
- They study token-by-token reasoning trajectories
- We study layer-by-layer activation trajectories
- Both test whether curvature captures logical structure beyond surface form
- If we find similar results, it validates cross-architecture generalization

**Computational note**: Menger curvature is O(n) per trajectory and requires no model weights or gradients — can be computed directly from stored activations.

---

## Week-by-Week Breakdown (Updated)

### Week 5: Original Analysis

**Days 1-2**: Path signatures and basic curvature
**Days 3-5**: Cross-domain transfer tests
**Days 6-7**: Baseline comparisons

### Week 6: Dynamical Systems Analysis (New)

**Days 1-2**: Vector Field Decomposition
- Implement `compute_vector_field()` and `helmholtz_decomposition()`
- Compute potential ratio and directional consistency
- Compare correct vs incorrect solutions

**Days 3-4**: Lyapunov Analysis
- Implement `compute_lyapunov_exponents()`
- Analyze per-layer stability profiles
- Test hypothesis: correct = more stable

**Days 5-6**: Menger Curvature Analysis (Zhou et al.)
- Implement `compute_menger_curvature()` and `analyze_menger_curvature()`
- Compute curvature profiles for correct vs incorrect solutions
- Test cross-domain curvature correlation (key H2 test)
- Compare to Zhou et al.'s finding: curvature captures logic > semantics

**Day 7**: Attractor Analysis
- Cluster final states
- Measure cluster purity (correct vs incorrect)
- Analyze convergence rates

### Week 7: Error-Detection Direction + Transfer Analysis

**Days 1-2**: Error-Detection Direction Analysis (Wynroe-style)
- Implement `extract_error_detection_direction()` using Phase 2 data
- Run `analyze_error_direction_per_layer()` for each model
- Compare direction strength: Base vs SFT vs RL-Zero vs Think
- Identify best layer for error detection

**Days 3-4**: Direction Transfer Tests
- Extract direction from GSM8K, test on HumanEval/LogiQA
- Extract direction from each domain, test cross-domain
- Compare transfer to full-feature classifier transfer
- This is a **direct H2 test** using a single interpretable feature

**Days 5-6**: Feature Importance Analysis
- Which features transfer best: direction vs path signatures vs dynamical features?
- Which are domain-specific?
- Model comparison: does RL-Zero show better direction transfer?

**Day 7**: Curvature Regime Analysis (if time permits)
- Load model weights
- Project trajectories onto singular value regimes
- Test high-curvature hypothesis

### Week 8: Synthesis and Report

**Days 1-3**: Integrate Results
- Combine original and dynamical analysis
- Identify consistent vs conflicting findings
- Characterize what transfers vs what doesn't

**Days 4-5**: Write Comprehensive Report
- Theoretical framing (interpolation view)
- Dynamical analysis results
- Transfer matrix with all feature sets
- Decision on H2

**Days 6-7**: Prepare for Phase 4 (or Pivot)
- If H2 succeeds: design steering intervention
- If H2 fails: characterize domain-specific geometry

---

## Deliverables (Updated)

### Data Files
- `results/h2_transfer_matrix.csv`: Original transfer results
- `results/h2_dynamical_features.csv`: Vector field, Lyapunov, attractor metrics
- `results/h2_dynamical_transfer.csv`: Transfer with dynamical features
- `results/h2_curvature_regime.csv`: Curvature regime analysis (if done)
- `results/h2_menger_curvature.csv`: Menger curvature profiles (per sample)
- `results/h2_menger_correlation.csv`: Cross-domain curvature correlation (Zhou et al. test)
- `results/h2_error_direction.csv`: Error-detection direction analysis (per model, per layer)
- `results/h2_direction_transfer.csv`: Direction transfer matrix (6 transfers × 4 models)

### Visualizations
- Transfer heatmap (original vs dynamical features)
- Lyapunov spectrum: correct vs incorrect
- Attractor cluster visualization (t-SNE/UMAP of final states)
- Vector field streamlines (2D projection)
- **Menger curvature profile**: Curvature by layer for correct vs incorrect
- **Cross-domain curvature scatter**: Correlation of curvature profiles across domains
- **Error-direction layer profile**: Effect size (d) by layer for each model
- **Direction transfer heatmap**: Train domain → Test domain accuracy
- **Model comparison bar chart**: Direction separation by training paradigm

### Reports
- `results/phase3_transfer_report.md`: Full analysis
- `results/phase3_dynamical_analysis.md`: Dynamical systems findings
- `results/phase3_decision.md`: Go/no-go for Phase 4

---

## Model Comparison (Continued)

**Test**: Compare mean transfer accuracy across models
- RL-Zero vs SFT: t-test on 6 transfer accuracies
- RL-Zero vs Think: t-test
- Base vs all fine-tuned: ANOVA

**Interpretation**:
- If RL-Zero > SFT: RLVR may learn more transferable geometry
- If no difference: Training paradigm doesn't affect transfer
- If Base > fine-tuned: Post-training may hurt transfer (unlikely)

---

## Confound Analysis

### Difficulty Confound

**Test**: Stratified transfer
- Bin problems by difficulty (using problem length as proxy)
- Train/test within same difficulty bin

**Interpretation**:
- If stratified transfer succeeds but overall fails: Difficulty is confound
- If both fail: Geometry doesn't transfer, regardless of difficulty

### Format Confound

**Test**: Within-format vs cross-format transfer
- Within: GSM8K → MATH (both math)
- Cross: GSM8K → HumanEval (math → code)

**Interpretation**:
- If within >> cross: Format is confound
- If within ≈ cross: Format is not the issue

### Length Confound

**Test**: Length-matched transfer
- Match correct/incorrect pairs by output length (±10%)
- Re-run transfer tests on matched subset

**Interpretation**:
- If matched transfer >> unmatched: Length is confound
- If similar: Length is not the issue

---

## Decision Tree

### If H2 Succeeds (Mean Transfer > 58%)

**Next steps**:
1. Proceed to Phase 4 (H4: Steering)
2. Write up positive results
3. Test H3 (non-verifiable domains) if resources allow

**Paper framing**: "Geometric Signatures of Reasoning Transfer Across Domains"

### If H2 Partially Succeeds (Some Transfers Work)

**Next steps**:
1. Characterize which transfers work and why
2. Measure transfer as function of domain similarity
3. Identify transferable vs domain-specific features

**Paper framing**: "Domain Similarity and Geometric Transfer in LLM Reasoning"

### If H2 Fails (No Transfer Above Chance)

**Pivot strategies**:

**Pivot 1: Domain-Specific Geometry**
- Characterize what differs across domains
- Train domain-specific detectors (still useful)
- Paper: "Domain-Specific Geometric Signatures of Reasoning"

**Pivot 2: Hierarchical Transfer**
- Test finer-grained domain similarities
- Math → Physics → Chemistry → Biology
- Measure transfer decay with domain distance

**Pivot 3: Feature Engineering**
- Current features may not capture transferable aspects
- Try: Attention patterns, gradient flow, layer-wise changes
- Paper: "What Geometric Features Capture Reasoning?"

---

## Risks and Mitigation

### Risk 1: Transfer Tests Are Underpowered

**Problem**: With 500 test samples, detecting small effects (55% vs 50%) requires large N.

**Mitigation**:
- Use bootstrap confidence intervals
- Report effect sizes, not just p-values
- Consider collecting more test samples if results are borderline

### Risk 2: Class Imbalance Affects Transfer

**Problem**: If test set is 80% correct, classifier can achieve 80% by always predicting "correct."

**Mitigation**:
- Report precision, recall, F1 (not just accuracy)
- Use balanced accuracy: (TPR + TNR) / 2
- Stratified sampling to balance test set

### Risk 3: Overfitting on Source Domain

**Problem**: Classifier may overfit to source domain specifics.

**Mitigation**:
- Use cross-validation on source domain
- Regularize classifier (max_depth, min_samples_leaf)
- Try simpler classifiers (logistic regression)

---

## Deliverables

### Data Files

- `results/h2_transfer_matrix.csv`: All transfer results
- `results/h2_feature_importance.csv`: Which features transfer
- `results/h2_difficulty_stratified.csv`: Stratified transfer results

### Visualizations

- Transfer heatmap (6×4 matrix: transfers × models)
- Feature importance bar chart
- Accuracy vs domain similarity scatter plot

### Reports

- `results/phase3_transfer_report.md`: Full analysis
- `results/phase3_decision.md`: Go/no-go for Phase 4

---

## Success Criteria

**Minimum viable**:
- All 24 transfer tests completed
- Statistical tests performed
- Decision made on H2 (succeed/fail/partial)

**Target**:
- Mean transfer accuracy > 55%
- At least 50% of transfers significantly above chance
- Clear interpretation of results

**Stretch**:
- Mean transfer accuracy > 60%
- Identified specific transferable geometric features
- Model comparison shows RL-Zero > SFT for transfer

---

## Timeline Contingencies

**If H2 clearly succeeds** (by Day 3 of Week 5):
- Skip some redundant tests
- Move to Phase 4 early
- Use extra time for H4 implementation

**If H2 clearly fails** (by Day 3 of Week 5):
- Stop transfer tests
- Focus on understanding why (confound analysis)
- Begin pivot planning

**If results are ambiguous**:
- Extend analysis by 1 week
- Collect more samples if needed
- Consult literature for similar effect sizes
