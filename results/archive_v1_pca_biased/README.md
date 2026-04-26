# Archive: v1 Analysis (PCA-Biased)

**Archived Date**: 2026-02-12
**Reason**: PCA bias invalidated these analyses

## Why These Files Were Archived

These files contain Phase 3 analysis results that used PCA-based dimensionality reduction. On 2026-02-12, we discovered a critical validity issue:

**The Problem**: PCA captures high-variance directions (dominated by frequent features due to neural superposition), NOT correctness-relevant directions. Additionally, activation covariance eigenvectors have no guaranteed correspondence with Jacobian singular vectors (functionally important directions).

**Impact**:
- Path signature analysis used `PCA(n_components=32)` before computing signatures - discarded exactly the tail eigenvectors where RLVR refinement occurs (3-8x more change than top eigenvectors)
- Attractor analysis used `PCA(n_components=50)` before K-means clustering - biased toward high-variance not correctness-relevant clusters
- Cross-domain transfer results were near-random (0.3-0.5 AUC)

**The Fix**: Created v2 analyses with three alternative projection methods:
1. **Random Projection** (GaussianRandomProjection) - preserves pairwise distances without variance bias
2. **Velocity-space PCA** - PCA on v = x_{l+1} - x_l (dynamics, not static position)
3. **Probe-informed CV** - Logistic probe weight as dim 0 + random orthogonal complement

**v2 Results**:
- Path signatures with probe-informed projection: **0.6-0.85 AUC** within-domain
- Cross-domain transfer: **0.7-0.9 AUC** (vs 0.3-0.5 in v1)
- **H2 (domain-invariant signatures) confirmed** when using correctness-informed projections

## Archived Files

### Analysis Reports
- **DYNAMICAL_ANALYSIS_FINDINGS.md** - Early dynamical analysis (Lyapunov, attractor, etc.)
- **PHASE3_H1H2_FINDINGS.md** - H1/H2 test results with PCA-biased path signatures

### Data Files
- **h2_path_signatures.csv** - Path signature results with PCA(n=32) - INVALID
- **h2_path_signatures_transfer.csv** - Cross-domain transfer with PCA - INVALID (0.3-0.5 AUC)
- **h2_true_jacobian.csv** - "True Jacobian" analysis (actually delta-based, inflated 1.4-8.9x)
- **h2_true_jacobian_layers.csv** - Per-layer Jacobian analysis (delta-based)

## Replacement Files

See current results directory:
- **V2_ANALYSIS_FINDINGS.md** - Comprehensive v2 analysis with PCA-bias fixes
- **PHASE3_COMPLETE_FINDINGS.md** - Updated master findings document
- **v2_0shot_20260211_212500/** - All v2 output files (CSV, JSON)

## Scripts Deprecated

The following scripts have deprecation warnings pointing to v2 versions:
- `scripts/analysis/path_signature_analysis.py` → `path_signature_analysis_v2.py`
- `scripts/analysis/phase3_dynamical_analysis.py` → `phase3_dynamical_analysis_v2.py`
- `scripts/analysis/h3_remaining_analyses.py` → (sections marked VALID/INVALID)

## Key Lesson

**Do not use PCA for dimensionality reduction when the goal is to find task-relevant features in neural superposition representations.** PCA finds high-variance directions, which are dominated by frequent features (due to superposition), not correctness-relevant features.

**Better alternatives**:
1. Random projections (Johnson-Lindenstrauss) - no bias
2. Task-informed projections (probe weights, gradients)
3. Sparse coding / dictionary learning (directly addresses superposition)
4. Cross-validated linear probes (learns relevant subspace)

---

For details on the v2 analysis, see: [../V2_ANALYSIS_FINDINGS.md](../V2_ANALYSIS_FINDINGS.md)
