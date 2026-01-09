"""
Geometric Measures Module

Computes geometric properties of activation manifolds:
- Effective rank
- Spectral decay
- Subspace preservation
- Local curvature
"""

import numpy as np
from scipy.linalg import svd, subspace_angles
from scipy.stats import entropy
from typing import Tuple, Optional
import warnings


def compute_effective_rank(activations: np.ndarray, eps: float = 1e-10) -> float:
    """
    Effective rank: exponential of entropy of normalized singular values.

    Measures how many dimensions are "actively used" in the representation.
    Low effective rank = concentrated on few dimensions (sparse/compressed)
    High effective rank = spread across many dimensions (distributed)

    Args:
        activations: (n_samples, d_model)
        eps: Epsilon for numerical stability

    Returns:
        Effective rank (scalar between 1 and min(n_samples, d_model))
    """
    # SVD
    _, s, _ = svd(activations, full_matrices=False)

    # Normalize to probability distribution
    s_normalized = s / (s.sum() + eps)

    # Remove near-zeros for entropy calculation
    s_normalized = s_normalized[s_normalized > eps]

    if len(s_normalized) == 0:
        warnings.warn("All singular values near zero, returning rank=1")
        return 1.0

    # Effective rank = exp(entropy)
    return np.exp(entropy(s_normalized))


def compute_spectral_decay(
    activations: np.ndarray,
    fit_range: Optional[Tuple[int, int]] = None
) -> Tuple[float, np.ndarray, float]:
    """
    Fit power-law to singular value decay: s_i ∝ i^(-α)

    Higher α = faster decay = more concentrated spectrum
    Lower α = slower decay = more distributed spectrum

    Args:
        activations: (n_samples, d_model)
        fit_range: Optional (start, end) indices for fitting. None = use all

    Returns:
        alpha: Power-law exponent
        singular_values: Raw singular values for inspection
        r_squared: Goodness of fit
    """
    _, s, _ = svd(activations, full_matrices=False)

    # Determine fit range
    if fit_range is None:
        start, end = 0, len(s)
    else:
        start, end = fit_range
        end = min(end, len(s))

    # Fit log-log regression: log(s) = -α * log(i) + c
    indices = np.arange(start + 1, end + 1)  # Start from 1 to avoid log(0)
    log_i = np.log(indices)
    log_s = np.log(s[start:end] + 1e-10)  # Add epsilon for numerical stability

    # Least squares fit
    A = np.vstack([log_i, np.ones_like(log_i)]).T
    alpha_neg, intercept = np.linalg.lstsq(A, log_s, rcond=None)[0]

    # Compute R-squared
    log_s_pred = alpha_neg * log_i + intercept
    ss_res = np.sum((log_s - log_s_pred) ** 2)
    ss_tot = np.sum((log_s - log_s.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return -alpha_neg, s, r_squared


def compute_subspace_preservation(
    base_activations: np.ndarray,
    finetuned_activations: np.ndarray,
    k: int = 100,
    method: str = "principal_angles"
) -> Tuple[float, np.ndarray]:
    """
    Measure how much of base model's top-k subspace is preserved after fine-tuning.

    Uses principal angles between subspaces.
    Preservation = 1: Perfect preservation (identical subspaces)
    Preservation = 0: Orthogonal subspaces (no overlap)

    Args:
        base_activations: (n_samples, d_model) from base model
        finetuned_activations: (n_samples, d_model) from fine-tuned model
        k: Number of top singular vectors to compare
        method: "principal_angles" or "projection" or "cka"

    Returns:
        preservation_score: Mean cosine of principal angles
        angles: Individual principal angles (in radians)
    """
    if base_activations.shape != finetuned_activations.shape:
        raise ValueError(
            f"Shape mismatch: base {base_activations.shape} "
            f"vs finetuned {finetuned_activations.shape}"
        )

    n_samples, d_model = base_activations.shape
    k = min(k, n_samples, d_model)

    if method == "principal_angles":
        # Get top-k right singular vectors
        _, _, Vt_base = svd(base_activations, full_matrices=False)
        _, _, Vt_ft = svd(finetuned_activations, full_matrices=False)

        V_base_k = Vt_base[:k, :].T  # (d_model, k)
        V_ft_k = Vt_ft[:k, :].T

        # Compute principal angles using scipy
        angles = subspace_angles(V_base_k, V_ft_k)

        # Preservation score
        preservation = np.cos(angles).mean()

        return preservation, angles

    elif method == "projection":
        # Alternative: measure how well base subspace projects onto finetuned
        _, _, Vt_base = svd(base_activations, full_matrices=False)
        _, _, Vt_ft = svd(finetuned_activations, full_matrices=False)

        V_base_k = Vt_base[:k, :].T
        V_ft_k = Vt_ft[:k, :].T

        # Projection: ||V_base^T V_ft||_F^2 / k
        projection = np.linalg.norm(V_base_k.T @ V_ft_k, 'fro') ** 2 / k
        preservation = projection / k  # Normalize to [0, 1]

        return preservation, None

    elif method == "cka":
        # Centered Kernel Alignment - alternative similarity metric
        preservation = compute_cka(base_activations, finetuned_activations)
        return preservation, None

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two activation matrices.

    CKA is invariant to orthogonal transformations and isotropic scaling.
    CKA = 1: Perfectly aligned representations
    CKA = 0: Uncorrelated representations

    Args:
        X: (n_samples, d1)
        Y: (n_samples, d2)

    Returns:
        CKA score in [0, 1]
    """
    def center_gram(K):
        """Center a Gram matrix."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    # Compute Gram matrices
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # Center
    K_X_centered = center_gram(K_X)
    K_Y_centered = center_gram(K_Y)

    # CKA formula
    hsic = np.sum(K_X_centered * K_Y_centered)
    var_x = np.sqrt(np.sum(K_X_centered * K_X_centered))
    var_y = np.sqrt(np.sum(K_Y_centered * K_Y_centered))

    if var_x * var_y == 0:
        return 0.0

    return hsic / (var_x * var_y)


def compute_participation_ratio(activations: np.ndarray) -> float:
    """
    Participation ratio: alternative dimensionality measure.

    PR = (sum(s_i))^2 / sum(s_i^2)

    More robust to noise than effective rank.

    Args:
        activations: (n_samples, d_model)

    Returns:
        Participation ratio
    """
    _, s, _ = svd(activations, full_matrices=False)

    # Compute PR
    pr = (s.sum() ** 2) / (s ** 2).sum()

    return pr


def compute_stable_rank(activations: np.ndarray) -> float:
    """
    Stable rank: ||A||_F^2 / ||A||_2^2 = sum(s_i^2) / max(s_i)^2

    More stable to outliers than matrix rank.

    Args:
        activations: (n_samples, d_model)

    Returns:
        Stable rank
    """
    _, s, _ = svd(activations, full_matrices=False)

    if s[0] == 0:
        return 0.0

    stable_rank = (s ** 2).sum() / (s[0] ** 2)

    return stable_rank


def compute_layer_alignment(
    activations_1: np.ndarray,
    activations_2: np.ndarray,
    method: str = "cka"
) -> float:
    """
    Compute alignment between two layers (same or different models).

    Useful for analyzing:
    - How representations evolve across layers
    - How similar two models are at a given layer

    Args:
        activations_1: (n_samples, d_model_1)
        activations_2: (n_samples, d_model_2)
        method: "cka" or "correlation"

    Returns:
        Alignment score in [0, 1]
    """
    if activations_1.shape[0] != activations_2.shape[0]:
        raise ValueError("Number of samples must match")

    if method == "cka":
        return compute_cka(activations_1, activations_2)

    elif method == "correlation":
        # If dimensions match, use correlation
        if activations_1.shape[1] == activations_2.shape[1]:
            # Flatten and compute correlation
            corr = np.corrcoef(activations_1.flatten(), activations_2.flatten())[0, 1]
            return abs(corr)  # Absolute value for alignment
        else:
            # Use CKA for different dimensions
            return compute_cka(activations_1, activations_2)

    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_spectrum(activations: np.ndarray) -> dict:
    """
    Comprehensive spectral analysis.

    Returns dict with multiple measures for robustness.
    """
    _, s, _ = svd(activations, full_matrices=False)

    # Various dimensionality measures
    effective_rank = compute_effective_rank(activations)
    participation_ratio = compute_participation_ratio(activations)
    stable_rank = compute_stable_rank(activations)

    # Spectral decay
    alpha, _, r2 = compute_spectral_decay(activations)

    # Condition number
    condition_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf

    # 90% energy threshold
    energy_cumsum = np.cumsum(s ** 2)
    energy_total = energy_cumsum[-1]
    n_components_90 = np.searchsorted(energy_cumsum, 0.9 * energy_total) + 1

    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "stable_rank": stable_rank,
        "spectral_decay_alpha": alpha,
        "spectral_decay_r2": r2,
        "condition_number": condition_number,
        "n_components_90pct": n_components_90,
        "singular_values": s,
    }
