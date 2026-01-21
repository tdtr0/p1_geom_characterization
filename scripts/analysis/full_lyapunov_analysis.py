#!/usr/bin/env python3
"""
Full Lyapunov Spectrum Analysis

Tests H_jac1, H_jac2, H_jac3 from the critique document.

Instead of Frobenius norm (which loses directional info), we compute:
1. SVD-based Lyapunov from layer transitions
2. Directional Lyapunov in error-direction subspace
3. Full Lyapunov spectrum comparison
"""

import numpy as np
import h5py
from scipy import stats, linalg
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def compute_layer_jacobian_svd(x_l, x_l1, k=100):
    """
    Estimate Jacobian spectrum between two layers using SVD.

    For layers l and l+1, we have activations at many sequence positions.
    The transition can be approximated as: x_{l+1} ≈ J @ x_l + b

    We estimate J via least squares and compute its singular values.

    Args:
        x_l: (seq_len, d_model) - activations at layer l
        x_l1: (seq_len, d_model) - activations at layer l+1
        k: number of dimensions to project to (for efficiency)

    Returns:
        singular_values: (k,) - approximated Jacobian singular values
    """
    seq_len, d_model = x_l.shape

    # Project to lower dimension for efficiency
    if d_model > k:
        # Use PCA to project both layers consistently
        pca = PCA(n_components=k, random_state=42)
        x_l_proj = pca.fit_transform(x_l)
        x_l1_proj = pca.transform(x_l1)
    else:
        x_l_proj = x_l
        x_l1_proj = x_l1
        k = d_model

    # Estimate Jacobian via least squares: x_{l+1} ≈ J @ x_l
    # J = x_{l+1}.T @ x_l @ (x_l.T @ x_l)^{-1}
    # Or equivalently, solve x_l @ J.T = x_{l+1}

    try:
        # Regularized least squares
        J_T, residuals, rank, s = linalg.lstsq(x_l_proj, x_l1_proj)
        J = J_T.T  # (k, k)

        # Compute singular values
        _, singular_values, _ = linalg.svd(J, full_matrices=False)
        return singular_values
    except:
        return np.ones(k)


def compute_lyapunov_spectrum(trajectory, k=100):
    """
    Compute full Lyapunov spectrum for a trajectory.

    Args:
        trajectory: (seq_len, n_layers, d_model)
        k: projection dimension

    Returns:
        dict with Lyapunov statistics
    """
    seq_len, n_layers, d_model = trajectory.shape

    # Collect Lyapunov exponents at each layer
    layer_max_lyapunov = []
    layer_mean_lyapunov = []
    layer_spectrum_width = []  # std of log(singular values)

    for l in range(n_layers - 1):
        x_l = trajectory[:, l, :]
        x_l1 = trajectory[:, l + 1, :]

        sv = compute_layer_jacobian_svd(x_l, x_l1, k=k)

        # Lyapunov exponents = log of singular values
        lyap = np.log(sv + 1e-10)

        layer_max_lyapunov.append(lyap[0])  # Largest
        layer_mean_lyapunov.append(np.mean(lyap))
        layer_spectrum_width.append(np.std(lyap))

    return {
        'max_lyapunov': np.mean(layer_max_lyapunov),
        'mean_lyapunov': np.mean(layer_mean_lyapunov),
        'spectrum_width': np.mean(layer_spectrum_width),
        'max_lyapunov_profile': np.array(layer_max_lyapunov),
        'mean_lyapunov_profile': np.array(layer_mean_lyapunov)
    }


def compute_directional_lyapunov(trajectory, direction, k=50):
    """
    Compute Lyapunov exponent in a specific direction (e.g., error-direction).

    Projects trajectory onto subspace around the direction, then computes
    expansion/contraction rate in that subspace.

    Args:
        trajectory: (seq_len, n_layers, d_model)
        direction: (d_model,) - normalized direction vector
        k: size of subspace around direction

    Returns:
        float: Lyapunov exponent in direction subspace
    """
    seq_len, n_layers, d_model = trajectory.shape

    # Project trajectory onto direction
    dir_proj = []
    for l in range(n_layers):
        proj = trajectory[:, l, :] @ direction  # (seq_len,)
        dir_proj.append(proj)

    dir_proj = np.array(dir_proj)  # (n_layers, seq_len)

    # Compute expansion rate in direction
    layer_expansion = []
    for l in range(n_layers - 1):
        # Variance ratio as proxy for expansion in this direction
        var_l = np.var(dir_proj[l])
        var_l1 = np.var(dir_proj[l + 1])
        if var_l > 1e-10:
            expansion = np.log(var_l1 / var_l + 1e-10) / 2  # sqrt for std
            layer_expansion.append(expansion)

    return np.mean(layer_expansion) if layer_expansion else 0.0


def extract_error_direction(correct_traj, incorrect_traj, layer_idx=-1):
    """Extract error-detection direction at a specific layer."""
    # Mean activations
    correct_mean = correct_traj[:, :, layer_idx, :].mean(axis=(0, 1))
    incorrect_mean = incorrect_traj[:, :, layer_idx, :].mean(axis=(0, 1))

    direction = incorrect_mean - correct_mean
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    return direction


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8)


def test_h_jac1(correct_lyap, incorrect_lyap):
    """H_jac1: Max Lyapunov differs (incorrect = more chaotic)"""
    print("\n" + "=" * 60)
    print("H_jac1: MAX LYAPUNOV (Incorrect = More Chaotic?)")
    print("=" * 60)

    correct_max = [l['max_lyapunov'] for l in correct_lyap]
    incorrect_max = [l['max_lyapunov'] for l in incorrect_lyap]

    d = cohens_d(incorrect_max, correct_max)  # Positive = incorrect higher
    t_stat, p_val = stats.ttest_ind(incorrect_max, correct_max)

    print(f"\nMax Lyapunov Exponent:")
    print(f"  Correct: {np.mean(correct_max):.4f} +/- {np.std(correct_max):.4f}")
    print(f"  Incorrect: {np.mean(incorrect_max):.4f} +/- {np.std(incorrect_max):.4f}")
    print(f"  Cohen's d = {d:.3f} (positive = incorrect more chaotic)")
    print(f"  p-value = {p_val:.4f}")

    if p_val < 0.05:
        if d > 0:
            print(f"  => CONFIRMED: Incorrect is MORE CHAOTIC (SIGNIFICANT)")
        else:
            print(f"  => OPPOSITE: Incorrect is LESS chaotic (significant)")
    else:
        print(f"  => No significant difference")

    return d, p_val


def test_h_jac2(correct_dir_lyap, incorrect_dir_lyap):
    """H_jac2: Lyapunov in error-direction subspace differs"""
    print("\n" + "=" * 60)
    print("H_jac2: DIRECTIONAL LYAPUNOV (Error-Direction Subspace)")
    print("=" * 60)

    d = cohens_d(incorrect_dir_lyap, correct_dir_lyap)
    t_stat, p_val = stats.ttest_ind(incorrect_dir_lyap, correct_dir_lyap)

    print(f"\nLyapunov in Error-Direction:")
    print(f"  Correct: {np.mean(correct_dir_lyap):.4f} +/- {np.std(correct_dir_lyap):.4f}")
    print(f"  Incorrect: {np.mean(incorrect_dir_lyap):.4f} +/- {np.std(incorrect_dir_lyap):.4f}")
    print(f"  Cohen's d = {d:.3f}")
    print(f"  p-value = {p_val:.4f}")

    if p_val < 0.05:
        print(f"  => SIGNIFICANT difference in error-direction stability")
    else:
        print(f"  => No significant difference")

    return d, p_val


def test_h_jac3(correct_lyap, incorrect_lyap):
    """H_jac3: Spectrum width (flatness) differs"""
    print("\n" + "=" * 60)
    print("H_jac3: SPECTRUM WIDTH (Flat = Uniform Expansion)")
    print("=" * 60)

    correct_width = [l['spectrum_width'] for l in correct_lyap]
    incorrect_width = [l['spectrum_width'] for l in incorrect_lyap]

    d = cohens_d(correct_width, incorrect_width)  # Positive = correct wider
    t_stat, p_val = stats.ttest_ind(correct_width, incorrect_width)

    print(f"\nSpectrum Width (std of Lyapunov exponents):")
    print(f"  Correct: {np.mean(correct_width):.4f} +/- {np.std(correct_width):.4f}")
    print(f"  Incorrect: {np.mean(incorrect_width):.4f} +/- {np.std(incorrect_width):.4f}")
    print(f"  Cohen's d = {d:.3f}")
    print(f"  p-value = {p_val:.4f}")

    if p_val < 0.05:
        if d > 0:
            print(f"  => Correct has WIDER spectrum (more anisotropic)")
        else:
            print(f"  => Correct has NARROWER spectrum (more isotropic)")
    else:
        print(f"  => No significant difference")

    return d, p_val


def main():
    print("=" * 60)
    print("FULL LYAPUNOV SPECTRUM ANALYSIS")
    print("=" * 60)
    print("\nThis tests H_jac1, H_jac2, H_jac3 from the critique document.")
    print("Unlike Frobenius norm, this captures DIRECTIONAL information.\n")

    # Data paths
    base_path = "/data/thanhdo/trajectories_0shot/olmo3_base"

    n_samples = 100  # Use 100 for reasonable runtime
    k_proj = 100  # Projection dimension

    results = {}

    for task in ["humaneval", "logiqa"]:
        print("\n" + "#" * 60)
        print(f"# {task.upper()}")
        print("#" * 60)

        filepath = f"{base_path}/{task}_trajectories.h5"

        with h5py.File(filepath, 'r') as f:
            traj = f['trajectories'][:n_samples].astype(np.float32)
            labels = f['is_correct'][:n_samples]

        print(f"\nLoaded: {traj.shape}")
        print(f"Correct: {labels.sum()}/{len(labels)} ({100*labels.mean():.1f}%)")

        # Split by correctness
        correct_idx = np.where(labels)[0]
        incorrect_idx = np.where(~labels)[0]

        correct_traj = traj[correct_idx]
        incorrect_traj = traj[incorrect_idx]

        print(f"\nComputing full Lyapunov spectrum (k={k_proj})...")
        print("This may take a few minutes...")

        # Compute Lyapunov for each sample
        correct_lyap = []
        for i, t in enumerate(correct_traj):
            if i % 5 == 0:
                print(f"  Correct: {i}/{len(correct_traj)}")
            correct_lyap.append(compute_lyapunov_spectrum(t, k=k_proj))

        incorrect_lyap = []
        for i, t in enumerate(incorrect_traj):
            if i % 10 == 0:
                print(f"  Incorrect: {i}/{len(incorrect_traj)}")
            incorrect_lyap.append(compute_lyapunov_spectrum(t, k=k_proj))

        # Extract error direction
        print("\nExtracting error direction...")
        error_dir = extract_error_direction(correct_traj, incorrect_traj, layer_idx=-1)

        # Compute directional Lyapunov
        print("Computing directional Lyapunov...")
        correct_dir_lyap = [compute_directional_lyapunov(t, error_dir) for t in correct_traj]
        incorrect_dir_lyap = [compute_directional_lyapunov(t, error_dir) for t in incorrect_traj]

        # Run tests
        d1, p1 = test_h_jac1(correct_lyap, incorrect_lyap)
        d2, p2 = test_h_jac2(correct_dir_lyap, incorrect_dir_lyap)
        d3, p3 = test_h_jac3(correct_lyap, incorrect_lyap)

        results[task] = {
            'h_jac1': {'d': d1, 'p': p1},
            'h_jac2': {'d': d2, 'p': p2},
            'h_jac3': {'d': d3, 'p': p3}
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n| Task | H_jac1 (Max λ) | H_jac2 (Dir λ) | H_jac3 (Width) |")
    print("|------|----------------|----------------|----------------|")
    for task, r in results.items():
        h1 = f"d={r['h_jac1']['d']:.2f}, p={r['h_jac1']['p']:.3f}"
        h2 = f"d={r['h_jac2']['d']:.2f}, p={r['h_jac2']['p']:.3f}"
        h3 = f"d={r['h_jac3']['d']:.2f}, p={r['h_jac3']['p']:.3f}"
        print(f"| {task} | {h1} | {h2} | {h3} |")

    print("\nKey:")
    print("  H_jac1: d > 0 means incorrect is more chaotic")
    print("  H_jac2: Signal in error-direction subspace")
    print("  H_jac3: Spectrum width (anisotropy)")


if __name__ == "__main__":
    main()
