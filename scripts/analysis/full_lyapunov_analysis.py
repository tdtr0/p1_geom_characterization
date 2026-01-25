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


def compute_layer_jacobian_svd(x_l, x_l1, k=50):
    """
    Estimate Jacobian spectrum between two layers using randomized SVD.

    Uses a faster approximation: compute SVD of the transition matrix.
    This captures how directions expand/contract between layers.

    Args:
        x_l: (seq_len, d_model) - activations at layer l
        x_l1: (seq_len, d_model) - activations at layer l+1
        k: number of singular values to compute

    Returns:
        singular_values: (k,) - approximated expansion factors
    """
    from sklearn.utils.extmath import randomized_svd

    # Compute delta (layer transition)
    delta = x_l1 - x_l  # (seq_len, d_model)

    # Use randomized SVD for efficiency - compute on delta's covariance proxy
    # This gives us the principal expansion directions
    try:
        # Randomized SVD is O(seq_len * d_model * k) instead of O(d^3)
        U, s, Vt = randomized_svd(delta, n_components=min(k, delta.shape[0]-1, delta.shape[1]),
                                   n_iter=2, random_state=42)

        # Normalize by input magnitude to get expansion ratios
        input_scale = np.linalg.norm(x_l, 'fro') / np.sqrt(x_l.shape[0])
        if input_scale > 1e-10:
            expansion_ratios = s / input_scale
        else:
            expansion_ratios = s

        return expansion_ratios
    except:
        return np.ones(k)


def compute_lyapunov_spectrum(trajectory, k=50):
    """
    Compute full Lyapunov spectrum for a trajectory.

    Args:
        trajectory: (seq_len, n_layers, d_model)
        k: number of singular values to compute

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

        if len(sv) > 0:
            # Lyapunov exponents = log of expansion ratios
            lyap = np.log(sv + 1e-10)

            layer_max_lyapunov.append(lyap[0])  # Largest
            layer_mean_lyapunov.append(np.mean(lyap))
            layer_spectrum_width.append(np.std(lyap))

    if not layer_max_lyapunov:
        return {
            'max_lyapunov': 0.0,
            'mean_lyapunov': 0.0,
            'spectrum_width': 0.0,
            'max_lyapunov_profile': np.array([]),
            'mean_lyapunov_profile': np.array([])
        }

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


def analyze_task(filepath, task_name, n_samples=100, k_proj=100):
    """Analyze a single task and return results."""
    print(f"\n{'#' * 60}")
    print(f"# {task_name}")
    print('#' * 60)

    try:
        with h5py.File(filepath, 'r') as f:
            traj = f['trajectories'][:n_samples].astype(np.float32)
            labels = f['is_correct'][:n_samples]
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return None

    print(f"\nLoaded: {traj.shape}")
    print(f"Correct: {labels.sum()}/{len(labels)} ({100*labels.mean():.1f}%)")

    # Split by correctness
    correct_idx = np.where(labels)[0]
    incorrect_idx = np.where(~labels)[0]

    if len(correct_idx) < 5 or len(incorrect_idx) < 5:
        print(f"SKIPPED: Not enough samples (correct={len(correct_idx)}, incorrect={len(incorrect_idx)})")
        return None

    correct_traj = traj[correct_idx]
    incorrect_traj = traj[incorrect_idx]

    print(f"\nComputing full Lyapunov spectrum (k={k_proj})...")

    # Compute Lyapunov for each sample
    correct_lyap = []
    for i, t in enumerate(correct_traj):
        if i % 10 == 0:
            print(f"  Correct: {i}/{len(correct_traj)}")
        correct_lyap.append(compute_lyapunov_spectrum(t, k=k_proj))

    incorrect_lyap = []
    for i, t in enumerate(incorrect_traj):
        if i % 20 == 0:
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

    return {
        'h_jac1': {'d': d1, 'p': p1},
        'h_jac2': {'d': d2, 'p': p2},
        'h_jac3': {'d': d3, 'p': p3}
    }


def main():
    print("=" * 60)
    print("FULL LYAPUNOV SPECTRUM ANALYSIS")
    print("=" * 60)
    print("\nThis tests H_jac1, H_jac2, H_jac3 from the critique document.")
    print("Unlike Frobenius norm, this captures DIRECTIONAL information.\n")

    # Data configuration
    base_dir = "/data/thanhdo/trajectories_0shot"
    models = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']
    tasks = ['gsm8k', 'humaneval', 'logiqa']

    n_samples = 50  # Reduced for faster runtime
    k_proj = 50  # Reduced projection dimension (was 100)

    all_results = {}

    for model in models:
        print("\n" + "=" * 60)
        print(f"MODEL: {model}")
        print("=" * 60)

        model_results = {}
        for task in tasks:
            filepath = f"{base_dir}/{model}/{task}_trajectories.h5"
            task_key = f"{model}/{task}"

            result = analyze_task(filepath, task_key, n_samples, k_proj)
            if result is not None:
                model_results[task] = result
                all_results[task_key] = result

        if not model_results:
            print(f"No valid data for {model}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n| Model/Task | H_jac1 (Max λ) | H_jac2 (Dir λ) | H_jac3 (Width) |")
    print("|------------|----------------|----------------|----------------|")
    for key, r in sorted(all_results.items()):
        h1 = f"d={r['h_jac1']['d']:.2f}, p={r['h_jac1']['p']:.3f}"
        h2 = f"d={r['h_jac2']['d']:.2f}, p={r['h_jac2']['p']:.3f}"
        h3 = f"d={r['h_jac3']['d']:.2f}, p={r['h_jac3']['p']:.3f}"
        print(f"| {key} | {h1} | {h2} | {h3} |")

    print("\nKey:")
    print("  H_jac1: d > 0 means incorrect is more chaotic")
    print("  H_jac2: Signal in error-direction subspace")
    print("  H_jac3: Spectrum width (anisotropy)")
    print("  p < 0.05 = significant")

    # Count significant results
    sig_count = sum(1 for r in all_results.values()
                    for h in ['h_jac1', 'h_jac2', 'h_jac3']
                    if r[h]['p'] < 0.05)
    total = len(all_results) * 3
    print(f"\nSignificant results: {sig_count}/{total}")

    # Save results
    import json
    output_path = "/home/thanhdo/p1_geom_characterization/results/full_lyapunov_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
