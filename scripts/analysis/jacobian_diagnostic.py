#!/usr/bin/env python3
"""
Diagnostic script to investigate why delta-based and true Jacobian differ.

Hypothesis: The delta measures displacement magnitude, while the Jacobian measures
sensitivity to perturbations. If the layer transformation is approximately an
orthogonal rotation (or subspace reshuffling), the Jacobian will have SVs ≈ 1
(no expansion/contraction) while delta can still be large.

This script computes:
1. Jacobian condition number (max_sv / min_sv) - how "stretched" is the transform
2. Jacobian spectral gap (max_sv - min_sv) - how anisotropic
3. Mean singular value of J - should be ≈ 1 if orthogonal
4. Comparison: ||delta||_F vs mean(SV(J))
5. Orthogonality measure: ||J @ J.T - I||_F / d

Usage:
    python jacobian_diagnostic.py --data-dir /path/to/trajectories_0shot --model olmo3_rl_zero --task gsm8k
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
from sklearn.utils.extmath import randomized_svd


def load_trajectories(filepath, max_samples=50):
    """Load trajectories and labels."""
    with h5py.File(filepath, 'r') as f:
        trajectories = f['trajectories'][:max_samples]
        if 'is_correct' in f:
            labels = f['is_correct'][:max_samples]
        else:
            labels = f['correct'][:max_samples]
    return trajectories.astype(np.float32), labels.astype(bool)


def empirical_jacobian_full(X_l, X_l1, reg=1e-6):
    """
    Compute full Jacobian matrix (not just SVD) for diagnostic purposes.
    J.T = pinv(X_l) @ X_l1
    """
    X_l_c = X_l - X_l.mean(axis=0)
    X_l1_c = X_l1 - X_l1.mean(axis=0)

    # Regularized pseudo-inverse
    U, s, Vt = np.linalg.svd(X_l_c, full_matrices=False)
    s_inv = s / (s**2 + reg)

    # J.T = V @ diag(s_inv) @ U.T @ X_l1_c
    # J = X_l1_c.T @ U @ diag(s_inv) @ Vt
    JT = Vt.T @ np.diag(s_inv) @ U.T @ X_l1_c
    J = JT.T

    return J


def analyze_layer_transition(X_l, X_l1, k=50):
    """Analyze a single layer transition."""
    n_samples, d_model = X_l.shape

    # 1. Delta statistics
    delta = X_l1 - X_l
    delta_frob = np.linalg.norm(delta, 'fro')
    delta_per_sample = np.linalg.norm(delta, axis=1).mean()

    # 2. Input/output norms
    X_l_norm = np.linalg.norm(X_l, 'fro') / np.sqrt(n_samples)
    X_l1_norm = np.linalg.norm(X_l1, 'fro') / np.sqrt(n_samples)

    # 3. Compute Jacobian (use subset for speed)
    n_sub = min(500, n_samples)  # Use at most 500 samples
    J = empirical_jacobian_full(X_l[:n_sub], X_l1[:n_sub])

    # 4. Jacobian SVD
    try:
        U, s_J, Vt = randomized_svd(J, n_components=min(k, d_model-1), n_iter=3, random_state=42)
    except:
        return None

    # 5. Jacobian statistics
    max_sv = s_J[0]
    min_sv = s_J[-1] if len(s_J) > 1 else s_J[0]
    mean_sv = s_J.mean()
    median_sv = np.median(s_J)
    condition_number = max_sv / (min_sv + 1e-10)
    spectral_gap = max_sv - min_sv

    # 6. Orthogonality measure (approximate - use subset of J)
    # If J is orthogonal, J @ J.T = I
    # Compute ||J @ J.T - I||_F / sqrt(d) as orthogonality error
    J_sub = J[:min(100, J.shape[0]), :min(100, J.shape[1])]  # Use 100x100 submatrix
    JJT = J_sub @ J_sub.T
    d_sub = JJT.shape[0]
    ortho_error = np.linalg.norm(JJT - np.eye(d_sub), 'fro') / np.sqrt(d_sub)

    # 7. Cosine similarity between X_l and X_l1 (per sample)
    X_l_normed = X_l / (np.linalg.norm(X_l, axis=1, keepdims=True) + 1e-10)
    X_l1_normed = X_l1 / (np.linalg.norm(X_l1, axis=1, keepdims=True) + 1e-10)
    cos_sim = (X_l_normed * X_l1_normed).sum(axis=1).mean()

    return {
        'delta_frob': float(delta_frob),
        'delta_per_sample': float(delta_per_sample),
        'X_l_norm': float(X_l_norm),
        'X_l1_norm': float(X_l1_norm),
        'norm_ratio': float(X_l1_norm / (X_l_norm + 1e-10)),
        'max_sv': float(max_sv),
        'min_sv': float(min_sv),
        'mean_sv': float(mean_sv),
        'median_sv': float(median_sv),
        'condition_number': float(condition_number),
        'spectral_gap': float(spectral_gap),
        'ortho_error': float(ortho_error),
        'cos_sim': float(cos_sim),
        # Key diagnostic: delta normalized by input norm vs mean SV
        'delta_normalized': float(delta_per_sample / (X_l_norm + 1e-10)),
        'lyapunov_proxy_delta': float(np.log(delta_per_sample / (X_l_norm + 1e-10) + 1e-10)),
        'lyapunov_true': float(np.log(mean_sv + 1e-10))
    }


def run_diagnostic(data_dir, model, task, max_samples=50):
    """Run diagnostic analysis."""
    data_dir = Path(data_dir)
    h5_path = data_dir / model / f"{task}_trajectories.h5"

    if not h5_path.exists():
        print(f"File not found: {h5_path}")
        return

    print(f"\n{'='*70}")
    print(f"Jacobian Diagnostic: {model}/{task}")
    print(f"{'='*70}")

    trajectories, labels = load_trajectories(h5_path, max_samples)
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    print(f"Loaded {n_samples} samples, {n_layers} layers, d={d_model}")
    print(f"Correct: {labels.sum()}, Incorrect: {(~labels).sum()}")

    # Analyze each layer transition
    print(f"\n{'Layer':<6} {'delta/||X|}':<12} {'mean_SV':<10} {'ortho_err':<12} {'cos_sim':<10} {'cond_num':<12}")
    print("-" * 70)

    results_correct = []
    results_incorrect = []

    for l in range(n_layers - 1):
        # Pool all samples at this layer
        X_l = trajectories[:, :, l, :].reshape(-1, d_model)
        X_l1 = trajectories[:, :, l + 1, :].reshape(-1, d_model)

        result = analyze_layer_transition(X_l, X_l1)
        if result:
            print(f"L{l:<5} {result['delta_normalized']:<12.4f} {result['mean_sv']:<10.4f} "
                  f"{result['ortho_error']:<12.4f} {result['cos_sim']:<10.4f} {result['condition_number']:<12.1f}")

        # Also analyze separately for correct/incorrect
        correct_mask = np.repeat(labels, seq_len)
        X_l_correct = X_l[correct_mask]
        X_l1_correct = X_l1[correct_mask]
        X_l_incorrect = X_l[~correct_mask]
        X_l1_incorrect = X_l1[~correct_mask]

        if len(X_l_correct) > 10:
            res_c = analyze_layer_transition(X_l_correct, X_l1_correct)
            if res_c:
                results_correct.append(res_c)

        if len(X_l_incorrect) > 10:
            res_i = analyze_layer_transition(X_l_incorrect, X_l1_incorrect)
            if res_i:
                results_incorrect.append(res_i)

    # Summary statistics
    if results_correct and results_incorrect:
        print(f"\n{'='*70}")
        print("Summary: Correct vs Incorrect")
        print(f"{'='*70}")

        metrics = ['delta_normalized', 'mean_sv', 'ortho_error', 'cos_sim']
        for metric in metrics:
            c_vals = [r[metric] for r in results_correct]
            i_vals = [r[metric] for r in results_incorrect]
            c_mean, i_mean = np.mean(c_vals), np.mean(i_vals)
            diff = c_mean - i_mean
            print(f"{metric:<20}: Correct={c_mean:.4f}, Incorrect={i_mean:.4f}, Diff={diff:+.4f}")

    # Key interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("""
If mean_sv ≈ 1 and ortho_error is small:
  → The transformation is approximately orthogonal (rotation/reflection)
  → Jacobian SVs ≈ 1 means no expansion/contraction (Lyapunov ≈ 0)
  → But delta can still be large because rotation moves points

If delta_normalized >> mean_sv:
  → Delta measures displacement, not sensitivity
  → The proxy was misleading: large delta ≠ large Jacobian eigenvalues
  → This happens when the transformation "reshuffles" the subspace

Key insight: Delta-based Lyapunov proxy is INVALID when transformations are
approximately orthogonal rotations, which appears to be the case in transformers.
""")


def main():
    parser = argparse.ArgumentParser(description='Jacobian Diagnostic Analysis')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='olmo3_rl_zero')
    parser.add_argument('--task', type=str, default='gsm8k')
    parser.add_argument('--max-samples', type=int, default=50)

    args = parser.parse_args()
    run_diagnostic(args.data_dir, args.model, args.task, args.max_samples)


if __name__ == '__main__':
    main()
