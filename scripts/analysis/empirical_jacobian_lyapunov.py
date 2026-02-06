#!/usr/bin/env python3
"""
True Empirical Jacobian Analysis - Phase 3 Extension

Computes the actual Jacobian spectrum via regression, replacing the delta-based proxy.

The delta-based approach (SVD of X_{l+1} - X_l) measures velocity, not sensitivity.
The true approach solves: X_{l+1} ≈ X_l @ J.T
The singular values of J reveal actual expansion/contraction dynamics.

Tests:
- Max singular value (max Lyapunov proxy)
- Mean singular value
- Spectrum width (std)
- Compares to delta-based proxy results

Usage:
    python empirical_jacobian_lyapunov.py \
        --data-dir /path/to/trajectories_0shot \
        --models olmo3_base,olmo3_sft,olmo3_rl_zero \
        --output-dir results
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils.extmath import randomized_svd
from sklearn.model_selection import KFold


def load_trajectories(filepath, max_samples=None):
    """Load trajectories and labels from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        trajectories = f['trajectories'][:]
        if 'is_correct' in f:
            labels = f['is_correct'][:]
        elif 'correct' in f:
            labels = f['correct'][:]
        else:
            raise KeyError(f"No correctness labels. Keys: {list(f.keys())}")

        if max_samples and max_samples < len(trajectories):
            trajectories = trajectories[:max_samples]
            labels = labels[:max_samples]

    return trajectories.astype(np.float32), labels.astype(bool)


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def empirical_jacobian_svd(X_l, X_l1, k=50, reg=1e-6):
    """
    Compute Jacobian spectrum from stored activations via regression.

    Solves: X_{l+1} ≈ X_l @ J.T
    Returns: top-k singular values of J

    Args:
        X_l:  (n_samples, d_model) - layer l activations
        X_l1: (n_samples, d_model) - layer l+1 activations
        k: number of singular values to return
        reg: regularization for pseudo-inverse

    Returns:
        s_J: top-k singular values of the Jacobian
    """
    # Center both
    X_l_c = X_l - X_l.mean(axis=0)
    X_l1_c = X_l1 - X_l1.mean(axis=0)

    # Use randomized SVD for efficiency
    n_components = min(k, X_l_c.shape[0] - 1, X_l_c.shape[1])
    if n_components < 1:
        return np.ones(k)

    try:
        U, s, Vt = randomized_svd(X_l_c, n_components=n_components, n_iter=3, random_state=42)
    except Exception:
        return np.ones(k)

    # Regularized pseudo-inverse: s_inv = s / (s² + reg)
    s_inv = s / (s**2 + reg)

    # J.T = pinv(X_l_c) @ X_l1_c
    # J = X_l1_c.T @ U @ diag(s_inv) @ Vt
    # But we only need singular values of J

    # Compute J @ V.T = X_l1_c.T @ U @ diag(s_inv)
    JVt = (X_l1_c.T @ U) * s_inv  # (d_model, rank)

    # SVD of JVt gives same singular values as J
    try:
        _, s_J, _ = randomized_svd(JVt, n_components=min(k, JVt.shape[0]-1, JVt.shape[1]-1),
                                    n_iter=2, random_state=42)
    except Exception:
        return np.ones(k)

    # Pad if needed
    if len(s_J) < k:
        s_J = np.concatenate([s_J, np.zeros(k - len(s_J))])

    return s_J[:k]


def delta_based_svd(X_l, X_l1, k=50):
    """
    Current proxy: SVD of delta = X_{l+1} - X_l.

    This is what h3_remaining_analyses.py computes.
    Included for comparison.
    """
    delta = X_l1 - X_l

    try:
        _, s, _ = randomized_svd(delta, n_components=min(k, delta.shape[0]-1, delta.shape[1]),
                                  n_iter=2, random_state=42)
        # Normalize by input scale
        input_scale = np.linalg.norm(X_l, 'fro') / np.sqrt(X_l.shape[0])
        if input_scale > 1e-10:
            return s / input_scale
        return s
    except:
        return np.ones(k)


def compute_per_sample_jacobian_stats(trajectories, labels, k=50, method='true'):
    """
    Compute Jacobian statistics for each sample.

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean
        k: number of singular values
        method: 'true' for regression-based, 'delta' for velocity-based

    Returns:
        DataFrame with per-sample statistics
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    results = []
    for i in range(n_samples):
        traj = trajectories[i]  # (seq_len, n_layers, d_model)

        layer_max_sv = []
        layer_mean_sv = []
        layer_std_sv = []

        for l in range(n_layers - 1):
            X_l = traj[:, l, :]   # (seq_len, d_model)
            X_l1 = traj[:, l + 1, :]  # (seq_len, d_model)

            if method == 'true':
                s = empirical_jacobian_svd(X_l, X_l1, k=k)
            else:
                s = delta_based_svd(X_l, X_l1, k=k)

            # Log-transform for Lyapunov interpretation
            lyap = np.log(s + 1e-10)

            layer_max_sv.append(lyap[0])  # Max Lyapunov
            layer_mean_sv.append(np.mean(lyap))
            layer_std_sv.append(np.std(lyap))

        results.append({
            'sample_idx': i,
            'is_correct': bool(labels[i]),
            'max_lyapunov': float(np.mean(layer_max_sv)),
            'mean_lyapunov': float(np.mean(layer_mean_sv)),
            'spectrum_width': float(np.mean(layer_std_sv))
        })

    return pd.DataFrame(results)


def compute_per_layer_jacobian_stats(trajectories, labels, k=50, method='true'):
    """
    Compute Jacobian statistics per layer, pooling across all samples.

    This gives a layer-by-layer view of dynamics.

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean
        k: number of singular values
        method: 'true' or 'delta'

    Returns:
        DataFrame with per-layer statistics
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    results = []
    for l in range(n_layers - 1):
        # Pool activations from all samples at this layer
        X_l_all = trajectories[:, :, l, :].reshape(-1, d_model)  # (n_samples*seq_len, d_model)
        X_l1_all = trajectories[:, :, l + 1, :].reshape(-1, d_model)

        # Get indices for correct/incorrect
        correct_mask = np.repeat(labels, seq_len)
        incorrect_mask = ~correct_mask

        # Compute for correct samples
        X_l_correct = X_l_all[correct_mask]
        X_l1_correct = X_l1_all[correct_mask]

        if method == 'true':
            s_correct = empirical_jacobian_svd(X_l_correct, X_l1_correct, k=k)
        else:
            s_correct = delta_based_svd(X_l_correct, X_l1_correct, k=k)

        lyap_correct = np.log(s_correct + 1e-10)

        # Compute for incorrect samples
        X_l_incorrect = X_l_all[incorrect_mask]
        X_l1_incorrect = X_l1_all[incorrect_mask]

        if method == 'true':
            s_incorrect = empirical_jacobian_svd(X_l_incorrect, X_l1_incorrect, k=k)
        else:
            s_incorrect = delta_based_svd(X_l_incorrect, X_l1_incorrect, k=k)

        lyap_incorrect = np.log(s_incorrect + 1e-10)

        results.append({
            'layer': l,
            'correct_max_lyap': float(lyap_correct[0]),
            'correct_mean_lyap': float(np.mean(lyap_correct)),
            'incorrect_max_lyap': float(lyap_incorrect[0]),
            'incorrect_mean_lyap': float(np.mean(lyap_incorrect)),
            'max_lyap_diff': float(lyap_correct[0] - lyap_incorrect[0]),
            'mean_lyap_diff': float(np.mean(lyap_correct) - np.mean(lyap_incorrect))
        })

    return pd.DataFrame(results)


def run_analysis(data_dir, models, output_dir, max_samples=100):
    """Run empirical Jacobian analysis for all model/task combinations."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ['gsm8k', 'humaneval', 'logiqa']
    all_results = []
    layer_results = []

    # Load existing results if any (for incremental saves)
    output_path = output_dir / 'h2_true_jacobian.csv'
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        all_results = existing_df.to_dict('records')
        completed = set((r['model'], r['task']) for r in all_results)
        print(f"  Loaded {len(all_results)} existing results")
    else:
        completed = set()

    for model in models:
        for task in tasks:
            # Skip if already completed
            if (model, task) in completed:
                print(f"  SKIP: {model}/{task} - already completed")
                continue
            h5_path = data_dir / model / f"{task}_trajectories.h5"

            if not h5_path.exists():
                print(f"  SKIP: {model}/{task} - file not found")
                continue

            print(f"\n  Processing {model}/{task}...")

            try:
                trajectories, labels = load_trajectories(h5_path, max_samples)
            except Exception as e:
                print(f"    ERROR loading: {e}")
                continue

            n_correct = labels.sum()
            n_incorrect = (~labels).sum()
            print(f"    Loaded {len(labels)} samples: {n_correct} correct, {n_incorrect} incorrect")

            if n_correct < 5 or n_incorrect < 5:
                print(f"    SKIP: insufficient samples")
                continue

            # === Per-sample analysis with TRUE Jacobian ===
            print("    Computing true Jacobian statistics...")
            df_true = compute_per_sample_jacobian_stats(trajectories, labels, method='true')

            correct_vals = df_true[df_true['is_correct']]
            incorrect_vals = df_true[~df_true['is_correct']]

            d_max = cohens_d(correct_vals['max_lyapunov'].values, incorrect_vals['max_lyapunov'].values)
            d_mean = cohens_d(correct_vals['mean_lyapunov'].values, incorrect_vals['mean_lyapunov'].values)
            d_width = cohens_d(correct_vals['spectrum_width'].values, incorrect_vals['spectrum_width'].values)

            _, p_max = stats.ttest_ind(correct_vals['max_lyapunov'].values, incorrect_vals['max_lyapunov'].values)
            _, p_mean = stats.ttest_ind(correct_vals['mean_lyapunov'].values, incorrect_vals['mean_lyapunov'].values)
            _, p_width = stats.ttest_ind(correct_vals['spectrum_width'].values, incorrect_vals['spectrum_width'].values)

            # === Per-sample analysis with DELTA proxy (for comparison) ===
            print("    Computing delta-based statistics (comparison)...")
            df_delta = compute_per_sample_jacobian_stats(trajectories, labels, method='delta')

            correct_vals_d = df_delta[df_delta['is_correct']]
            incorrect_vals_d = df_delta[~df_delta['is_correct']]

            d_max_delta = cohens_d(correct_vals_d['max_lyapunov'].values, incorrect_vals_d['max_lyapunov'].values)

            all_results.append({
                'model': model,
                'task': task,
                'method': 'true_jacobian',
                'max_lyap_d': float(d_max),
                'max_lyap_p': float(p_max),
                'mean_lyap_d': float(d_mean),
                'mean_lyap_p': float(p_mean),
                'width_d': float(d_width),
                'width_p': float(p_width),
                'delta_max_lyap_d': float(d_max_delta),  # For comparison
                'n_correct': int(n_correct),
                'n_incorrect': int(n_incorrect)
            })

            print(f"    True Jacobian: max_d={d_max:.3f} (p={p_max:.3f}), mean_d={d_mean:.3f}")
            print(f"    Delta proxy:   max_d={d_max_delta:.3f}")

            # === Per-layer analysis ===
            print("    Computing per-layer statistics...")
            df_layer = compute_per_layer_jacobian_stats(trajectories, labels, method='true')
            df_layer['model'] = model
            df_layer['task'] = task
            layer_results.append(df_layer)

            # Save incrementally after each model/task
            df_main = pd.DataFrame(all_results)
            df_main.to_csv(output_dir / 'h2_true_jacobian.csv', index=False)
            print(f"    Saved incremental results ({len(all_results)} total)")

    # Save final results
    if all_results:
        df_main = pd.DataFrame(all_results)
        output_path = output_dir / 'h2_true_jacobian.csv'
        df_main.to_csv(output_path, index=False)
        print(f"\n  Saved main results to {output_path}")

        print("\n  === True Jacobian Results ===")
        print(df_main[['model', 'task', 'max_lyap_d', 'max_lyap_p', 'delta_max_lyap_d']].to_string(index=False))

    if layer_results:
        df_layers = pd.concat(layer_results, ignore_index=True)
        layer_path = output_dir / 'h2_true_jacobian_layers.csv'
        df_layers.to_csv(layer_path, index=False)
        print(f"\n  Saved layer results to {layer_path}")


def main():
    parser = argparse.ArgumentParser(description='True Empirical Jacobian Analysis')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing trajectory HDF5 files')
    parser.add_argument('--models', type=str, default='olmo3_base,olmo3_sft,olmo3_rl_zero',
                        help='Comma-separated list of models')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Maximum samples per task')

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]

    print(f"\n{'='*60}")
    print("True Empirical Jacobian Analysis")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Models: {models}")
    print(f"{'='*60}")
    print()
    print("Key difference from delta-based proxy:")
    print("  Delta: SVD(X_{l+1} - X_l) = velocity magnitude")
    print("  True:  SVD(J) where X_{l+1} ≈ X_l @ J.T = sensitivity")
    print(f"{'='*60}")

    run_analysis(
        data_dir=args.data_dir,
        models=models,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
