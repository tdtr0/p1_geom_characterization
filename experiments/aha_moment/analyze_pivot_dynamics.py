#!/usr/bin/env python3
"""
Analyze trajectory dynamics at pivot points (aha moments).

This script compares trajectory behavior at pivot tokens vs random tokens,
testing whether self-correction points show phase transition signatures.

Usage:
    python analyze_pivot_dynamics.py \
        --thinking data/thinking_traces/deepseek_r1_decimal.h5 \
        --output results/

Metrics computed:
1. Velocity magnitude at pivot vs surrounding tokens
2. Direction change (cosine similarity) at pivot
3. Local Lyapunov estimate around pivot
4. Layer-wise divergence pattern
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


# ============================================================================
# Trajectory Metrics
# ============================================================================

def compute_velocity(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute layer transition velocity at each token position.

    Args:
        trajectory: (seq_len, n_layers, hidden_dim)

    Returns:
        velocity_magnitude: (seq_len, n_layers-1) - magnitude of h_{l+1} - h_l
    """
    # Layer-to-layer differences
    layer_diff = trajectory[:, 1:, :] - trajectory[:, :-1, :]  # (seq_len, n_layers-1, hidden)
    velocity_mag = np.linalg.norm(layer_diff, axis=-1)  # (seq_len, n_layers-1)
    return velocity_mag


def compute_direction_change(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute direction change (1 - cosine similarity) between consecutive token positions.

    Args:
        trajectory: (seq_len, n_layers, hidden_dim)

    Returns:
        direction_change: (seq_len-1, n_layers) - how much direction changes token-to-token
    """
    # Token-to-token differences at each layer
    token_diff = trajectory[1:, :, :] - trajectory[:-1, :, :]  # (seq_len-1, n_layers, hidden)

    # Compute cosine similarity between consecutive differences
    # diff[t] vs diff[t+1]
    direction_changes = []
    for t in range(token_diff.shape[0] - 1):
        v1 = token_diff[t]  # (n_layers, hidden)
        v2 = token_diff[t + 1]  # (n_layers, hidden)

        # Cosine similarity per layer
        norm1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8
        norm2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8
        cos_sim = np.sum(v1 * v2, axis=-1) / (norm1.squeeze() * norm2.squeeze())
        direction_change = 1 - cos_sim  # High = more change
        direction_changes.append(direction_change)

    return np.array(direction_changes)  # (seq_len-2, n_layers)


def compute_local_lyapunov(trajectory: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Estimate local Lyapunov exponent around each token position.

    Uses singular value ratio of local layer transitions as proxy.

    Args:
        trajectory: (seq_len, n_layers, hidden_dim)
        window: Number of tokens to use for local estimate

    Returns:
        lyapunov: (seq_len - window + 1,) - local stability estimate
    """
    seq_len, n_layers, hidden_dim = trajectory.shape
    lyapunov = []

    for t in range(seq_len - window + 1):
        window_traj = trajectory[t:t + window]  # (window, n_layers, hidden)

        # Compute layer-to-layer expansion across window
        expansions = []
        for l in range(n_layers - 1):
            x_l = window_traj[:, l, :]  # (window, hidden)
            x_l1 = window_traj[:, l + 1, :]  # (window, hidden)

            # SVD of transition
            delta = x_l1 - x_l
            _, s, _ = np.linalg.svd(delta, full_matrices=False)

            # Log ratio of top to bottom singular values
            expansion = np.log(s[0] / (s[-1] + 1e-8))
            expansions.append(expansion)

        lyapunov.append(np.mean(expansions))

    return np.array(lyapunov)


def compute_token_velocity_magnitude(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute total velocity magnitude at each token (summed across layers).

    Args:
        trajectory: (seq_len, n_layers, hidden_dim)

    Returns:
        total_velocity: (seq_len-1,) - sum of layer velocities at each token transition
    """
    # Token-to-token difference
    token_diff = trajectory[1:, :, :] - trajectory[:-1, :, :]  # (seq_len-1, n_layers, hidden)
    token_velocity = np.linalg.norm(token_diff, axis=-1).sum(axis=-1)  # (seq_len-1,)
    return token_velocity


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_pivot_points(
    trajectory: np.ndarray,
    pivot_indices: list[int],
    window: int = 5,
) -> dict:
    """
    Analyze trajectory metrics at pivot points vs random positions.

    Args:
        trajectory: (seq_len, n_layers, hidden_dim)
        pivot_indices: List of token indices where pivots occur
        window: Window size for context around pivot

    Returns:
        Dictionary of metrics
    """
    seq_len = trajectory.shape[0]

    # Compute metrics
    token_velocity = compute_token_velocity_magnitude(trajectory)
    direction_change = compute_direction_change(trajectory)
    lyapunov = compute_local_lyapunov(trajectory, window=window)

    results = {
        'pivot_indices': pivot_indices,
        'pivot_velocities': [],
        'pivot_direction_changes': [],
        'pivot_lyapunov': [],
        'random_velocities': [],
        'random_direction_changes': [],
        'random_lyapunov': [],
    }

    # Get metrics at pivot points
    for idx in pivot_indices:
        if 0 < idx < len(token_velocity):
            results['pivot_velocities'].append(token_velocity[idx - 1])
        if 0 < idx < len(direction_change) + 1:
            results['pivot_direction_changes'].append(direction_change[idx - 1].mean())
        if window // 2 < idx < len(lyapunov) + window // 2:
            results['pivot_lyapunov'].append(lyapunov[idx - window // 2])

    # Get metrics at random positions (excluding pivots and boundaries)
    valid_indices = set(range(window, seq_len - window)) - set(pivot_indices)
    valid_indices = list(valid_indices)

    if len(valid_indices) > 0:
        # Sample same number as pivots, or all if fewer
        n_random = min(len(valid_indices), max(len(pivot_indices) * 3, 10))
        random_indices = np.random.choice(valid_indices, n_random, replace=False)

        for idx in random_indices:
            if 0 < idx < len(token_velocity):
                results['random_velocities'].append(token_velocity[idx - 1])
            if 0 < idx < len(direction_change) + 1:
                results['random_direction_changes'].append(direction_change[idx - 1].mean())
            if window // 2 < idx < len(lyapunov) + window // 2:
                results['random_lyapunov'].append(lyapunov[idx - window // 2])

    return results


def run_statistical_tests(all_results: list[dict]) -> dict:
    """
    Run statistical tests comparing pivot vs random positions.

    Args:
        all_results: List of per-sample analysis results

    Returns:
        Dictionary of test results
    """
    # Aggregate across samples
    all_pivot_vel = []
    all_random_vel = []
    all_pivot_dir = []
    all_random_dir = []
    all_pivot_lyap = []
    all_random_lyap = []

    for r in all_results:
        all_pivot_vel.extend(r['pivot_velocities'])
        all_random_vel.extend(r['random_velocities'])
        all_pivot_dir.extend(r['pivot_direction_changes'])
        all_random_dir.extend(r['random_direction_changes'])
        all_pivot_lyap.extend(r['pivot_lyapunov'])
        all_random_lyap.extend(r['random_lyapunov'])

    tests = {}

    # Velocity test
    if len(all_pivot_vel) > 1 and len(all_random_vel) > 1:
        t_stat, p_val = stats.ttest_ind(all_pivot_vel, all_random_vel)
        effect_size = (np.mean(all_pivot_vel) - np.mean(all_random_vel)) / np.std(all_random_vel)
        tests['velocity'] = {
            'pivot_mean': np.mean(all_pivot_vel),
            'pivot_std': np.std(all_pivot_vel),
            'random_mean': np.mean(all_random_vel),
            'random_std': np.std(all_random_vel),
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size_d': effect_size,
            'n_pivot': len(all_pivot_vel),
            'n_random': len(all_random_vel),
        }

    # Direction change test
    if len(all_pivot_dir) > 1 and len(all_random_dir) > 1:
        t_stat, p_val = stats.ttest_ind(all_pivot_dir, all_random_dir)
        effect_size = (np.mean(all_pivot_dir) - np.mean(all_random_dir)) / np.std(all_random_dir)
        tests['direction_change'] = {
            'pivot_mean': np.mean(all_pivot_dir),
            'pivot_std': np.std(all_pivot_dir),
            'random_mean': np.mean(all_random_dir),
            'random_std': np.std(all_random_dir),
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size_d': effect_size,
            'n_pivot': len(all_pivot_dir),
            'n_random': len(all_random_dir),
        }

    # Lyapunov test
    if len(all_pivot_lyap) > 1 and len(all_random_lyap) > 1:
        t_stat, p_val = stats.ttest_ind(all_pivot_lyap, all_random_lyap)
        effect_size = (np.mean(all_pivot_lyap) - np.mean(all_random_lyap)) / np.std(all_random_lyap)
        tests['lyapunov'] = {
            'pivot_mean': np.mean(all_pivot_lyap),
            'pivot_std': np.std(all_pivot_lyap),
            'random_mean': np.mean(all_random_lyap),
            'random_std': np.std(all_random_lyap),
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size_d': effect_size,
            'n_pivot': len(all_pivot_lyap),
            'n_random': len(all_random_lyap),
        }

    return tests


# ============================================================================
# Visualization
# ============================================================================

def plot_pivot_comparison(tests: dict, output_dir: str):
    """Create comparison plots for pivot vs random positions."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['velocity', 'direction_change', 'lyapunov']
    titles = ['Token Velocity', 'Direction Change', 'Local Lyapunov']

    for ax, metric, title in zip(axes, metrics, titles):
        if metric in tests:
            t = tests[metric]
            x = ['Pivot', 'Random']
            means = [t['pivot_mean'], t['random_mean']]
            stds = [t['pivot_std'], t['random_std']]

            bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#e74c3c', '#3498db'])
            ax.set_title(f'{title}\np={t["p_value"]:.4f}, d={t["effect_size_d"]:.2f}')
            ax.set_ylabel('Value')

            # Add significance stars
            if t['p_value'] < 0.001:
                sig = '***'
            elif t['p_value'] < 0.01:
                sig = '**'
            elif t['p_value'] < 0.05:
                sig = '*'
            else:
                sig = 'n.s.'

            ax.annotate(sig, xy=(0.5, max(means) * 1.1), ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'pivot_comparison.png', dpi=150)
    plt.close()


def plot_trajectory_example(
    trajectory: np.ndarray,
    pivot_indices: list[int],
    output_path: str,
):
    """Plot example trajectory with pivot points marked."""
    token_velocity = compute_token_velocity_magnitude(trajectory)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(token_velocity, 'b-', alpha=0.7, label='Token velocity')

    for idx in pivot_indices:
        if 0 < idx < len(token_velocity):
            ax.axvline(idx, color='r', linestyle='--', alpha=0.7)
            ax.scatter([idx - 1], [token_velocity[idx - 1]], color='r', s=100, zorder=5)

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Velocity Magnitude')
    ax.set_title('Trajectory Velocity with Pivot Points')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze pivot point dynamics')
    parser.add_argument('--thinking', type=str, required=True,
                        help='Path to thinking traces HDF5 file')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    parser.add_argument('--window', type=int, default=5,
                        help='Window size for local analysis')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading traces from: {args.thinking}")

    all_results = []

    with h5py.File(args.thinking, 'r') as f:
        n_samples = f.attrs['n_samples']

        for i in tqdm(range(n_samples), desc="Analyzing"):
            # Load trajectory
            if f'trajectories/sample_{i}' not in f:
                continue

            trajectory = f[f'trajectories/sample_{i}'][:]
            meta = json.loads(f[f'metadata/sample_{i}'][()])

            # Get pivot indices
            pivot_indices = [p['token_idx'] for p in meta['pivot_tokens']]

            if not pivot_indices:
                continue

            # Analyze
            result = analyze_pivot_points(trajectory, pivot_indices, args.window)
            result['sample_id'] = i
            result['is_correct'] = meta['is_correct']
            all_results.append(result)

            # Plot first example
            if i == 0:
                plot_trajectory_example(
                    trajectory,
                    pivot_indices,
                    str(Path(args.output) / 'example_trajectory.png')
                )

    print(f"\nAnalyzed {len(all_results)} samples with pivot tokens")

    # Statistical tests
    tests = run_statistical_tests(all_results)

    # Print results
    print("\n=== Statistical Tests ===")
    for metric, t in tests.items():
        print(f"\n{metric.upper()}:")
        print(f"  Pivot:  {t['pivot_mean']:.4f} +/- {t['pivot_std']:.4f} (n={t['n_pivot']})")
        print(f"  Random: {t['random_mean']:.4f} +/- {t['random_std']:.4f} (n={t['n_random']})")
        print(f"  t={t['t_statistic']:.3f}, p={t['p_value']:.4f}, d={t['effect_size_d']:.3f}")
        if t['p_value'] < 0.05:
            print(f"  --> SIGNIFICANT (p < 0.05)")

    # Save results
    results_file = Path(args.output) / 'pivot_analysis.json'
    with open(results_file, 'w') as f:
        json.dump({
            'n_samples_analyzed': len(all_results),
            'statistical_tests': tests,
            'window_size': args.window,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Plot
    plot_pivot_comparison(tests, args.output)
    print(f"Plots saved to: {args.output}")


if __name__ == '__main__':
    main()
