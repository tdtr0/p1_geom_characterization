#!/usr/bin/env python3
"""
Analyze pivot point dynamics in Phase 2 trajectory data.

This script uses Phase 2 HDF5 files (trajectories) and pivot labels from
detect_pivots.py to analyze trajectory behavior at self-correction points.

Usage:
    python analyze_phase2_pivots.py \
        --trajectories data/phase2/olmo3_think_gsm8k.h5 \
        --pivots data/pivot_labels.json \
        --output results/

Metrics computed:
1. Velocity magnitude at pivot vs surrounding tokens
2. Direction change (cosine similarity) at pivot
3. Local Lyapunov estimate around pivot
4. Comparison: correct vs incorrect samples
5. Comparison across models/tasks
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm


# ============================================================================
# Trajectory Metrics (from analyze_pivot_dynamics.py)
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


def compute_direction_change(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute direction change (1 - cosine similarity) between consecutive token positions.

    Args:
        trajectory: (seq_len, n_layers, hidden_dim)

    Returns:
        direction_change: (seq_len-2,) - mean direction change per token (averaged across layers)
    """
    # Token-to-token differences at each layer
    token_diff = trajectory[1:, :, :] - trajectory[:-1, :, :]  # (seq_len-1, n_layers, hidden)

    # Compute direction change between consecutive diffs
    direction_changes = []
    for t in range(token_diff.shape[0] - 1):
        v1 = token_diff[t].flatten()  # Flatten across layers
        v2 = token_diff[t + 1].flatten()

        # Cosine similarity
        norm1 = np.linalg.norm(v1) + 1e-8
        norm2 = np.linalg.norm(v2) + 1e-8
        cos_sim = np.dot(v1, v2) / (norm1 * norm2)
        direction_change = 1 - cos_sim
        direction_changes.append(direction_change)

    return np.array(direction_changes)


def compute_local_lyapunov(trajectory: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Estimate local Lyapunov exponent around each token position.

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
            try:
                _, s, _ = np.linalg.svd(delta, full_matrices=False)
                # Log ratio of top to bottom singular values
                expansion = np.log(s[0] / (s[-1] + 1e-8))
                expansions.append(expansion)
            except np.linalg.LinAlgError:
                continue

        if expansions:
            lyapunov.append(np.mean(expansions))
        else:
            lyapunov.append(0.0)

    return np.array(lyapunov)


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_sample(
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
        'pivot_velocities': [],
        'pivot_direction_changes': [],
        'pivot_lyapunov': [],
        'random_velocities': [],
        'random_direction_changes': [],
        'random_lyapunov': [],
    }

    # Get metrics at pivot points
    valid_pivots = []
    for idx in pivot_indices:
        if 0 < idx < len(token_velocity):
            results['pivot_velocities'].append(float(token_velocity[idx - 1]))
            valid_pivots.append(idx)
        if 0 < idx < len(direction_change) + 1:
            results['pivot_direction_changes'].append(float(direction_change[idx - 1]))
        if window // 2 < idx < len(lyapunov) + window // 2:
            results['pivot_lyapunov'].append(float(lyapunov[idx - window // 2]))

    # Get metrics at random positions (excluding pivots and boundaries)
    valid_indices = set(range(window, seq_len - window)) - set(pivot_indices)
    valid_indices = list(valid_indices)

    if valid_indices and valid_pivots:
        # Sample same number as pivots, or all if fewer
        n_random = min(len(valid_indices), max(len(valid_pivots) * 3, 10))
        np.random.seed(42)  # Reproducibility
        random_indices = np.random.choice(valid_indices, n_random, replace=False)

        for idx in random_indices:
            if 0 < idx < len(token_velocity):
                results['random_velocities'].append(float(token_velocity[idx - 1]))
            if 0 < idx < len(direction_change) + 1:
                results['random_direction_changes'].append(float(direction_change[idx - 1]))
            if window // 2 < idx < len(lyapunov) + window // 2:
                results['random_lyapunov'].append(float(lyapunov[idx - window // 2]))

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
        effect_size = (np.mean(all_pivot_vel) - np.mean(all_random_vel)) / (np.std(all_random_vel) + 1e-8)
        tests['velocity'] = {
            'pivot_mean': float(np.mean(all_pivot_vel)),
            'pivot_std': float(np.std(all_pivot_vel)),
            'random_mean': float(np.mean(all_random_vel)),
            'random_std': float(np.std(all_random_vel)),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'effect_size_d': float(effect_size),
            'n_pivot': len(all_pivot_vel),
            'n_random': len(all_random_vel),
        }

    # Direction change test
    if len(all_pivot_dir) > 1 and len(all_random_dir) > 1:
        t_stat, p_val = stats.ttest_ind(all_pivot_dir, all_random_dir)
        effect_size = (np.mean(all_pivot_dir) - np.mean(all_random_dir)) / (np.std(all_random_dir) + 1e-8)
        tests['direction_change'] = {
            'pivot_mean': float(np.mean(all_pivot_dir)),
            'pivot_std': float(np.std(all_pivot_dir)),
            'random_mean': float(np.mean(all_random_dir)),
            'random_std': float(np.std(all_random_dir)),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'effect_size_d': float(effect_size),
            'n_pivot': len(all_pivot_dir),
            'n_random': len(all_random_dir),
        }

    # Lyapunov test
    if len(all_pivot_lyap) > 1 and len(all_random_lyap) > 1:
        t_stat, p_val = stats.ttest_ind(all_pivot_lyap, all_random_lyap)
        effect_size = (np.mean(all_pivot_lyap) - np.mean(all_random_lyap)) / (np.std(all_random_lyap) + 1e-8)
        tests['lyapunov'] = {
            'pivot_mean': float(np.mean(all_pivot_lyap)),
            'pivot_std': float(np.std(all_pivot_lyap)),
            'random_mean': float(np.mean(all_random_lyap)),
            'random_std': float(np.std(all_random_lyap)),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'effect_size_d': float(effect_size),
            'n_pivot': len(all_pivot_lyap),
            'n_random': len(all_random_lyap),
        }

    return tests


# ============================================================================
# Visualization
# ============================================================================

def plot_pivot_comparison(tests: dict, output_dir: str, title_suffix: str = ''):
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

    plt.suptitle(f'Pivot vs Random Token Dynamics{title_suffix}', fontsize=14)
    plt.tight_layout()
    filename = 'pivot_comparison' + title_suffix.replace(' ', '_').replace(':', '').lower() + '.png'
    plt.savefig(Path(output_dir) / filename, dpi=150)
    plt.close()


def plot_velocity_trajectory(
    trajectory: np.ndarray,
    pivot_indices: list[int],
    output_path: str,
    sample_id: str = '',
):
    """Plot velocity trajectory with pivot points marked."""
    token_velocity = compute_token_velocity_magnitude(trajectory)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(token_velocity, 'b-', alpha=0.7, linewidth=0.8, label='Token velocity')

    # Mark pivots
    for idx in pivot_indices:
        if 0 < idx < len(token_velocity):
            ax.axvline(idx, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.scatter([idx - 1], [token_velocity[idx - 1]], color='r', s=50, zorder=5)

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Velocity Magnitude')
    ax.set_title(f'Trajectory Velocity with Pivot Points {sample_id}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Main Processing
# ============================================================================

def process_file(
    trajectory_file: str,
    pivot_data: dict,
    window: int = 5,
) -> tuple[list[dict], list[dict]]:
    """
    Process a single trajectory file with pivot labels.

    Args:
        trajectory_file: Path to Phase 2 HDF5 file
        pivot_data: Pivot labels from detect_pivots.py for this file
        window: Lyapunov window size

    Returns:
        Tuple of (all_results, results_correct, results_incorrect)
    """
    all_results = []
    results_correct = []
    results_incorrect = []

    with h5py.File(trajectory_file, 'r') as f:
        trajectories = f['trajectories'][:]  # (n_samples, seq_len, n_layers, hidden)

        n_samples = trajectories.shape[0]

        for i in tqdm(range(n_samples), desc=f"Analyzing {Path(trajectory_file).name}", leave=False):
            # Get pivot info for this sample
            sample_info = pivot_data['samples'][i] if i < len(pivot_data['samples']) else None
            if sample_info is None:
                continue

            pivot_indices = [p['token_idx'] for p in sample_info.get('pivots', [])]

            # Skip samples without pivots
            if not pivot_indices:
                continue

            # Get trajectory (Phase 2 format: n_samples, seq_len, n_layers, hidden)
            trajectory = trajectories[i]  # (seq_len, n_layers, hidden)

            # Analyze
            result = analyze_sample(trajectory, pivot_indices, window=window)
            result['sample_idx'] = i
            result['is_correct'] = sample_info.get('is_correct')
            result['n_pivots'] = len(pivot_indices)

            all_results.append(result)

            # Split by correctness
            if sample_info.get('is_correct') is True:
                results_correct.append(result)
            elif sample_info.get('is_correct') is False:
                results_incorrect.append(result)

    return all_results, results_correct, results_incorrect


def main():
    parser = argparse.ArgumentParser(
        description='Analyze pivot point dynamics in Phase 2 data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--trajectories', '-t',
        type=str,
        nargs='+',
        required=True,
        help='Path to Phase 2 HDF5 trajectory file(s)'
    )
    parser.add_argument(
        '--pivots', '-p',
        type=str,
        required=True,
        help='Path to pivot labels JSON from detect_pivots.py'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/',
        help='Output directory'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=5,
        help='Window size for local Lyapunov estimation'
    )
    parser.add_argument(
        '--plot-examples',
        type=int,
        default=3,
        help='Number of example trajectories to plot'
    )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load pivot labels
    print(f"Loading pivot labels from: {args.pivots}")
    with open(args.pivots, 'r') as f:
        all_pivot_data = json.load(f)

    # Process each trajectory file
    combined_results = []
    combined_correct = []
    combined_incorrect = []
    file_results = {}

    for traj_file in args.trajectories:
        traj_path = Path(traj_file)
        if not traj_path.exists():
            print(f"Warning: {traj_file} does not exist, skipping")
            continue

        # Find matching pivot data
        key = traj_path.name
        if key not in all_pivot_data:
            print(f"Warning: No pivot data for {key}, skipping")
            continue

        pivot_data = all_pivot_data[key]

        # Process
        results, results_c, results_i = process_file(
            str(traj_path), pivot_data, window=args.window
        )

        combined_results.extend(results)
        combined_correct.extend(results_c)
        combined_incorrect.extend(results_i)
        file_results[key] = {
            'n_samples_with_pivots': len(results),
            'n_correct': len(results_c),
            'n_incorrect': len(results_i),
        }

        print(f"\n{key}:")
        print(f"  Samples with pivots: {len(results)}")
        print(f"  Correct: {len(results_c)}, Incorrect: {len(results_i)}")

    if not combined_results:
        print("\nNo samples with pivots found!")
        return

    # Statistical tests - All samples
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS: ALL SAMPLES")
    print("=" * 60)
    tests_all = run_statistical_tests(combined_results)
    print_test_results(tests_all)

    # Statistical tests - Correct samples only
    if combined_correct:
        print("\n" + "=" * 60)
        print("STATISTICAL TESTS: CORRECT SAMPLES ONLY")
        print("=" * 60)
        tests_correct = run_statistical_tests(combined_correct)
        print_test_results(tests_correct)
    else:
        tests_correct = {}

    # Statistical tests - Incorrect samples only
    if combined_incorrect:
        print("\n" + "=" * 60)
        print("STATISTICAL TESTS: INCORRECT SAMPLES ONLY")
        print("=" * 60)
        tests_incorrect = run_statistical_tests(combined_incorrect)
        print_test_results(tests_incorrect)
    else:
        tests_incorrect = {}

    # Plots
    plot_pivot_comparison(tests_all, args.output, title_suffix=' (All Samples)')
    if tests_correct:
        plot_pivot_comparison(tests_correct, args.output, title_suffix=' (Correct Only)')
    if tests_incorrect:
        plot_pivot_comparison(tests_incorrect, args.output, title_suffix=' (Incorrect Only)')

    # Plot example trajectories
    if args.plot_examples > 0 and combined_results:
        print(f"\nPlotting {args.plot_examples} example trajectories...")
        with h5py.File(args.trajectories[0], 'r') as f:
            trajectories = f['trajectories'][:]
            for i, result in enumerate(combined_results[:args.plot_examples]):
                sample_idx = result['sample_idx']
                traj = trajectories[sample_idx]
                # Get pivot indices from pivot data
                key = Path(args.trajectories[0]).name
                pivots = all_pivot_data[key]['samples'][sample_idx]['pivots']
                pivot_indices = [p['token_idx'] for p in pivots]

                plot_velocity_trajectory(
                    traj, pivot_indices,
                    str(Path(args.output) / f'example_trajectory_{i}.png'),
                    sample_id=f'(sample {sample_idx})'
                )

    # Save results
    results_file = Path(args.output) / 'pivot_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'n_files': len(file_results),
            'file_results': file_results,
            'n_total_samples': len(combined_results),
            'n_correct': len(combined_correct),
            'n_incorrect': len(combined_incorrect),
            'tests_all': tests_all,
            'tests_correct': tests_correct,
            'tests_incorrect': tests_incorrect,
            'window_size': args.window,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Plots saved to: {args.output}")


def print_test_results(tests: dict):
    """Print formatted test results."""
    for metric, t in tests.items():
        print(f"\n{metric.upper()}:")
        print(f"  Pivot:  {t['pivot_mean']:.4f} +/- {t['pivot_std']:.4f} (n={t['n_pivot']})")
        print(f"  Random: {t['random_mean']:.4f} +/- {t['random_std']:.4f} (n={t['n_random']})")
        print(f"  t={t['t_statistic']:.3f}, p={t['p_value']:.4f}, d={t['effect_size_d']:.3f}")
        if t['p_value'] < 0.05:
            print(f"  --> SIGNIFICANT (p < 0.05)")


if __name__ == '__main__':
    main()
