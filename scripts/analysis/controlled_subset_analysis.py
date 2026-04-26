#!/usr/bin/env python3
"""
Controlled Subset Analysis

Runs Phase 3 analyses on homogeneous problem-type subsets to test if
controlling for problem diversity reveals correctness signal.

Analyses:
1. Linear probe (baseline)
2. Menger curvature magnitude
3. Belief smoothness (if clause data available)

Usage:
    # First run classifier
    python scripts/analysis/gsm8k_problem_classifier.py

    # Then run controlled analysis
    python scripts/analysis/controlled_subset_analysis.py --problem-type money

Expected outcome:
    If diversity is the issue, effect sizes (Cohen's d) should increase
    on homogeneous subsets compared to the full dataset baseline.
"""

import os
import sys
import json
import h5py
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_trajectories(hdf5_path: str, sample_indices: list = None):
    """Load trajectories and labels from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        sample_indices: Optional list of indices to load (for subset analysis)

    Returns:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean correctness labels
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Handle different naming conventions
        if 'trajectories' in f:
            traj_key = 'trajectories'
        elif 'activations' in f:
            traj_key = 'activations'
        else:
            raise KeyError(f"No trajectories found in {hdf5_path}")

        if 'is_correct' in f:
            label_key = 'is_correct'
        elif 'correct' in f:
            label_key = 'correct'
        else:
            raise KeyError(f"No correctness labels found in {hdf5_path}")

        if sample_indices is not None:
            # Load only specified samples
            trajectories = f[traj_key][sample_indices]
            labels = f[label_key][sample_indices]
        else:
            trajectories = f[traj_key][:]
            labels = f[label_key][:]

    return trajectories, labels


def mean_pool_over_sequence(trajectories: np.ndarray) -> np.ndarray:
    """Mean pool trajectories over sequence dimension.

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)

    Returns:
        (n_samples, n_layers, d_model)
    """
    return np.nanmean(trajectories, axis=1)


def compute_linear_probe_separability(activations: np.ndarray, labels: np.ndarray,
                                       layer_idx: int = -1) -> dict:
    """Compute linear probe AUC and Cohen's d at specified layer.

    Args:
        activations: (n_samples, n_layers, d_model)
        labels: (n_samples,) boolean

    Returns:
        dict with 'auc', 'd', 'accuracy'
    """
    # Get activations at specified layer
    X = activations[:, layer_idx, :]  # (n_samples, d_model)
    y = labels.astype(int)

    # Remove NaN samples
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    y = y[valid]

    if len(np.unique(y)) < 2:
        return {'auc': np.nan, 'd': np.nan, 'accuracy': np.nan, 'n_correct': y.sum(), 'n_total': len(y)}

    # Cross-validated predictions
    cv = StratifiedKFold(n_splits=min(5, y.sum(), (1-y).sum()), shuffle=True, random_state=42)

    try:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
        preds = cross_val_predict(clf, X, y, cv=cv)

        # AUC
        from sklearn.metrics import roc_auc_score, accuracy_score
        auc = roc_auc_score(y, probs)
        accuracy = accuracy_score(y, preds)

        # Cohen's d using difference-in-means direction
        correct_mean = X[y == 1].mean(axis=0)
        incorrect_mean = X[y == 0].mean(axis=0)
        direction = correct_mean - incorrect_mean
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Project onto direction
        projections = X @ direction
        proj_correct = projections[y == 1]
        proj_incorrect = projections[y == 0]

        # Cohen's d
        pooled_std = np.sqrt(((len(proj_correct) - 1) * proj_correct.std()**2 +
                              (len(proj_incorrect) - 1) * proj_incorrect.std()**2) /
                             (len(proj_correct) + len(proj_incorrect) - 2))
        d = (proj_correct.mean() - proj_incorrect.mean()) / (pooled_std + 1e-8)

        return {
            'auc': float(auc),
            'd': float(d),
            'accuracy': float(accuracy),
            'n_correct': int(y.sum()),
            'n_total': int(len(y))
        }

    except Exception as e:
        print(f"  Warning: Linear probe failed: {e}")
        return {'auc': np.nan, 'd': np.nan, 'accuracy': np.nan, 'n_correct': y.sum(), 'n_total': len(y)}


def compute_menger_curvature(activations: np.ndarray, labels: np.ndarray) -> dict:
    """Compute Menger curvature on layer trajectory.

    Menger curvature κ = 4A / (|a||b||c|) for triangle formed by
    consecutive layer points.

    Args:
        activations: (n_samples, n_layers, d_model) mean-pooled
        labels: (n_samples,) boolean

    Returns:
        dict with 'd', 'p', 'correct_mean', 'incorrect_mean'
    """
    n_samples, n_layers, d_model = activations.shape

    curvatures = []
    for i in range(n_samples):
        sample_curvatures = []
        for l in range(n_layers - 2):
            p0 = activations[i, l, :]
            p1 = activations[i, l+1, :]
            p2 = activations[i, l+2, :]

            # Skip if NaN
            if np.isnan(p0).any() or np.isnan(p1).any() or np.isnan(p2).any():
                continue

            # Side lengths
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p2 - p0)

            # Area via Heron's formula
            s = (a + b + c) / 2
            area_sq = s * (s - a) * (s - b) * (s - c)
            if area_sq < 0:
                area_sq = 0  # Numerical stability
            area = np.sqrt(area_sq)

            # Menger curvature
            denom = a * b * c
            if denom > 1e-8:
                kappa = 4 * area / denom
                sample_curvatures.append(kappa)

        if sample_curvatures:
            curvatures.append(np.mean(sample_curvatures))
        else:
            curvatures.append(np.nan)

    curvatures = np.array(curvatures)
    valid = ~np.isnan(curvatures)

    correct_curvs = curvatures[valid & labels]
    incorrect_curvs = curvatures[valid & ~labels]

    if len(correct_curvs) < 2 or len(incorrect_curvs) < 2:
        return {'d': np.nan, 'p': np.nan, 'correct_mean': np.nan, 'incorrect_mean': np.nan}

    # Cohen's d
    pooled_std = np.sqrt(((len(correct_curvs) - 1) * correct_curvs.std()**2 +
                          (len(incorrect_curvs) - 1) * incorrect_curvs.std()**2) /
                         (len(correct_curvs) + len(incorrect_curvs) - 2))
    d = (correct_curvs.mean() - incorrect_curvs.mean()) / (pooled_std + 1e-8)

    # T-test
    t_stat, p_val = stats.ttest_ind(correct_curvs, incorrect_curvs)

    return {
        'd': float(d),
        'p': float(p_val),
        'correct_mean': float(correct_curvs.mean()),
        'incorrect_mean': float(incorrect_curvs.mean()),
        'n_correct': int(len(correct_curvs)),
        'n_incorrect': int(len(incorrect_curvs)),
    }


def run_analysis(hdf5_path: str, sample_indices: list = None, subset_name: str = "full"):
    """Run full analysis suite on dataset or subset.

    Returns:
        dict with analysis results
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {subset_name} (n={len(sample_indices) if sample_indices else 'all'})")
    print(f"{'='*60}")

    # Load data
    trajectories, labels = load_trajectories(hdf5_path, sample_indices)
    print(f"Loaded: {trajectories.shape}, correct: {labels.sum()}/{len(labels)}")

    # Mean pool
    activations = mean_pool_over_sequence(trajectories)
    print(f"Mean-pooled: {activations.shape}")

    results = {
        'subset': subset_name,
        'n_samples': len(labels),
        'n_correct': int(labels.sum()),
        'accuracy': float(labels.mean()),
    }

    # Linear probe at last layer
    print("\n1. Linear Probe (last layer)...")
    probe_results = compute_linear_probe_separability(activations, labels, layer_idx=-1)
    results['linear_probe'] = probe_results
    print(f"   AUC: {probe_results['auc']:.3f}, d: {probe_results['d']:.3f}")

    # Menger curvature
    print("\n2. Menger Curvature...")
    curv_results = compute_menger_curvature(activations, labels)
    results['menger_curvature'] = curv_results
    print(f"   d: {curv_results['d']:.3f}, p: {curv_results['p']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Controlled Subset Analysis')
    parser.add_argument('--hdf5', type=str, required=True,
                        help='Path to HDF5 trajectory file')
    parser.add_argument('--indices-file', type=str, default='results/gsm8k_subset_indices.json',
                        help='JSON file with problem type indices')
    parser.add_argument('--problem-type', type=str, default=None,
                        help='Specific problem type to analyze (default: all)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    args = parser.parse_args()

    # Load subset indices
    indices_path = Path(args.indices_file)
    if indices_path.exists():
        with open(indices_path) as f:
            subset_indices = json.load(f)
        print(f"Loaded subset indices from {indices_path}")
    else:
        print(f"Warning: {indices_path} not found. Run gsm8k_problem_classifier.py first.")
        print("Running on full dataset only.")
        subset_indices = {}

    # Run analysis on full dataset first
    all_results = []

    print("\n" + "="*70)
    print("FULL DATASET BASELINE")
    print("="*70)
    full_results = run_analysis(args.hdf5, sample_indices=None, subset_name="full")
    all_results.append(full_results)

    # Run on subsets
    if args.problem_type:
        # Single problem type
        types_to_run = [args.problem_type]
    else:
        # All problem types with enough samples
        types_to_run = [t for t, indices in subset_indices.items() if len(indices) >= 20]

    for problem_type in types_to_run:
        if problem_type not in subset_indices:
            print(f"\nWarning: Problem type '{problem_type}' not found in indices file")
            continue

        indices = subset_indices[problem_type]
        if len(indices) < 20:
            print(f"\nSkipping '{problem_type}': only {len(indices)} samples")
            continue

        subset_results = run_analysis(args.hdf5, sample_indices=indices, subset_name=problem_type)
        all_results.append(subset_results)

    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Subset':<20} {'N':<6} {'Acc%':<7} {'Probe d':<10} {'Curv d':<10} {'Curv p':<10}")
    print("-" * 70)

    for r in all_results:
        print(f"{r['subset']:<20} {r['n_samples']:<6} {100*r['accuracy']:<7.1f} "
              f"{r['linear_probe']['d']:<10.3f} "
              f"{r['menger_curvature']['d']:<10.3f} "
              f"{r['menger_curvature']['p']:<10.4f}")

    # Effect size comparison
    baseline = all_results[0]
    print("\n" + "="*70)
    print("EFFECT SIZE CHANGE VS BASELINE")
    print("="*70)

    for r in all_results[1:]:
        probe_ratio = r['linear_probe']['d'] / (baseline['linear_probe']['d'] + 1e-8)
        curv_ratio = abs(r['menger_curvature']['d']) / (abs(baseline['menger_curvature']['d']) + 1e-8)
        print(f"\n{r['subset']}:")
        print(f"  Linear probe d: {r['linear_probe']['d']:.3f} ({probe_ratio:.2f}x baseline)")
        print(f"  Curvature d: {r['menger_curvature']['d']:.3f} ({curv_ratio:.2f}x baseline)")

        if probe_ratio > 1.5 or curv_ratio > 1.5:
            print(f"  → SUBSTANTIAL INCREASE: Controlling for problem type reveals signal!")
        elif probe_ratio > 1.2 or curv_ratio > 1.2:
            print(f"  → MODERATE INCREASE: Some benefit from homogeneous subset")
        else:
            print(f"  → NO CHANGE: Problem diversity is not the main issue")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'controlled_subset_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
