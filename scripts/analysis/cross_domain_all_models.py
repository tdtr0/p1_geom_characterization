#!/usr/bin/env python3
"""
Cross-domain subspace alignment analysis for all models.

Tests whether error directions align across GSM8K and LogiQA tasks.
Low alignment = surface structure hypothesis (task-specific patterns)
High alignment = universal reasoning signature
"""

import argparse
import json
import numpy as np
import h5py
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def load_trajectories(filepath, max_samples=None):
    """Load trajectories and labels from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        traj = f['trajectories'][:]
        labels = f['is_correct'][:]

    if max_samples and len(traj) > max_samples:
        indices = np.random.RandomState(42).choice(len(traj), max_samples, replace=False)
        traj = traj[indices]
        labels = labels[indices]

    return traj.astype(np.float32), labels.astype(bool)


def compute_error_direction(trajectories, labels, use_cv=True, n_folds=5):
    """
    Compute error direction with optional cross-validation.

    Returns the error direction and CV-corrected projections.
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    # Use last layer, averaged over tokens
    activations = trajectories[:, :, -1, :].mean(axis=1)  # (n_samples, d_model)

    if not use_cv:
        # Simple mean difference (for alignment computation)
        correct_mean = activations[labels].mean(axis=0)
        incorrect_mean = activations[~labels].mean(axis=0)
        error_dir = incorrect_mean - correct_mean
        error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)
        return error_dir, activations @ error_dir

    # Cross-validated projections
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n_samples // n_folds

    cv_projections = np.zeros(n_samples)

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        train_labels = labels[train_idx]
        train_act = activations[train_idx]

        if train_labels.sum() < 3 or (~train_labels).sum() < 3:
            continue

        correct_mean = train_act[train_labels].mean(axis=0)
        incorrect_mean = train_act[~train_labels].mean(axis=0)
        error_dir = incorrect_mean - correct_mean
        error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)

        for idx in test_idx:
            cv_projections[idx] = activations[idx] @ error_dir

    # Also compute non-CV direction for alignment
    correct_mean = activations[labels].mean(axis=0)
    incorrect_mean = activations[~labels].mean(axis=0)
    error_dir = incorrect_mean - correct_mean
    error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)

    return error_dir, cv_projections


def cross_domain_analysis(traj1, labels1, traj2, labels2, task1_name, task2_name):
    """
    Analyze cross-domain subspace alignment between two tasks.
    """
    print(f"      {task1_name} vs {task2_name}...")

    # Compute error directions (non-CV for alignment, CV for transfer test)
    dir1, proj1 = compute_error_direction(traj1, labels1, use_cv=False)
    dir2, proj2 = compute_error_direction(traj2, labels2, use_cv=False)

    # 1. Cosine similarity of error directions
    cos_sim = np.abs(np.dot(dir1, dir2))

    # 2. Principal angle between subspaces
    # Get top-10 PCA components of difference
    act1 = traj1[:, :, -1, :].mean(axis=1)
    act2 = traj2[:, :, -1, :].mean(axis=1)

    diff1 = act1[~labels1] - act1[labels1].mean(axis=0)
    diff2 = act2[~labels2] - act2[labels2].mean(axis=0)

    k = min(10, diff1.shape[0]-1, diff2.shape[0]-1)
    pca1 = PCA(n_components=k).fit(diff1)
    pca2 = PCA(n_components=k).fit(diff2)

    S = pca1.components_ @ pca2.components_.T
    _, singular_values, _ = np.linalg.svd(S)
    principal_angles = np.arccos(np.clip(singular_values, -1, 1))
    mean_angle_rad = np.mean(principal_angles)
    mean_angle_deg = np.degrees(mean_angle_rad)

    # 3. Cross-transfer test (task1 direction on task2 data)
    # Using CV to avoid circularity
    _, cv_proj2_from_dir1 = compute_error_direction(traj2, labels2, use_cv=True)

    # Actually project task2 onto task1's direction
    act2_proj = act2 @ dir1

    correct_proj = act2_proj[labels2]
    incorrect_proj = act2_proj[~labels2]

    transfer_d_1to2 = cohens_d(incorrect_proj, correct_proj)
    _, transfer_p_1to2 = stats.mannwhitneyu(incorrect_proj, correct_proj, alternative='two-sided')

    # 4. Reverse transfer (task2 direction on task1 data)
    act1_proj = act1 @ dir2

    correct_proj = act1_proj[labels1]
    incorrect_proj = act1_proj[~labels1]

    transfer_d_2to1 = cohens_d(incorrect_proj, correct_proj)
    _, transfer_p_2to1 = stats.mannwhitneyu(incorrect_proj, correct_proj, alternative='two-sided')

    results = {
        'task1': task1_name,
        'task2': task2_name,
        'n1_correct': int(labels1.sum()),
        'n1_total': len(labels1),
        'n2_correct': int(labels2.sum()),
        'n2_total': len(labels2),
        'direction_cosine_similarity': float(cos_sim),
        'mean_principal_angle_rad': float(mean_angle_rad),
        'mean_principal_angle_deg': float(mean_angle_deg),
        'transfer_1to2': {
            'd': float(transfer_d_1to2),
            'p': float(transfer_p_1to2),
            'direction': f'{task1_name} → {task2_name}'
        },
        'transfer_2to1': {
            'd': float(transfer_d_2to1),
            'p': float(transfer_p_2to1),
            'direction': f'{task2_name} → {task1_name}'
        },
        'interpretation': 'aligned' if cos_sim > 0.5 else ('weak' if cos_sim > 0.3 else 'orthogonal')
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Cross-domain alignment analysis')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--models', type=str, default='olmo3_base,olmo3_sft,olmo3_rl_zero,olmo3_think')
    parser.add_argument('--max-samples', type=int, default=300)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(',')]

    all_results = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        # Load GSM8K and LogiQA
        gsm8k_path = data_dir / model / "gsm8k_trajectories.h5"
        logiqa_path = data_dir / model / "logiqa_trajectories.h5"

        if not gsm8k_path.exists() or not logiqa_path.exists():
            print(f"  Missing data files, skipping")
            continue

        gsm8k_traj, gsm8k_labels = load_trajectories(gsm8k_path, args.max_samples)
        logiqa_traj, logiqa_labels = load_trajectories(logiqa_path, args.max_samples)

        print(f"  GSM8K: {gsm8k_labels.sum()}/{len(gsm8k_labels)} correct ({100*gsm8k_labels.sum()/len(gsm8k_labels):.1f}%)")
        print(f"  LogiQA: {logiqa_labels.sum()}/{len(logiqa_labels)} correct ({100*logiqa_labels.sum()/len(logiqa_labels):.1f}%)")

        # Skip if insufficient samples
        if gsm8k_labels.sum() < 5 or logiqa_labels.sum() < 5:
            print(f"  Insufficient correct samples, skipping")
            continue

        # Cross-domain analysis
        results = cross_domain_analysis(
            gsm8k_traj, gsm8k_labels,
            logiqa_traj, logiqa_labels,
            f"{model}/gsm8k", f"{model}/logiqa"
        )

        all_results[model] = results

        print(f"  Cosine similarity: {results['direction_cosine_similarity']:.3f}")
        print(f"  Principal angle: {results['mean_principal_angle_deg']:.1f}°")
        print(f"  Transfer GSM8K→LogiQA: d={results['transfer_1to2']['d']:.3f}, p={results['transfer_1to2']['p']:.4f}")
        print(f"  Transfer LogiQA→GSM8K: d={results['transfer_2to1']['d']:.3f}, p={results['transfer_2to1']['p']:.4f}")
        print(f"  Interpretation: {results['interpretation']}")

    # Save results
    output_file = output_dir / "cross_domain_all_models.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_file}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Cross-Domain Alignment (GSM8K ↔ LogiQA)")
    print("="*80)
    print("\n| Model | Cosine Sim | Angle | GSM8K→LogiQA d | LogiQA→GSM8K d | Pattern |")
    print("|-------|------------|-------|----------------|----------------|---------|")

    for model, r in sorted(all_results.items()):
        cos_sim = r['direction_cosine_similarity']
        angle = r['mean_principal_angle_deg']
        d_1to2 = r['transfer_1to2']['d']
        d_2to1 = r['transfer_2to1']['d']
        pattern = r['interpretation']

        print(f"| {model:12s} | {cos_sim:.3f} | {angle:.1f}° | {d_1to2:+.3f} | {d_2to1:+.3f} | {pattern} |")

    print("\nInterpretation:")
    print("  - Cosine sim < 0.3: Orthogonal (different patterns)")
    print("  - Cosine sim 0.3-0.5: Weak alignment")
    print("  - Cosine sim > 0.5: Aligned (shared pattern)")
    print(f"\nComplete. Results in {output_dir}")


if __name__ == '__main__':
    main()
