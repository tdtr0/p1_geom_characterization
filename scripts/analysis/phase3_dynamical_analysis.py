#!/usr/bin/env python3
"""
Phase 3 Dynamical Systems Analysis

Implements analyses from PHASE3_DETAILED_PLAN.md:
1. Error-Detection Direction Analysis (Wynroe-style)
2. Menger Curvature Analysis (Zhou et al., 2025)
3. Lyapunov Exponent Analysis
4. Attractor Analysis

Usage:
    python phase3_dynamical_analysis.py --data-dir /path/to/trajectories --model olmo3_base --tasks humaneval,logiqa
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================================
# 1. ERROR-DETECTION DIRECTION ANALYSIS (Wynroe-style)
# ============================================================================

def extract_error_detection_direction(correct_trajectories, incorrect_trajectories, layer_idx=-1):
    """
    Extract error-detection direction via difference-in-means.

    Args:
        correct_trajectories: (n_correct, seq_len, n_layers, d_model)
        incorrect_trajectories: (n_incorrect, seq_len, n_layers, d_model)
        layer_idx: Which layer to analyze (-1 = last layer)

    Returns:
        direction: (d_model,) - the error-detection direction
        statistics: dict with significance tests
    """
    # Use mean activation across sequence
    correct_acts = correct_trajectories[:, :, layer_idx, :].mean(axis=1)  # (n_correct, d_model)
    incorrect_acts = incorrect_trajectories[:, :, layer_idx, :].mean(axis=1)  # (n_incorrect, d_model)

    # Difference-in-means direction
    mean_correct = correct_acts.mean(axis=0)
    mean_incorrect = incorrect_acts.mean(axis=0)

    direction = mean_incorrect - mean_correct  # Points toward "error" region
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction = direction / norm

    # Project all samples onto direction
    correct_proj = correct_acts @ direction
    incorrect_proj = incorrect_acts @ direction

    # Statistical test
    t_stat, p_value = stats.ttest_ind(incorrect_proj, correct_proj)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(correct_proj)-1)*np.var(correct_proj) +
                          (len(incorrect_proj)-1)*np.var(incorrect_proj)) /
                         (len(correct_proj) + len(incorrect_proj) - 2))
    if pooled_std > 1e-8:
        effect_size = (incorrect_proj.mean() - correct_proj.mean()) / pooled_std
    else:
        effect_size = 0

    # Classification accuracy using this single direction
    threshold = (correct_proj.mean() + incorrect_proj.mean()) / 2
    correct_pred = (correct_proj < threshold).sum()
    incorrect_pred = (incorrect_proj > threshold).sum()
    accuracy = (correct_pred + incorrect_pred) / (len(correct_proj) + len(incorrect_proj))

    return direction, {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'effect_size_d': float(effect_size),
        'classification_accuracy': float(accuracy),
        'correct_mean': float(correct_proj.mean()),
        'incorrect_mean': float(incorrect_proj.mean()),
        'separation': float(incorrect_proj.mean() - correct_proj.mean()),
        'n_correct': len(correct_proj),
        'n_incorrect': len(incorrect_proj)
    }


def analyze_error_direction_per_layer(trajectories, labels, actual_layer_indices=None):
    """
    Find the best layer for error detection.

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean array
        actual_layer_indices: List of actual layer indices (e.g., [0, 2, 4, ...])
    """
    n_layers = trajectories.shape[2]

    correct_traj = trajectories[labels == True]
    incorrect_traj = trajectories[labels == False]

    if len(correct_traj) == 0 or len(incorrect_traj) == 0:
        return pd.DataFrame()

    results = []
    for layer_idx in range(n_layers):
        actual_layer = actual_layer_indices[layer_idx] if actual_layer_indices else layer_idx * 2
        try:
            _, layer_stats = extract_error_detection_direction(
                correct_traj, incorrect_traj, layer_idx=layer_idx
            )
            layer_stats['layer_idx'] = layer_idx
            layer_stats['actual_layer'] = actual_layer
            results.append(layer_stats)
        except Exception as e:
            print(f"  Warning: Layer {layer_idx} failed: {e}")

    return pd.DataFrame(results)


def test_direction_transfer(train_traj, train_labels, test_traj, test_labels, layer_idx=-1):
    """
    Test if error-detection direction transfers across domains.
    """
    correct_train = train_traj[train_labels == True]
    incorrect_train = train_traj[train_labels == False]

    if len(correct_train) == 0 or len(incorrect_train) == 0:
        return {'train_accuracy': 0, 'test_accuracy': 0, 'transfer_ratio': 0}

    direction, train_stats = extract_error_detection_direction(
        correct_train, incorrect_train, layer_idx=layer_idx
    )

    # Apply to test domain
    test_acts = test_traj[:, :, layer_idx, :].mean(axis=1)
    test_proj = test_acts @ direction

    # Use train threshold
    threshold = train_stats['correct_mean'] + (train_stats['separation'] / 2)

    correct_test_proj = test_proj[test_labels == True]
    incorrect_test_proj = test_proj[test_labels == False]

    if len(correct_test_proj) == 0 or len(incorrect_test_proj) == 0:
        return {'train_accuracy': train_stats['classification_accuracy'],
                'test_accuracy': 0, 'transfer_ratio': 0}

    correct_pred = (correct_test_proj < threshold).sum()
    incorrect_pred = (incorrect_test_proj > threshold).sum()
    test_accuracy = (correct_pred + incorrect_pred) / len(test_labels)

    train_acc = train_stats['classification_accuracy']
    transfer_ratio = test_accuracy / train_acc if train_acc > 0 else 0

    return {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_accuracy),
        'transfer_ratio': float(transfer_ratio)
    }


# ============================================================================
# 2. MENGER CURVATURE ANALYSIS (Zhou et al., 2025)
# ============================================================================

def compute_menger_curvature(p1, p2, p3):
    """
    Compute Menger curvature for three consecutive points.
    κ = 4 * Area(triangle) / (|p1-p2| * |p2-p3| * |p3-p1|)
    """
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)

    if a < 1e-10 or b < 1e-10 or c < 1e-10:
        return 0.0

    # Area via cross product (Gram matrix for high-dim)
    v1 = p2 - p1
    v2 = p3 - p1
    gram = np.array([[np.dot(v1, v1), np.dot(v1, v2)],
                     [np.dot(v1, v2), np.dot(v2, v2)]])
    area = 0.5 * np.sqrt(max(0, np.linalg.det(gram)))

    # Menger curvature
    curvature = 4 * area / (a * b * c) if (a * b * c) > 1e-10 else 0

    return curvature


def compute_trajectory_curvature_profile(trajectory):
    """
    Compute Menger curvature at each point along a layer trajectory.

    Args:
        trajectory: (n_layers, d_model) - single sample's layer trajectory

    Returns:
        curvatures: (n_layers-2,) - curvature at each interior point
    """
    n_layers = trajectory.shape[0]
    curvatures = []

    for i in range(1, n_layers - 1):
        p1 = trajectory[i-1]
        p2 = trajectory[i]
        p3 = trajectory[i+1]

        kappa = compute_menger_curvature(p1, p2, p3)
        curvatures.append(kappa)

    return np.array(curvatures)


def analyze_menger_curvature(trajectories, labels):
    """
    Compare curvature profiles between correct and incorrect solutions.
    """
    n_samples = trajectories.shape[0]
    results = []

    for i in range(n_samples):
        # Average trajectory across sequence positions
        mean_traj = trajectories[i].mean(axis=0)  # (n_layers, d_model)

        curvatures = compute_trajectory_curvature_profile(mean_traj)

        if len(curvatures) > 0:
            n_half = len(curvatures) // 2
            results.append({
                'sample_idx': i,
                'is_correct': bool(labels[i]),
                'mean_curvature': float(curvatures.mean()),
                'max_curvature': float(curvatures.max()),
                'curvature_variance': float(curvatures.var()),
                'early_curvature': float(curvatures[:n_half].mean()) if n_half > 0 else 0,
                'late_curvature': float(curvatures[n_half:].mean()) if n_half > 0 else 0,
            })

    return pd.DataFrame(results)


def compute_curvature_correlation(traj1, labels1, traj2, labels2):
    """
    Test if correct solutions show correlated curvature profiles across domains.
    """
    correct1 = traj1[labels1 == True]
    correct2 = traj2[labels2 == True]

    if len(correct1) < 2 or len(correct2) < 2:
        return {'pearson_correlation': 0, 'pearson_pvalue': 1,
                'spearman_correlation': 0, 'spearman_pvalue': 1}

    # Compute mean curvature profiles
    profiles1 = []
    for traj in correct1[:100]:  # Limit for speed
        mean_traj = traj.mean(axis=0)
        profiles1.append(compute_trajectory_curvature_profile(mean_traj))

    profiles2 = []
    for traj in correct2[:100]:
        mean_traj = traj.mean(axis=0)
        profiles2.append(compute_trajectory_curvature_profile(mean_traj))

    mean_profile1 = np.mean(profiles1, axis=0)
    mean_profile2 = np.mean(profiles2, axis=0)

    # Correlation
    try:
        pearson_r, pearson_p = stats.pearsonr(mean_profile1, mean_profile2)
        spearman_r, spearman_p = stats.spearmanr(mean_profile1, mean_profile2)
    except:
        pearson_r, pearson_p = 0, 1
        spearman_r, spearman_p = 0, 1

    return {
        'pearson_correlation': float(pearson_r),
        'pearson_pvalue': float(pearson_p),
        'spearman_correlation': float(spearman_r),
        'spearman_pvalue': float(spearman_p)
    }


# ============================================================================
# 3. LYAPUNOV EXPONENT ANALYSIS
# ============================================================================

def compute_lyapunov_exponents(trajectory):
    """
    Compute local Lyapunov exponents along a trajectory.

    For discrete dynamics x_{l+1} = f(x_l), estimate expansion/contraction
    using singular value ratios of layer transitions.

    Args:
        trajectory: (seq_len, n_layers, d_model)

    Returns:
        dict with Lyapunov statistics
    """
    seq_len, n_layers, d_model = trajectory.shape

    layer_lyapunov = []
    for l in range(n_layers - 1):
        x_l = trajectory[:, l, :]      # (seq_len, d_model)
        x_l1 = trajectory[:, l+1, :]   # (seq_len, d_model)

        delta_x = x_l1 - x_l

        # SVD to estimate expansion
        try:
            _, s, _ = np.linalg.svd(delta_x, full_matrices=False)
            if len(s) > 1 and s[-1] > 1e-10:
                expansion = np.log(s[0] / (s[-1] + 1e-8))
            else:
                expansion = 0
        except:
            expansion = 0

        layer_lyapunov.append(expansion)

    layer_lyapunov = np.array(layer_lyapunov)

    # Fit trend
    if len(layer_lyapunov) > 1:
        trend = np.polyfit(range(len(layer_lyapunov)), layer_lyapunov, 1)[0]
    else:
        trend = 0

    return {
        'mean_lyapunov': float(layer_lyapunov.mean()),
        'max_lyapunov': float(layer_lyapunov.max()),
        'min_lyapunov': float(layer_lyapunov.min()),
        'lyapunov_std': float(layer_lyapunov.std()),
        'lyapunov_trend': float(trend),  # Positive = diverging through layers
        'layer_lyapunov': layer_lyapunov.tolist()
    }


def analyze_lyapunov(trajectories, labels):
    """
    Analyze Lyapunov exponents for correct vs incorrect solutions.
    """
    results = []

    for i in range(len(trajectories)):
        lyap = compute_lyapunov_exponents(trajectories[i])
        lyap['sample_idx'] = i
        lyap['is_correct'] = bool(labels[i])
        # Remove list for DataFrame
        lyap_profile = lyap.pop('layer_lyapunov')
        results.append(lyap)

    return pd.DataFrame(results)


# ============================================================================
# 4. ATTRACTOR ANALYSIS
# ============================================================================

def analyze_attractors(trajectories, labels, n_clusters=10):
    """
    Characterize attractor structure in trajectory space.
    """
    n_samples = trajectories.shape[0]

    # Final layer activations as "attractor proxies"
    final_states = trajectories[:, :, -1, :]  # (n_samples, seq_len, d_model)
    final_mean = final_states.mean(axis=1)    # (n_samples, d_model)

    # Reduce dimensionality for clustering
    pca = PCA(n_components=min(50, final_mean.shape[1]))
    final_reduced = pca.fit_transform(final_mean)

    # Cluster final states
    n_clusters = min(n_clusters, n_samples // 2)
    if n_clusters < 2:
        return pd.DataFrame()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(final_reduced)

    # Analyze cluster composition
    cluster_stats = []
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        n_total = mask.sum()
        if n_total > 0:
            n_correct = labels[mask].sum()
            purity = max(n_correct, n_total - n_correct) / n_total
            correct_rate = n_correct / n_total
            cluster_stats.append({
                'cluster_id': cluster_id,
                'n_samples': int(n_total),
                'n_correct': int(n_correct),
                'n_incorrect': int(n_total - n_correct),
                'purity': float(purity),
                'correct_rate': float(correct_rate)
            })

    return pd.DataFrame(cluster_stats)


def analyze_convergence(trajectories, labels):
    """
    Analyze convergence rate to final state.
    """
    results = []

    for i in range(len(trajectories)):
        traj = trajectories[i]  # (seq_len, n_layers, d_model)
        final = traj[:, -1, :]  # (seq_len, d_model)

        # Distance to final state at each layer
        distances = []
        for l in range(traj.shape[1]):
            dist = np.linalg.norm(traj[:, l, :] - final, axis=-1).mean()
            distances.append(dist)

        distances = np.array(distances)

        # Fit exponential decay: d(l) = d_0 * exp(-λ * l)
        if len(distances) > 1 and distances[0] > 1e-8:
            log_dist = np.log(distances + 1e-8)
            decay_rate, _ = np.polyfit(range(len(distances)), log_dist, 1)
        else:
            decay_rate = 0

        results.append({
            'sample_idx': i,
            'is_correct': bool(labels[i]),
            'decay_rate': float(-decay_rate),  # Positive = converging
            'initial_distance': float(distances[0]),
            'final_distance': float(distances[-1]),
            'distance_ratio': float(distances[-1] / (distances[0] + 1e-8))
        })

    return pd.DataFrame(results)


# ============================================================================
# MAIN
# ============================================================================

def load_trajectories(filepath, max_samples=None):
    """Load trajectories and labels from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        trajectories = f['trajectories'][:]
        # Try different key names for correctness labels
        if 'is_correct' in f:
            labels = f['is_correct'][:]
        elif 'correct' in f:
            labels = f['correct'][:]
        else:
            raise KeyError(f"No correctness labels found. Available keys: {list(f.keys())}")

        if max_samples and max_samples < len(trajectories):
            trajectories = trajectories[:max_samples]
            labels = labels[:max_samples]

    return trajectories.astype(np.float32), labels.astype(bool)


def compute_statistics(df, metric_col, group_col='is_correct'):
    """Compute summary statistics for correct vs incorrect."""
    correct = df[df[group_col] == True][metric_col]
    incorrect = df[df[group_col] == False][metric_col]

    if len(correct) < 2 or len(incorrect) < 2:
        return {
            'correct_mean': float(correct.mean()) if len(correct) > 0 else 0,
            'incorrect_mean': float(incorrect.mean()) if len(incorrect) > 0 else 0,
            't_statistic': 0,
            'p_value': 1,
            'effect_size': 0
        }

    t_stat, p_val = stats.ttest_ind(correct, incorrect)

    pooled_std = np.sqrt(((len(correct)-1)*correct.var() +
                          (len(incorrect)-1)*incorrect.var()) /
                         (len(correct) + len(incorrect) - 2))
    effect_size = (correct.mean() - incorrect.mean()) / (pooled_std + 1e-8)

    return {
        'correct_mean': float(correct.mean()),
        'incorrect_mean': float(incorrect.mean()),
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'effect_size': float(effect_size)
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 3 Dynamical Systems Analysis')
    parser.add_argument('--data-dir', required=True, help='Directory containing trajectory HDF5 files')
    parser.add_argument('--model', required=True, help='Model name (e.g., olmo3_base)')
    parser.add_argument('--tasks', required=True, help='Comma-separated tasks (e.g., humaneval,logiqa)')
    parser.add_argument('--output', default='results/phase3_dynamical.json', help='Output JSON file')
    parser.add_argument('--max-samples', type=int, default=300, help='Max samples per task')

    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(',')]
    results = {
        'model': args.model,
        'tasks': tasks,
        'error_direction': {},
        'menger_curvature': {},
        'lyapunov': {},
        'attractor': {},
        'convergence': {},
        'transfer': {}
    }

    # Actual layer indices (even layers 0-30)
    actual_layer_indices = list(range(0, 32, 2))

    # Load all tasks
    task_data = {}
    for task in tasks:
        filepath = Path(args.data_dir) / args.model / f'{task}_trajectories.h5'
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping {task}")
            continue

        print(f"\nLoading {task}...")
        traj, labels = load_trajectories(filepath, args.max_samples)
        print(f"  Shape: {traj.shape}, Correct: {labels.sum()}/{len(labels)} ({100*labels.mean():.1f}%)")
        task_data[task] = {'trajectories': traj, 'labels': labels}

    if not task_data:
        print("Error: No task data loaded!")
        sys.exit(1)

    # ========================================================================
    # 1. Error-Detection Direction Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("1. ERROR-DETECTION DIRECTION ANALYSIS")
    print("="*70)

    for task, data in task_data.items():
        print(f"\n{task.upper()}:")
        traj, labels = data['trajectories'], data['labels']

        # Per-layer analysis
        layer_df = analyze_error_direction_per_layer(traj, labels, actual_layer_indices)

        if len(layer_df) > 0:
            # Find best layer
            best_idx = layer_df['effect_size_d'].abs().idxmax()
            best_layer = layer_df.loc[best_idx]

            print(f"  Best layer: {int(best_layer['actual_layer'])} (d={best_layer['effect_size_d']:.3f}, p={best_layer['p_value']:.4f})")
            print(f"  Classification accuracy: {best_layer['classification_accuracy']:.1%}")

            results['error_direction'][task] = {
                'best_layer': int(best_layer['actual_layer']),
                'best_effect_size': float(best_layer['effect_size_d']),
                'best_accuracy': float(best_layer['classification_accuracy']),
                'per_layer': layer_df.to_dict('records')
            }
        else:
            print(f"  Insufficient data for analysis")

    # ========================================================================
    # 2. Menger Curvature Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("2. MENGER CURVATURE ANALYSIS")
    print("="*70)

    for task, data in task_data.items():
        print(f"\n{task.upper()}:")
        traj, labels = data['trajectories'], data['labels']

        curvature_df = analyze_menger_curvature(traj, labels)

        if len(curvature_df) > 0:
            stats_mean = compute_statistics(curvature_df, 'mean_curvature')
            stats_early = compute_statistics(curvature_df, 'early_curvature')
            stats_late = compute_statistics(curvature_df, 'late_curvature')

            print(f"  Mean curvature - Correct: {stats_mean['correct_mean']:.6f}, Incorrect: {stats_mean['incorrect_mean']:.6f}")
            print(f"    Effect size: {stats_mean['effect_size']:.3f}, p={stats_mean['p_value']:.4f}")

            results['menger_curvature'][task] = {
                'mean_curvature_stats': stats_mean,
                'early_curvature_stats': stats_early,
                'late_curvature_stats': stats_late
            }

    # Cross-domain curvature correlation
    if len(task_data) >= 2:
        print("\n  Cross-domain curvature correlation:")
        task_list = list(task_data.keys())
        for i, task1 in enumerate(task_list):
            for task2 in task_list[i+1:]:
                corr = compute_curvature_correlation(
                    task_data[task1]['trajectories'], task_data[task1]['labels'],
                    task_data[task2]['trajectories'], task_data[task2]['labels']
                )
                print(f"    {task1} <-> {task2}: r={corr['pearson_correlation']:.3f} (p={corr['pearson_pvalue']:.4f})")
                results['menger_curvature'][f'{task1}_vs_{task2}_correlation'] = corr

    # ========================================================================
    # 3. Lyapunov Exponent Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("3. LYAPUNOV EXPONENT ANALYSIS")
    print("="*70)

    for task, data in task_data.items():
        print(f"\n{task.upper()}:")
        traj, labels = data['trajectories'], data['labels']

        lyapunov_df = analyze_lyapunov(traj, labels)

        stats_mean = compute_statistics(lyapunov_df, 'mean_lyapunov')
        stats_trend = compute_statistics(lyapunov_df, 'lyapunov_trend')

        print(f"  Mean Lyapunov - Correct: {stats_mean['correct_mean']:.3f}, Incorrect: {stats_mean['incorrect_mean']:.3f}")
        print(f"    Effect size: {stats_mean['effect_size']:.3f}, p={stats_mean['p_value']:.4f}")
        print(f"  Lyapunov trend - Correct: {stats_trend['correct_mean']:.3f}, Incorrect: {stats_trend['incorrect_mean']:.3f}")

        results['lyapunov'][task] = {
            'mean_lyapunov_stats': stats_mean,
            'lyapunov_trend_stats': stats_trend
        }

    # ========================================================================
    # 4. Attractor Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("4. ATTRACTOR ANALYSIS")
    print("="*70)

    for task, data in task_data.items():
        print(f"\n{task.upper()}:")
        traj, labels = data['trajectories'], data['labels']

        # Cluster analysis
        cluster_df = analyze_attractors(traj, labels, n_clusters=8)
        if len(cluster_df) > 0:
            mean_purity = cluster_df['purity'].mean()
            print(f"  Mean cluster purity: {mean_purity:.1%}")

            # Identify correct-dominated vs incorrect-dominated clusters
            correct_clusters = cluster_df[cluster_df['correct_rate'] > 0.5]
            incorrect_clusters = cluster_df[cluster_df['correct_rate'] <= 0.5]
            print(f"  Correct-dominated clusters: {len(correct_clusters)}, Incorrect-dominated: {len(incorrect_clusters)}")

            results['attractor'][task] = {
                'mean_purity': float(mean_purity),
                'n_correct_clusters': len(correct_clusters),
                'n_incorrect_clusters': len(incorrect_clusters),
                'clusters': cluster_df.to_dict('records')
            }

        # Convergence analysis
        conv_df = analyze_convergence(traj, labels)
        stats_decay = compute_statistics(conv_df, 'decay_rate')
        stats_ratio = compute_statistics(conv_df, 'distance_ratio')

        print(f"  Decay rate - Correct: {stats_decay['correct_mean']:.3f}, Incorrect: {stats_decay['incorrect_mean']:.3f}")
        print(f"    Effect size: {stats_decay['effect_size']:.3f}, p={stats_decay['p_value']:.4f}")

        results['convergence'][task] = {
            'decay_rate_stats': stats_decay,
            'distance_ratio_stats': stats_ratio
        }

    # ========================================================================
    # 5. Error Direction Transfer Test
    # ========================================================================
    if len(task_data) >= 2:
        print("\n" + "="*70)
        print("5. ERROR DIRECTION TRANSFER TEST")
        print("="*70)

        task_list = list(task_data.keys())
        for train_task in task_list:
            for test_task in task_list:
                if train_task == test_task:
                    continue

                transfer = test_direction_transfer(
                    task_data[train_task]['trajectories'],
                    task_data[train_task]['labels'],
                    task_data[test_task]['trajectories'],
                    task_data[test_task]['labels'],
                    layer_idx=-1  # Use last layer
                )

                key = f'{train_task}_to_{test_task}'
                print(f"  {train_task} -> {test_task}: Train acc={transfer['train_accuracy']:.1%}, Test acc={transfer['test_accuracy']:.1%}")
                results['transfer'][key] = transfer

    # ========================================================================
    # Save Results
    # ========================================================================
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n1. Error-Detection Direction:")
    for task, stats in results['error_direction'].items():
        print(f"   {task}: Best layer {stats['best_layer']}, d={stats['best_effect_size']:.3f}, acc={stats['best_accuracy']:.1%}")

    print("\n2. Menger Curvature:")
    for task, stats in results['menger_curvature'].items():
        if 'mean_curvature_stats' in stats:
            s = stats['mean_curvature_stats']
            print(f"   {task}: d={s['effect_size']:.3f}, p={s['p_value']:.4f}")

    print("\n3. Lyapunov Exponents:")
    for task, stats in results['lyapunov'].items():
        s = stats['mean_lyapunov_stats']
        print(f"   {task}: d={s['effect_size']:.3f}, p={s['p_value']:.4f}")

    print("\n4. Attractor Purity:")
    for task, stats in results['attractor'].items():
        print(f"   {task}: Mean purity {stats['mean_purity']:.1%}")

    print("\n5. Direction Transfer:")
    for key, stats in results['transfer'].items():
        status = "✓" if stats['test_accuracy'] > 0.55 else "✗"
        print(f"   {key}: {stats['test_accuracy']:.1%} {status}")


if __name__ == '__main__':
    main()
