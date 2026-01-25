#!/usr/bin/env python3
"""
H3 Remaining Analyses - Phase 3 Completion

Implements 4 analyses not yet completed:
1. PCA Velocity Field Analysis (semantic vs architectural flow)
2. Full Jacobian/Lyapunov Analysis (H_jac1, H_jac2, H_jac3)
3. Vector Field Decomposition (potential ratio, consistency)
4. Difficulty Stratification (cross-stratum transfer)

Usage:
    python h3_remaining_analyses.py --data-dir /path/to/trajectories_0shot --output-dir /path/to/results
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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

        # Get sequence lengths if available
        if 'sequence_lengths' in f:
            seq_lengths = f['sequence_lengths'][:]
        else:
            seq_lengths = np.full(len(trajectories), trajectories.shape[1])

        if max_samples and max_samples < len(trajectories):
            trajectories = trajectories[:max_samples]
            labels = labels[:max_samples]
            seq_lengths = seq_lengths[:max_samples]

    return trajectories.astype(np.float32), labels.astype(bool), seq_lengths


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


def compute_statistics(correct_vals, incorrect_vals):
    """Compute statistical comparison."""
    if len(correct_vals) < 2 or len(incorrect_vals) < 2:
        return {'d': 0, 'p': 1, 'correct_mean': 0, 'incorrect_mean': 0}

    d = cohens_d(correct_vals, incorrect_vals)
    _, p = stats.ttest_ind(correct_vals, incorrect_vals)

    return {
        'd': float(d),
        'p': float(p),
        'correct_mean': float(np.mean(correct_vals)),
        'incorrect_mean': float(np.mean(incorrect_vals)),
        'correct_std': float(np.std(correct_vals)),
        'incorrect_std': float(np.std(incorrect_vals))
    }


# ============================================================================
# ANALYSIS 1: PCA VELOCITY FIELD
# ============================================================================

def pca_velocity_analysis(trajectories, labels, k_arch=10, k_total=50):
    """
    PCA on velocity field to separate architectural vs semantic flow.

    Key insight: Top PCs capture architectural flow (shared across correct/incorrect).
    Remaining PCs capture semantic differences.
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    # Compute velocity field
    velocities = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]
    # Shape: (n_samples, seq_len, n_layers-1, d_model)

    # Flatten for PCA (subsample for memory)
    n_subsample = min(n_samples * seq_len, 50000)
    v_flat = velocities.reshape(-1, d_model)
    idx = np.random.choice(len(v_flat), n_subsample, replace=False)
    v_subsample = v_flat[idx]

    # PCA
    pca = PCA(n_components=min(k_total, d_model, n_subsample))
    pca.fit(v_subsample)

    # Split into architectural and semantic
    k_arch = min(k_arch, len(pca.components_))
    V_arch = pca.components_[:k_arch]
    V_sem = pca.components_[k_arch:]

    results = []
    for i in range(n_samples):
        v = velocities[i]  # (seq_len, n_layers-1, d_model)
        v_mean = v.mean(axis=0)  # Average over tokens: (n_layers-1, d_model)

        # Project onto subspaces
        v_arch = v_mean @ V_arch.T  # (n_layers-1, k_arch)
        v_sem = v_mean @ V_sem.T if len(V_sem) > 0 else np.zeros((v_mean.shape[0], 1))

        # Metrics
        arch_mag = np.linalg.norm(v_arch)
        sem_mag = np.linalg.norm(v_sem)
        semantic_ratio = sem_mag / (arch_mag + sem_mag + 1e-8)

        # Consistency in semantic subspace
        if sem_mag > 1e-8:
            v_sem_norm = v_sem / (np.linalg.norm(v_sem, axis=-1, keepdims=True) + 1e-8)
            sem_consistency = np.abs(v_sem_norm.mean(axis=0)).mean()
        else:
            sem_consistency = 0.0

        results.append({
            'sample_idx': i,
            'is_correct': bool(labels[i]),
            'arch_magnitude': float(arch_mag),
            'sem_magnitude': float(sem_mag),
            'semantic_ratio': float(semantic_ratio),
            'sem_consistency': float(sem_consistency)
        })

    df = pd.DataFrame(results)

    # Statistics
    stats_results = {
        'semantic_ratio': compute_statistics(
            df[df['is_correct']]['semantic_ratio'].values,
            df[~df['is_correct']]['semantic_ratio'].values
        ),
        'sem_consistency': compute_statistics(
            df[df['is_correct']]['sem_consistency'].values,
            df[~df['is_correct']]['sem_consistency'].values
        ),
        'arch_magnitude': compute_statistics(
            df[df['is_correct']]['arch_magnitude'].values,
            df[~df['is_correct']]['arch_magnitude'].values
        ),
        'explained_variance_arch': float(pca.explained_variance_ratio_[:k_arch].sum()),
        'explained_variance_total': float(pca.explained_variance_ratio_.sum()),
        'n_correct': int(labels.sum()),
        'n_incorrect': int((~labels).sum())
    }

    return df, stats_results


# ============================================================================
# ANALYSIS 2: FULL JACOBIAN / LYAPUNOV (with proper cross-validation)
# ============================================================================

def compute_layer_jacobian_svd(x_l, x_l1, k=50):
    """Estimate Jacobian spectrum between layers using randomized SVD."""
    delta = x_l1 - x_l

    try:
        _, s, _ = randomized_svd(delta, n_components=min(k, delta.shape[0]-1, delta.shape[1]),
                                  n_iter=2, random_state=42)
        input_scale = np.linalg.norm(x_l, 'fro') / np.sqrt(x_l.shape[0])
        if input_scale > 1e-10:
            return s / input_scale
        return s
    except:
        return np.ones(k)


def compute_directional_lyapunov(traj, error_dir, n_layers):
    """Compute directional Lyapunov exponent for a single trajectory."""
    dir_lyap = []
    for l in range(n_layers - 1):
        x_l = traj[:, l, :]
        x_l1 = traj[:, l + 1, :]

        proj_l = x_l @ error_dir
        proj_l1 = x_l1 @ error_dir
        var_l = np.var(proj_l)
        var_l1 = np.var(proj_l1)
        if var_l > 1e-10:
            dir_lyap.append(np.log(var_l1 / var_l + 1e-10) / 2)

    return np.mean(dir_lyap) if dir_lyap else 0.0


def full_jacobian_analysis(trajectories, labels, k_proj=50, n_folds=5, n_permutations=100):
    """
    Full Lyapunov spectrum analysis with PROPER CROSS-VALIDATION.

    Tests H_jac1 (max Lyapunov), H_jac2 (directional), H_jac3 (spectrum width).

    CRITICAL FIX: H_jac2 now uses k-fold CV to avoid circular analysis:
    - Error direction computed on train fold only
    - Directional Lyapunov tested on held-out fold

    Also computes:
    - Random direction baseline (what's d for random direction?)
    - Permutation null distribution (shuffle labels)
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    # =========================================================================
    # H_jac1 and H_jac3: These don't have circularity issues
    # =========================================================================
    results_jac1_jac3 = []
    for i in range(n_samples):
        traj = trajectories[i]

        layer_max_lyap = []
        layer_mean_lyap = []
        layer_width = []

        for l in range(n_layers - 1):
            x_l = traj[:, l, :]
            x_l1 = traj[:, l + 1, :]

            sv = compute_layer_jacobian_svd(x_l, x_l1, k=k_proj)
            lyap = np.log(sv + 1e-10)

            layer_max_lyap.append(lyap[0])
            layer_mean_lyap.append(np.mean(lyap))
            layer_width.append(np.std(lyap))

        results_jac1_jac3.append({
            'sample_idx': i,
            'is_correct': bool(labels[i]),
            'max_lyapunov': float(np.mean(layer_max_lyap)),
            'mean_lyapunov': float(np.mean(layer_mean_lyap)),
            'spectrum_width': float(np.mean(layer_width))
        })

    df_jac1_jac3 = pd.DataFrame(results_jac1_jac3)

    # =========================================================================
    # H_jac2: Cross-validated directional Lyapunov (FIXES CIRCULAR ANALYSIS)
    # =========================================================================
    print("      Running {}-fold cross-validation for H_jac2...".format(n_folds))

    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    fold_size = n_samples // n_folds
    cv_dir_lyap = np.zeros(n_samples)

    for fold in range(n_folds):
        # Split into train/test
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        # Compute error direction from TRAIN fold only
        train_labels = labels[train_idx]
        train_traj = trajectories[train_idx]

        train_correct = train_traj[train_labels]
        train_incorrect = train_traj[~train_labels]

        if len(train_correct) >= 3 and len(train_incorrect) >= 3:
            error_dir = train_incorrect.mean(axis=(0,1,2)) - train_correct.mean(axis=(0,1,2))
            error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)
        else:
            error_dir = np.random.randn(d_model)
            error_dir = error_dir / np.linalg.norm(error_dir)

        # Test on HELD-OUT fold
        for idx in test_idx:
            cv_dir_lyap[idx] = compute_directional_lyapunov(
                trajectories[idx], error_dir, n_layers
            )

    # Add CV results to dataframe
    df_jac1_jac3['directional_lyapunov_cv'] = cv_dir_lyap

    # =========================================================================
    # Random direction baseline
    # =========================================================================
    print("      Computing random direction baseline...")
    random_dir = np.random.randn(d_model)
    random_dir = random_dir / np.linalg.norm(random_dir)

    random_dir_lyap = np.array([
        compute_directional_lyapunov(trajectories[i], random_dir, n_layers)
        for i in range(n_samples)
    ])
    df_jac1_jac3['directional_lyapunov_random'] = random_dir_lyap

    # =========================================================================
    # Permutation test for null distribution
    # =========================================================================
    print("      Running {} permutation tests...".format(n_permutations))

    # Get the observed CV effect size
    correct_mask = df_jac1_jac3['is_correct'].values
    observed_d = cohens_d(
        cv_dir_lyap[~correct_mask],
        cv_dir_lyap[correct_mask]
    )

    # Permutation null
    perm_d_values = []
    for perm in range(n_permutations):
        perm_labels = np.random.permutation(labels)
        perm_d = cohens_d(
            cv_dir_lyap[~perm_labels],
            cv_dir_lyap[perm_labels]
        )
        perm_d_values.append(perm_d)

    perm_d_values = np.array(perm_d_values)
    perm_p_value = (np.abs(perm_d_values) >= np.abs(observed_d)).mean()

    # =========================================================================
    # Compute all statistics
    # =========================================================================
    correct = df_jac1_jac3[df_jac1_jac3['is_correct']]
    incorrect = df_jac1_jac3[~df_jac1_jac3['is_correct']]

    stats_results = {
        'h_jac1_max_lyapunov': compute_statistics(
            incorrect['max_lyapunov'].values,
            correct['max_lyapunov'].values
        ),
        'h_jac2_directional_cv': {
            **compute_statistics(
                incorrect['directional_lyapunov_cv'].values,
                correct['directional_lyapunov_cv'].values
            ),
            'method': f'{n_folds}-fold cross-validation',
            'permutation_p': float(perm_p_value),
            'n_permutations': n_permutations
        },
        'h_jac2_random_baseline': compute_statistics(
            incorrect['directional_lyapunov_random'].values,
            correct['directional_lyapunov_random'].values
        ),
        'h_jac3_spectrum_width': compute_statistics(
            correct['spectrum_width'].values,
            incorrect['spectrum_width'].values
        )
    }

    return df_jac1_jac3, stats_results


# ============================================================================
# ANALYSIS 3: VECTOR FIELD DECOMPOSITION
# ============================================================================

def vector_field_decomposition(trajectories, labels):
    """
    Analyze vector field structure: potential vs rotational flow.
    Uses SVD to estimate potential ratio (main flow direction dominance).
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    # Compute velocity field
    velocities = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]

    results = []
    for i in range(n_samples):
        v = velocities[i]  # (seq_len, n_layers-1, d_model)

        potential_ratios = []
        consistencies = []
        magnitudes = []

        for layer in range(v.shape[1]):
            v_layer = v[:, layer, :]  # (seq_len, d_model)

            # SVD: top SV = main flow direction (potential-like)
            try:
                _, s, _ = randomized_svd(v_layer, n_components=min(10, v_layer.shape[0]-1),
                                          n_iter=2, random_state=42)
                potential_ratio = s[0] / (s.sum() + 1e-8)
            except:
                potential_ratio = 0.5
            potential_ratios.append(potential_ratio)

            # Directional consistency
            v_norm = v_layer / (np.linalg.norm(v_layer, axis=-1, keepdims=True) + 1e-8)
            consistency = np.abs(v_norm.mean(axis=0)).mean()
            consistencies.append(consistency)

            magnitudes.append(np.linalg.norm(v_layer))

        results.append({
            'sample_idx': i,
            'is_correct': bool(labels[i]),
            'mean_potential_ratio': float(np.mean(potential_ratios)),
            'mean_consistency': float(np.mean(consistencies)),
            'mean_magnitude': float(np.mean(magnitudes)),
            'potential_trend': float(np.polyfit(range(len(potential_ratios)), potential_ratios, 1)[0])
        })

    df = pd.DataFrame(results)

    stats_results = {
        'potential_ratio': compute_statistics(
            df[df['is_correct']]['mean_potential_ratio'].values,
            df[~df['is_correct']]['mean_potential_ratio'].values
        ),
        'consistency': compute_statistics(
            df[df['is_correct']]['mean_consistency'].values,
            df[~df['is_correct']]['mean_consistency'].values
        ),
        'magnitude': compute_statistics(
            df[df['is_correct']]['mean_magnitude'].values,
            df[~df['is_correct']]['mean_magnitude'].values
        )
    }

    return df, stats_results


# ============================================================================
# ANALYSIS 4: DIFFICULTY STRATIFICATION
# ============================================================================

def difficulty_stratification(trajectories, labels, seq_lengths):
    """
    Test if error-direction transfers across difficulty strata.
    Uses sequence length as proxy for difficulty.
    """
    n_samples = len(trajectories)

    # Bin by sequence length (tertiles)
    tertiles = np.percentile(seq_lengths, [33, 66])

    strata = np.zeros(n_samples, dtype=int)
    strata[seq_lengths > tertiles[1]] = 2  # Hard
    strata[(seq_lengths > tertiles[0]) & (seq_lengths <= tertiles[1])] = 1  # Medium

    strata_names = ['easy', 'medium', 'hard']

    results = []
    for train_stratum in range(3):
        for test_stratum in range(3):
            train_mask = strata == train_stratum
            test_mask = strata == test_stratum

            train_correct = trajectories[train_mask & labels]
            train_incorrect = trajectories[train_mask & ~labels]

            if len(train_correct) < 5 or len(train_incorrect) < 5:
                continue

            # Extract direction from train stratum
            direction = train_incorrect.mean(axis=(0,1,2)) - train_correct.mean(axis=(0,1,2))
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # Test on test stratum
            test_traj = trajectories[test_mask]
            test_labels = labels[test_mask]

            if len(test_traj) < 5:
                continue

            # Project to last layer mean
            test_acts = test_traj[:, :, -1, :].mean(axis=1)
            test_proj = test_acts @ direction

            # Classify using median threshold
            threshold = np.median(test_proj)
            pred_incorrect = test_proj > threshold

            # Accuracy (direction points to incorrect)
            acc = ((pred_incorrect == ~test_labels).sum()) / len(test_labels)

            results.append({
                'train_stratum': strata_names[train_stratum],
                'test_stratum': strata_names[test_stratum],
                'accuracy': float(acc),
                'n_train': int(train_mask.sum()),
                'n_test': int(test_mask.sum()),
                'n_train_correct': int((train_mask & labels).sum()),
                'n_train_incorrect': int((train_mask & ~labels).sum())
            })

    df = pd.DataFrame(results)

    # Summary statistics
    if len(df) > 0:
        within_acc = df[df['train_stratum'] == df['test_stratum']]['accuracy'].mean()
        cross_acc = df[df['train_stratum'] != df['test_stratum']]['accuracy'].mean()
    else:
        within_acc = 0.5
        cross_acc = 0.5

    stats_results = {
        'within_stratum_accuracy': float(within_acc),
        'cross_stratum_accuracy': float(cross_acc),
        'difficulty_is_confound': bool(within_acc > cross_acc + 0.05)
    }

    return df, stats_results


# ============================================================================
# ANALYSIS 5: CROSS-DOMAIN DIRECTION TRANSFER
# ============================================================================

def cross_domain_direction_transfer(source_traj, source_labels, target_traj, target_labels):
    """
    Test if error direction from source domain works on target domain.

    This tests H2: Are dynamical signatures domain-invariant?
    """
    n_samples_source, seq_len, n_layers, d_model = source_traj.shape

    # Compute error direction from SOURCE domain
    source_correct = source_traj[source_labels]
    source_incorrect = source_traj[~source_labels]

    if len(source_correct) < 3 or len(source_incorrect) < 3:
        return None

    # Direction at last layer
    error_dir = source_incorrect.mean(axis=(0, 1, 2)) - source_correct.mean(axis=(0, 1, 2))
    error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)

    # Test on TARGET domain
    target_correct_lyap = []
    target_incorrect_lyap = []

    for i, (traj, label) in enumerate(zip(target_traj, target_labels)):
        dir_lyap = compute_directional_lyapunov(traj, error_dir, n_layers)
        if label:
            target_correct_lyap.append(dir_lyap)
        else:
            target_incorrect_lyap.append(dir_lyap)

    if len(target_correct_lyap) < 2 or len(target_incorrect_lyap) < 2:
        return None

    d = cohens_d(target_incorrect_lyap, target_correct_lyap)
    _, p = stats.ttest_ind(target_incorrect_lyap, target_correct_lyap)

    return {
        'd': float(d),
        'p': float(p),
        'n_source_correct': len(source_correct),
        'n_source_incorrect': len(source_incorrect),
        'n_target_correct': len(target_correct_lyap),
        'n_target_incorrect': len(target_incorrect_lyap)
    }


# ============================================================================
# ANALYSIS 6: CROSS-MODEL DIRECTION TRANSFER
# ============================================================================

def cross_model_direction_transfer(source_traj, source_labels, target_traj, target_labels):
    """
    Test if error direction from source MODEL works on target MODEL.

    E.g., does direction from base model work on rl_zero model?
    This tests whether training method changes the error-detection geometry.
    """
    # Same implementation as cross_domain, just different data sources
    return cross_domain_direction_transfer(source_traj, source_labels, target_traj, target_labels)


# ============================================================================
# ANALYSIS 7: LAYER-BY-LAYER SEPARATION
# ============================================================================

def layer_by_layer_analysis(trajectories, labels, n_folds=5):
    """
    Analyze where the correct/incorrect separation emerges across layers.

    For each layer:
    1. Compute error direction (with CV to avoid leakage)
    2. Compute directional Lyapunov at that layer
    3. Measure separation (Cohen's d)

    Returns profile of separation across layers.
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    # For each layer, compute CV-ed separation
    layer_results = []

    for layer in range(n_layers):
        # K-fold CV for this layer
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)

        fold_size = n_samples // n_folds
        cv_scores = []

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            # Compute direction from TRAIN at this layer
            train_labels = labels[train_idx]
            train_acts = trajectories[train_idx, :, layer, :].mean(axis=1)  # (n_train, d_model)

            train_correct = train_acts[train_labels]
            train_incorrect = train_acts[~train_labels]

            if len(train_correct) < 3 or len(train_incorrect) < 3:
                continue

            error_dir = train_incorrect.mean(axis=0) - train_correct.mean(axis=0)
            error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)

            # Test on held-out at this layer
            test_labels = labels[test_idx]
            test_acts = trajectories[test_idx, :, layer, :].mean(axis=1)
            test_proj = test_acts @ error_dir

            test_correct_proj = test_proj[test_labels]
            test_incorrect_proj = test_proj[~test_labels]

            if len(test_correct_proj) >= 2 and len(test_incorrect_proj) >= 2:
                d = cohens_d(test_incorrect_proj, test_correct_proj)
                cv_scores.append(d)

        if cv_scores:
            layer_results.append({
                'layer': layer,
                'd_mean': float(np.mean(cv_scores)),
                'd_std': float(np.std(cv_scores)),
                'n_folds_valid': len(cv_scores)
            })
        else:
            layer_results.append({
                'layer': layer,
                'd_mean': 0.0,
                'd_std': 0.0,
                'n_folds_valid': 0
            })

    # Find peak separation layer
    d_values = [r['d_mean'] for r in layer_results]
    peak_layer = int(np.argmax(np.abs(d_values)))

    return {
        'layer_profile': layer_results,
        'peak_layer': peak_layer,
        'peak_d': float(d_values[peak_layer]),
        'early_mean_d': float(np.mean(d_values[:n_layers//3])),
        'middle_mean_d': float(np.mean(d_values[n_layers//3:2*n_layers//3])),
        'late_mean_d': float(np.mean(d_values[2*n_layers//3:]))
    }


# ============================================================================
# MAIN
# ============================================================================

def analyze_single_file(filepath, task_name, max_samples=200):
    """Run all 4 analyses on a single file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {task_name}")
    print(f"File: {filepath}")
    print('='*60)

    try:
        traj, labels, seq_lengths = load_trajectories(filepath, max_samples)
    except Exception as e:
        print(f"ERROR loading: {e}")
        return None

    print(f"Shape: {traj.shape}")
    print(f"Correct: {labels.sum()}/{len(labels)} ({100*labels.mean():.1f}%)")

    if labels.sum() < 5 or (~labels).sum() < 5:
        print("SKIPPED: Insufficient samples for analysis")
        return None

    results = {'task': task_name, 'n_samples': len(traj), 'n_correct': int(labels.sum())}

    # Analysis 1: PCA Velocity
    print("\n1. PCA Velocity Field Analysis...")
    try:
        _, pca_stats = pca_velocity_analysis(traj, labels)
        results['pca_velocity'] = pca_stats
        print(f"   Semantic ratio: d={pca_stats['semantic_ratio']['d']:.3f}, p={pca_stats['semantic_ratio']['p']:.4f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        results['pca_velocity'] = {'error': str(e)}

    # Analysis 2: Full Jacobian (with cross-validation for H_jac2)
    print("\n2. Full Jacobian/Lyapunov Analysis (CV for H_jac2)...")
    try:
        _, jac_stats = full_jacobian_analysis(traj, labels, n_folds=5, n_permutations=100)
        results['jacobian'] = jac_stats
        print(f"   H_jac1 (max λ): d={jac_stats['h_jac1_max_lyapunov']['d']:.3f}, p={jac_stats['h_jac1_max_lyapunov']['p']:.4f}")
        cv_stats = jac_stats['h_jac2_directional_cv']
        print(f"   H_jac2 (dir λ, CV): d={cv_stats['d']:.3f}, p={cv_stats['p']:.4f}, perm_p={cv_stats['permutation_p']:.4f}")
        rand_stats = jac_stats['h_jac2_random_baseline']
        print(f"   H_jac2 (random dir): d={rand_stats['d']:.3f}, p={rand_stats['p']:.4f} [BASELINE]")
        print(f"   H_jac3 (width): d={jac_stats['h_jac3_spectrum_width']['d']:.3f}, p={jac_stats['h_jac3_spectrum_width']['p']:.4f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['jacobian'] = {'error': str(e)}

    # Analysis 3: Vector Field
    print("\n3. Vector Field Decomposition...")
    try:
        _, vf_stats = vector_field_decomposition(traj, labels)
        results['vector_field'] = vf_stats
        print(f"   Potential ratio: d={vf_stats['potential_ratio']['d']:.3f}, p={vf_stats['potential_ratio']['p']:.4f}")
        print(f"   Consistency: d={vf_stats['consistency']['d']:.3f}, p={vf_stats['consistency']['p']:.4f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        results['vector_field'] = {'error': str(e)}

    # Analysis 4: Difficulty Stratification
    print("\n4. Difficulty Stratification...")
    try:
        strat_df, strat_stats = difficulty_stratification(traj, labels, seq_lengths)
        results['difficulty'] = strat_stats
        results['difficulty']['transfer_matrix'] = strat_df.to_dict('records') if len(strat_df) > 0 else []
        print(f"   Within-stratum acc: {strat_stats['within_stratum_accuracy']:.1%}")
        print(f"   Cross-stratum acc: {strat_stats['cross_stratum_accuracy']:.1%}")
        print(f"   Difficulty is confound: {strat_stats['difficulty_is_confound']}")
    except Exception as e:
        print(f"   ERROR: {e}")
        results['difficulty'] = {'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description='H3 Remaining Analyses')
    parser.add_argument('--data-dir', required=True, help='Directory with trajectory files')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--models', default='olmo3_base,olmo3_sft,olmo3_rl_zero,olmo3_think',
                        help='Comma-separated model names')
    parser.add_argument('--tasks', default='gsm8k,humaneval,logiqa',
                        help='Comma-separated task names')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Max samples per file (for memory)')

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    tasks = [t.strip() for t in args.tasks.split(',')]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("H3 REMAINING ANALYSES")
    print("="*60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Data dir: {args.data_dir}")
    print(f"Models: {models}")
    print(f"Tasks: {tasks}")
    print(f"Max samples: {args.max_samples}")

    all_results = {}

    for model in models:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model}")
        print('#'*60)

        model_results = {}

        for task in tasks:
            filepath = Path(args.data_dir) / model / f'{task}_trajectories.h5'

            if not filepath.exists():
                print(f"\nWARNING: {filepath} not found, skipping")
                continue

            result = analyze_single_file(str(filepath), f"{model}/{task}", args.max_samples)

            if result is not None:
                model_results[task] = result
                all_results[f"{model}/{task}"] = result

        # Save per-model results
        model_output = output_dir / f"{model}_h3_results.json"
        with open(model_output, 'w') as f:
            json.dump(model_results, f, indent=2)
        print(f"\nSaved: {model_output}")

    # Save combined results
    combined_output = output_dir / "h3_all_results.json"
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined: {combined_output}")

    # =========================================================================
    # CROSS-DOMAIN AND CROSS-MODEL TRANSFER ANALYSIS
    # =========================================================================
    print("\n" + "="*60)
    print("CROSS-DOMAIN DIRECTION TRANSFER")
    print("="*60)

    # Load all data for cross-analysis
    cross_data = {}
    for model in models:
        for task in tasks:
            filepath = Path(args.data_dir) / model / f'{task}_trajectories.h5'
            if filepath.exists():
                try:
                    traj, labels, _ = load_trajectories(str(filepath), args.max_samples)
                    if labels.sum() >= 5 and (~labels).sum() >= 5:
                        cross_data[f"{model}/{task}"] = (traj, labels)
                        print(f"  Loaded {model}/{task}: {traj.shape}, {labels.sum()} correct")
                except Exception as e:
                    print(f"  ERROR loading {model}/{task}: {e}")

    # Cross-domain transfer (same model, different tasks)
    cross_domain_results = {}
    print("\n5. Cross-Domain Direction Transfer (same model):")
    for model in models:
        model_tasks = [k for k in cross_data.keys() if k.startswith(model)]
        for source_key in model_tasks:
            for target_key in model_tasks:
                if source_key == target_key:
                    continue
                source_traj, source_labels = cross_data[source_key]
                target_traj, target_labels = cross_data[target_key]

                result = cross_domain_direction_transfer(
                    source_traj, source_labels, target_traj, target_labels
                )
                if result:
                    key = f"{source_key}→{target_key.split('/')[-1]}"
                    cross_domain_results[key] = result
                    sig = "✓" if result['p'] < 0.05 else ""
                    print(f"   {key}: d={result['d']:.3f}, p={result['p']:.4f} {sig}")

    all_results['cross_domain_transfer'] = cross_domain_results

    # Cross-model transfer (same task, different models)
    cross_model_results = {}
    print("\n6. Cross-Model Direction Transfer (same task):")
    for task in tasks:
        task_models = [k for k in cross_data.keys() if k.endswith(task)]
        for source_key in task_models:
            for target_key in task_models:
                if source_key == target_key:
                    continue
                source_traj, source_labels = cross_data[source_key]
                target_traj, target_labels = cross_data[target_key]

                result = cross_model_direction_transfer(
                    source_traj, source_labels, target_traj, target_labels
                )
                if result:
                    source_model = source_key.split('/')[0].replace('olmo3_', '')
                    target_model = target_key.split('/')[0].replace('olmo3_', '')
                    key = f"{source_model}→{target_model}/{task}"
                    cross_model_results[key] = result
                    sig = "✓" if result['p'] < 0.05 else ""
                    print(f"   {key}: d={result['d']:.3f}, p={result['p']:.4f} {sig}")

    all_results['cross_model_transfer'] = cross_model_results

    # Layer-by-layer analysis
    layer_results = {}
    print("\n7. Layer-by-Layer Separation Analysis:")
    for key, (traj, labels) in cross_data.items():
        print(f"\n   {key}:")
        result = layer_by_layer_analysis(traj, labels)
        layer_results[key] = result
        print(f"      Peak layer: {result['peak_layer']} (d={result['peak_d']:.3f})")
        print(f"      Early/Mid/Late: {result['early_mean_d']:.3f} / {result['middle_mean_d']:.3f} / {result['late_mean_d']:.3f}")

    all_results['layer_by_layer'] = layer_results

    # Save updated results
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nUpdated: {combined_output}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (H_jac2 now uses 5-fold CV)")
    print("="*60)

    print("\n| Model/Task           | PCA Sem | Jac1   | Jac2 CV | Jac2 Rand | Jac3   | VF Pot  | Diff? |")
    print("|----------------------|---------|--------|---------|-----------|--------|---------|-------|")

    for key, r in sorted(all_results.items()):
        pca_d = r.get('pca_velocity', {}).get('semantic_ratio', {}).get('d', 'N/A')
        jac1_d = r.get('jacobian', {}).get('h_jac1_max_lyapunov', {}).get('d', 'N/A')
        jac2_cv_d = r.get('jacobian', {}).get('h_jac2_directional_cv', {}).get('d', 'N/A')
        jac2_rand_d = r.get('jacobian', {}).get('h_jac2_random_baseline', {}).get('d', 'N/A')
        jac3_d = r.get('jacobian', {}).get('h_jac3_spectrum_width', {}).get('d', 'N/A')
        vf_d = r.get('vector_field', {}).get('potential_ratio', {}).get('d', 'N/A')
        diff = r.get('difficulty', {}).get('difficulty_is_confound', 'N/A')

        pca_str = f"{pca_d:.2f}" if isinstance(pca_d, float) else str(pca_d)[:7]
        jac1_str = f"{jac1_d:.2f}" if isinstance(jac1_d, float) else str(jac1_d)[:6]
        jac2_cv_str = f"{jac2_cv_d:.2f}" if isinstance(jac2_cv_d, float) else str(jac2_cv_d)[:7]
        jac2_rand_str = f"{jac2_rand_d:.2f}" if isinstance(jac2_rand_d, float) else str(jac2_rand_d)[:9]
        jac3_str = f"{jac3_d:.2f}" if isinstance(jac3_d, float) else str(jac3_d)[:6]
        vf_str = f"{vf_d:.2f}" if isinstance(vf_d, float) else str(vf_d)[:7]
        diff_str = "Yes" if diff == True else ("No" if diff == False else "N/A")

        print(f"| {key:20s} | {pca_str:>7s} | {jac1_str:>6s} | {jac2_cv_str:>7s} | {jac2_rand_str:>9s} | {jac3_str:>6s} | {vf_str:>7s} | {diff_str:>5s} |")

    print("\nNote: Jac2 CV = cross-validated directional Lyapunov (true effect)")
    print("      Jac2 Rand = random direction baseline (should be ~0)")

    # Print cross-domain summary
    print("\n" + "="*60)
    print("CROSS-DOMAIN TRANSFER SUMMARY")
    print("="*60)
    if 'cross_domain_transfer' in all_results:
        sig_count = sum(1 for r in all_results['cross_domain_transfer'].values() if r['p'] < 0.05)
        total = len(all_results['cross_domain_transfer'])
        print(f"Significant transfers: {sig_count}/{total}")
        for key, r in sorted(all_results['cross_domain_transfer'].items()):
            sig = "✓" if r['p'] < 0.05 else ""
            print(f"  {key}: d={r['d']:.3f}, p={r['p']:.4f} {sig}")

    print("\n" + "="*60)
    print("CROSS-MODEL TRANSFER SUMMARY")
    print("="*60)
    if 'cross_model_transfer' in all_results:
        sig_count = sum(1 for r in all_results['cross_model_transfer'].values() if r['p'] < 0.05)
        total = len(all_results['cross_model_transfer'])
        print(f"Significant transfers: {sig_count}/{total}")
        for key, r in sorted(all_results['cross_model_transfer'].items()):
            sig = "✓" if r['p'] < 0.05 else ""
            print(f"  {key}: d={r['d']:.3f}, p={r['p']:.4f} {sig}")

    print("\n" + "="*60)
    print("LAYER-BY-LAYER SUMMARY")
    print("="*60)
    if 'layer_by_layer' in all_results:
        for key, r in sorted(all_results['layer_by_layer'].items()):
            print(f"  {key}: peak at L{r['peak_layer']} (d={r['peak_d']:.3f}), late_d={r['late_mean_d']:.3f}")

    print(f"\nComplete. Results in {output_dir}")


if __name__ == '__main__':
    main()
