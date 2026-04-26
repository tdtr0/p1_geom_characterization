#!/usr/bin/env python3
"""
Cross-Model Alignment Analysis

Tests whether different models (base, SFT, RL-Zero, Think) have aligned representations
when processing the SAME inputs. This answers whether RL-Zero destroys cross-domain
transfer through rotation (features preserved but moved) or destruction (features replaced).

Analyses:
1. Cross-Model Procrustes: Measure rotation vs content change
2. CKA Similarity: Rotation-invariant representation similarity
3. Error Direction Rotation Test: Does R @ e_base align with e_target?
4. Eigenvector Correspondence: Which base eigenvectors map to which target eigenvectors?
5. Cross-Model Probe Transfer: Train on base, test on target (with/without rotation)

Usage:
    python cross_model_alignment.py \
        --data-dir /data/thanhdo/trajectories_0shot \
        --tasks gsm8k,humaneval \
        --output-dir results/cross_model_alignment
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.extmath import randomized_svd


# =============================================================================
# DATA LOADING
# =============================================================================

def load_trajectories(filepath, max_samples=None):
    """Load trajectories and labels from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        traj = f['trajectories'][:]
        if 'is_correct' in f:
            labels = f['is_correct'][:]
        elif 'correct' in f:
            labels = f['correct'][:]
        else:
            raise KeyError(f"No correctness labels. Keys: {list(f.keys())}")

    if max_samples and len(traj) > max_samples:
        indices = np.random.RandomState(42).choice(len(traj), max_samples, replace=False)
        traj = traj[indices]
        labels = labels[indices]

    return traj.astype(np.float32), labels.astype(bool)


def get_mean_activations(trajectories, layer=-1):
    """Get mean activations across sequence for a specific layer."""
    # trajectories: (n_samples, seq_len, n_layers, d_model)
    # Return: (n_samples, d_model)
    return trajectories[:, :, layer, :].mean(axis=1)


# =============================================================================
# PROCRUSTES ANALYSIS
# =============================================================================

def procrustes_alignment(A_source, A_target, center=True):
    """
    Compute Procrustes alignment from source to target.

    Returns:
        R: Optimal rotation matrix
        scale: Optimal scale factor
        residual: ||A_target - scale * A_source @ R|| / ||A_target||
        rotation_norm: ||R - I|| (Frobenius norm, measures how much rotation)
    """
    if center:
        A_source = A_source - A_source.mean(axis=0)
        A_target = A_target - A_target.mean(axis=0)

    # Procrustes: find R such that A_source @ R ≈ A_target
    R, scale = orthogonal_procrustes(A_source, A_target)

    # Compute residual
    A_predicted = scale * A_source @ R
    residual_norm = np.linalg.norm(A_target - A_predicted, 'fro') / (np.linalg.norm(A_target, 'fro') + 1e-10)

    # Compute rotation magnitude
    identity = np.eye(R.shape[0])
    rotation_norm = np.linalg.norm(R - identity, 'fro')

    return R, scale, residual_norm, rotation_norm


# =============================================================================
# CKA SIMILARITY
# =============================================================================

def linear_CKA(X, Y):
    """
    Compute linear Centered Kernel Alignment (CKA).

    CKA is rotation-invariant and measures how similar two representations are.
    Range: [0, 1], 1 = identical (up to rotation/scale), 0 = orthogonal.

    Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
    """
    # Center the data
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Compute Gram matrices
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_xy = np.trace(K_X @ K_Y)
    hsic_xx = np.trace(K_X @ K_X)
    hsic_yy = np.trace(K_Y @ K_Y)

    # CKA
    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)

    return cka


# =============================================================================
# ERROR DIRECTION ANALYSIS
# =============================================================================

def compute_error_direction(activations, labels):
    """Compute error direction as difference in means."""
    correct_mean = activations[labels].mean(axis=0)
    incorrect_mean = activations[~labels].mean(axis=0)
    error_dir = incorrect_mean - correct_mean
    error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)
    return error_dir


def error_direction_rotation_test(e_source, e_target, R):
    """
    Test if source error direction, after rotation, aligns with target error direction.

    High cos(R @ e_source, e_target) = direction preserved but rotated
    Low cos(R @ e_source, e_target) = direction destroyed
    """
    e_source_rotated = R @ e_source
    e_source_rotated = e_source_rotated / (np.linalg.norm(e_source_rotated) + 1e-10)

    # Direct alignment (without rotation)
    cos_direct = np.abs(np.dot(e_source, e_target))

    # Alignment after Procrustes rotation
    cos_rotated = np.abs(np.dot(e_source_rotated, e_target))

    return {
        'cos_direct': float(cos_direct),
        'cos_rotated': float(cos_rotated),
        'improvement': float(cos_rotated - cos_direct)
    }


# =============================================================================
# EIGENVECTOR CORRESPONDENCE
# =============================================================================

def eigenvector_correspondence(A_source, A_target, n_components=50):
    """
    Compute correspondence matrix between eigenvectors of two activation matrices.

    High diagonal dominance = eigenvectors preserved
    High off-diagonal = eigenvectors shuffled/mixed

    Returns:
        correspondence: (n_components, n_components) matrix of cosine similarities
        diagonal_dominance: Mean of diagonal values
        row_entropy: Mean entropy of row distributions
    """
    # SVD on each
    U_s, S_s, Vt_s = randomized_svd(A_source - A_source.mean(axis=0),
                                     n_components=n_components, n_iter=3, random_state=42)
    U_t, S_t, Vt_t = randomized_svd(A_target - A_target.mean(axis=0),
                                     n_components=n_components, n_iter=3, random_state=42)

    # Correspondence matrix: C[i,j] = |cos(v_source_i, v_target_j)|
    # Using right singular vectors (directions in feature space)
    correspondence = np.abs(Vt_s @ Vt_t.T)  # (n_components, n_components)

    # Diagonal dominance: how much do eigenvectors stay in place?
    diagonal_dominance = np.mean(np.diag(correspondence))

    # Row entropy: how spread out is each source eigenvector's correspondence?
    # Low entropy = maps to few target eigenvectors (preserved)
    # High entropy = maps to many target eigenvectors (shuffled)
    row_probs = correspondence / (correspondence.sum(axis=1, keepdims=True) + 1e-10)
    row_entropy = -np.sum(row_probs * np.log(row_probs + 1e-10), axis=1)
    mean_entropy = np.mean(row_entropy)
    max_entropy = np.log(n_components)  # Uniform distribution
    normalized_entropy = mean_entropy / max_entropy

    # Singular value alignment (how similar are the spectra?)
    sv_correlation = np.corrcoef(S_s, S_t)[0, 1]

    return {
        'correspondence': correspondence,
        'diagonal_dominance': float(diagonal_dominance),
        'normalized_entropy': float(normalized_entropy),
        'sv_correlation': float(sv_correlation)
    }


# =============================================================================
# CROSS-MODEL PROBE TRANSFER
# =============================================================================

def cross_model_probe_transfer(A_source, labels_source, A_target, labels_target, R=None):
    """
    Train linear probe on source model, test on target model.

    Returns AUC for:
    - Direct transfer (no rotation)
    - Rotated transfer (if R provided)
    - Within-target (upper bound)
    """
    results = {}

    # Train on source
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(A_source, labels_source)

    # Test on target (direct transfer)
    try:
        probs_direct = clf.predict_proba(A_target)[:, 1]
        auc_direct = roc_auc_score(labels_target, probs_direct)
    except:
        auc_direct = 0.5
    results['auc_direct_transfer'] = float(auc_direct)

    # Test on target (with Procrustes rotation)
    if R is not None:
        A_target_rotated = A_target @ R.T  # Rotate target to align with source
        try:
            probs_rotated = clf.predict_proba(A_target_rotated)[:, 1]
            auc_rotated = roc_auc_score(labels_target, probs_rotated)
        except:
            auc_rotated = 0.5
        results['auc_rotated_transfer'] = float(auc_rotated)

    # Upper bound: train and test on target
    try:
        auc_within = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=42),
            A_target, labels_target,
            scoring='roc_auc', cv=5
        ).mean()
    except:
        auc_within = 0.5
    results['auc_within_target'] = float(auc_within)

    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_model_pair(source_path, target_path, source_name, target_name, max_samples=200):
    """Run all analyses for a single model pair."""
    print(f"\n  Analyzing {source_name} -> {target_name}...")

    # Load data
    try:
        traj_source, labels_source = load_trajectories(source_path, max_samples)
        traj_target, labels_target = load_trajectories(target_path, max_samples)
    except Exception as e:
        print(f"    ERROR loading: {e}")
        return None

    # Use minimum of both sample counts
    n_samples = min(len(traj_source), len(traj_target))
    traj_source = traj_source[:n_samples]
    traj_target = traj_target[:n_samples]
    labels_source = labels_source[:n_samples]
    labels_target = labels_target[:n_samples]

    print(f"    Using {n_samples} matched samples")
    print(f"    Source: {labels_source.sum()} correct, {(~labels_source).sum()} incorrect")
    print(f"    Target: {labels_target.sum()} correct, {(~labels_target).sum()} incorrect")

    results = {
        'source_model': source_name,
        'target_model': target_name,
        'n_samples': n_samples
    }

    # Get activations (last layer, mean over sequence)
    A_source = get_mean_activations(traj_source, layer=-1)
    A_target = get_mean_activations(traj_target, layer=-1)

    # 1. Procrustes alignment
    print("    Computing Procrustes alignment...")
    R, scale, residual, rotation_norm = procrustes_alignment(A_source, A_target)
    results['procrustes_residual'] = float(residual)
    results['procrustes_scale'] = float(scale)
    results['rotation_norm'] = float(rotation_norm)
    print(f"      Residual: {residual:.4f}, Rotation norm: {rotation_norm:.4f}")

    # 2. CKA similarity
    print("    Computing CKA similarity...")
    cka = linear_CKA(A_source, A_target)
    results['cka_similarity'] = float(cka)
    print(f"      CKA: {cka:.4f}")

    # 3. Error direction rotation test
    print("    Testing error direction rotation...")
    e_source = compute_error_direction(A_source, labels_source)
    e_target = compute_error_direction(A_target, labels_target)
    rotation_test = error_direction_rotation_test(e_source, e_target, R)
    results.update({f'error_dir_{k}': v for k, v in rotation_test.items()})
    print(f"      Direct alignment: {rotation_test['cos_direct']:.4f}")
    print(f"      After rotation: {rotation_test['cos_rotated']:.4f}")

    # 4. Eigenvector correspondence
    print("    Computing eigenvector correspondence...")
    eigenvec_results = eigenvector_correspondence(A_source, A_target)
    results['eigenvec_diagonal_dominance'] = eigenvec_results['diagonal_dominance']
    results['eigenvec_normalized_entropy'] = eigenvec_results['normalized_entropy']
    results['eigenvec_sv_correlation'] = eigenvec_results['sv_correlation']
    print(f"      Diagonal dominance: {eigenvec_results['diagonal_dominance']:.4f}")
    print(f"      Normalized entropy: {eigenvec_results['normalized_entropy']:.4f}")

    # 5. Cross-model probe transfer
    print("    Testing cross-model probe transfer...")
    probe_results = cross_model_probe_transfer(A_source, labels_source, A_target, labels_target, R)
    results.update({f'probe_{k}': v for k, v in probe_results.items()})
    print(f"      Direct transfer AUC: {probe_results['auc_direct_transfer']:.4f}")
    if 'auc_rotated_transfer' in probe_results:
        print(f"      Rotated transfer AUC: {probe_results['auc_rotated_transfer']:.4f}")
    print(f"      Within-target AUC: {probe_results['auc_within_target']:.4f}")

    return results, eigenvec_results['correspondence']


def run_analysis(data_dir, tasks, models, output_dir, max_samples=200):
    """Run cross-model alignment analysis for all model pairs."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    correspondence_matrices = {}

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        # Use base as source, compare to all others
        source_model = 'olmo3_base'
        source_path = data_dir / source_model / f"{task}_trajectories.h5"

        if not source_path.exists():
            print(f"  SKIP: {source_model}/{task} not found")
            continue

        for target_model in models:
            if target_model == source_model:
                continue

            target_path = data_dir / target_model / f"{task}_trajectories.h5"

            if not target_path.exists():
                print(f"  SKIP: {target_model}/{task} not found")
                continue

            result = analyze_model_pair(
                source_path, target_path,
                source_model, target_model,
                max_samples
            )

            if result is not None:
                results_dict, corr_matrix = result
                results_dict['task'] = task
                all_results.append(results_dict)
                correspondence_matrices[f"{task}_{source_model}_{target_model}"] = corr_matrix

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)

        # Summary CSV
        summary_path = output_dir / 'cross_model_alignment_summary.csv'
        df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to {summary_path}")

        # Print summary table
        print("\n" + "="*80)
        print("CROSS-MODEL ALIGNMENT SUMMARY")
        print("="*80)

        cols = ['task', 'target_model', 'procrustes_residual', 'cka_similarity',
                'error_dir_cos_direct', 'error_dir_cos_rotated',
                'eigenvec_diagonal_dominance', 'probe_auc_direct_transfer']
        print(df[cols].to_string(index=False))

        # Save correspondence matrices
        np.savez(
            output_dir / 'eigenvector_correspondence.npz',
            **correspondence_matrices
        )
        print(f"\nSaved correspondence matrices to {output_dir / 'eigenvector_correspondence.npz'}")

        # Generate interpretation
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)

        for task in tasks:
            task_df = df[df['task'] == task]
            if len(task_df) == 0:
                continue

            print(f"\n{task}:")
            for _, row in task_df.iterrows():
                target = row['target_model']
                residual = row['procrustes_residual']
                cka = row['cka_similarity']
                cos_direct = row['error_dir_cos_direct']
                cos_rotated = row['error_dir_cos_rotated']
                diag_dom = row['eigenvec_diagonal_dominance']

                # Interpret
                if residual < 0.3 and cka > 0.8:
                    structure = "PRESERVED (low residual, high CKA)"
                elif residual > 0.5 or cka < 0.5:
                    structure = "CHANGED (high residual or low CKA)"
                else:
                    structure = "PARTIALLY PRESERVED"

                if cos_rotated > 0.5:
                    error_dir = "PRESERVED (rotated)"
                elif cos_rotated > 0.3:
                    error_dir = "PARTIALLY PRESERVED"
                else:
                    error_dir = "DESTROYED"

                improvement = cos_rotated - cos_direct
                if improvement > 0.1:
                    rotation_helps = "YES (rotation helps significantly)"
                elif improvement > 0:
                    rotation_helps = "SLIGHTLY (rotation helps a bit)"
                else:
                    rotation_helps = "NO (rotation doesn't help)"

                print(f"  base -> {target}:")
                print(f"    Overall structure: {structure}")
                print(f"    Error direction: {error_dir}")
                print(f"    Does Procrustes rotation help?: {rotation_helps}")
                print(f"    Eigenvector preservation: {diag_dom:.2%} diagonal dominance")

    return df if all_results else None


def main():
    parser = argparse.ArgumentParser(description='Cross-Model Alignment Analysis')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing trajectory HDF5 files')
    parser.add_argument('--tasks', type=str, default='gsm8k,humaneval',
                        help='Comma-separated list of tasks')
    parser.add_argument('--models', type=str, default='olmo3_base,olmo3_sft,olmo3_rl_zero,olmo3_think',
                        help='Comma-separated list of models')
    parser.add_argument('--output-dir', type=str, default='results/cross_model_alignment',
                        help='Output directory')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Maximum samples per model/task')

    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(',')]
    models = [m.strip() for m in args.models.split(',')]

    print(f"\n{'='*60}")
    print("Cross-Model Alignment Analysis")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Tasks: {tasks}")
    print(f"Models: {models}")
    print(f"Max samples: {args.max_samples}")
    print(f"{'='*60}")
    print()
    print("Analyses:")
    print("  1. Procrustes: Measure rotation vs content change")
    print("  2. CKA: Rotation-invariant similarity")
    print("  3. Error Direction: Does rotation preserve error direction?")
    print("  4. Eigenvector Correspondence: Which maps to which?")
    print("  5. Probe Transfer: Does probe trained on base work on target?")
    print(f"{'='*60}")

    run_analysis(
        data_dir=args.data_dir,
        tasks=tasks,
        models=models,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
