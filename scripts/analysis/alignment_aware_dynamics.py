#!/usr/bin/env python3
"""
Alignment-Aware Dynamics Analysis: Procrustes + SVCCA (Fast Version)

This script addresses the fundamental problem with Lyapunov/Jacobian analysis
on transformers: consecutive layer representations are nearly orthogonal
(cos_sim ≈ 0.1), which makes the Jacobian essentially a rotation with SVs ≈ 1.

Solution: Use alignment-aware methods that account for subspace rotation
before measuring dynamics.

Methods:
1. Procrustes Alignment: Find optimal rotation R such that R @ X_l ≈ X_{l+1}
   - Unsupervised (no labels used in alignment)
   - Measures reconstruction error as "true transformation" vs "just rotation"

2. SVCCA: Find maximally correlated subspaces between layers
   - Unsupervised alignment
   - Compare canonical correlations for correct vs incorrect

Usage:
    python alignment_aware_dynamics.py \
        --data-dir /path/to/trajectories_0shot \
        --models olmo3_base,olmo3_sft,olmo3_rl_zero \
        --output-dir results \
        --max-samples 100
"""

import argparse
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd


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


# =============================================================================
# PROCRUSTES ANALYSIS (Pooled Per-Layer)
# =============================================================================

def procrustes_per_layer(trajectories, labels, n_pca=256):
    """
    Compute Procrustes statistics per layer, pooling across samples.

    Uses PCA reduction first to make Procrustes tractable.

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean
        n_pca: Number of PCA components to reduce to

    Returns:
        DataFrame with per-layer statistics
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape
    n_pca = min(n_pca, n_samples * seq_len - 1, d_model)

    results = []
    for l in range(n_layers - 1):
        print(f"      Layer {l}/{n_layers-2}...")

        # Pool activations from all samples at this layer
        X_l_all = trajectories[:, :, l, :].reshape(-1, d_model)  # (n*seq, d_model)
        X_l1_all = trajectories[:, :, l + 1, :].reshape(-1, d_model)

        # PCA reduction (fit on combined data for fair comparison)
        X_combined = np.vstack([X_l_all, X_l1_all])
        X_combined_c = X_combined - X_combined.mean(axis=0)

        try:
            U, s, Vt = randomized_svd(X_combined_c, n_components=n_pca, n_iter=3, random_state=42)
            pca_components = Vt  # (n_pca, d_model)
        except Exception as e:
            print(f"        PCA failed: {e}")
            continue

        # Project both layers
        X_l_pca = (X_l_all - X_combined.mean(axis=0)[:d_model]) @ pca_components.T  # (n*seq, n_pca)
        X_l1_pca = (X_l1_all - X_combined.mean(axis=0)[:d_model]) @ pca_components.T

        # Get indices for correct/incorrect
        correct_mask = np.repeat(labels, seq_len)

        # Fit Procrustes on ALL data (unsupervised)
        try:
            X_l_c = X_l_pca - X_l_pca.mean(axis=0)
            X_l1_c = X_l1_pca - X_l1_pca.mean(axis=0)
            R, scale = orthogonal_procrustes(X_l_c, X_l1_c)

            # Compute reconstruction error
            X_l1_pred = X_l_c @ R * scale
            global_error = np.linalg.norm(X_l1_c - X_l1_pred, 'fro') / (np.linalg.norm(X_l1_c, 'fro') + 1e-10)

            # Per-token distances (normalized)
            per_token_dist = np.linalg.norm(X_l1_c - X_l1_pred, axis=1) / (np.linalg.norm(X_l1_c, axis=1) + 1e-10)

        except Exception as e:
            print(f"        Procrustes failed: {e}")
            continue

        # Split by correctness
        correct_distances = per_token_dist[correct_mask]
        incorrect_distances = per_token_dist[~correct_mask]

        d = cohens_d(correct_distances, incorrect_distances)
        _, p = stats.ttest_ind(correct_distances, incorrect_distances)

        results.append({
            'layer': l,
            'global_reconstruction_error': float(global_error),
            'procrustes_scale': float(scale),
            'correct_mean_dist': float(np.mean(correct_distances)),
            'incorrect_mean_dist': float(np.mean(incorrect_distances)),
            'cohens_d': float(d),
            'p_value': float(p)
        })

    return pd.DataFrame(results)


# =============================================================================
# SVCCA ANALYSIS (Pooled Per-Layer)
# =============================================================================

def svcca_per_layer(trajectories, labels, n_pca=256, n_cca=20):
    """
    Compute SVCCA statistics per layer, pooling across samples.

    Fit CCA on ALL data (unsupervised), then compare correct vs incorrect.

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean
        n_pca: Number of PCA components for SVD step
        n_cca: Number of CCA components

    Returns:
        DataFrame with per-layer statistics
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape
    n_pca = min(n_pca, n_samples * seq_len - 1, d_model)
    n_cca = min(n_cca, n_pca - 1)

    results = []
    for l in range(n_layers - 1):
        print(f"      Layer {l}/{n_layers-2}...")

        # Pool activations from all samples at this layer
        X_l_all = trajectories[:, :, l, :].reshape(-1, d_model)
        X_l1_all = trajectories[:, :, l + 1, :].reshape(-1, d_model)

        # Get indices for correct/incorrect
        correct_mask = np.repeat(labels, seq_len)

        try:
            # SVD reduction for layer l
            X_l_c = X_l_all - X_l_all.mean(axis=0)
            U_l, s_l, _ = randomized_svd(X_l_c, n_components=n_pca, n_iter=3, random_state=42)
            X_l_reduced = U_l * s_l  # (n*seq, n_pca)

            # SVD reduction for layer l+1
            X_l1_c = X_l1_all - X_l1_all.mean(axis=0)
            U_l1, s_l1, _ = randomized_svd(X_l1_c, n_components=n_pca, n_iter=3, random_state=42)
            X_l1_reduced = U_l1 * s_l1

            # CCA on ALL data (unsupervised)
            cca = CCA(n_components=n_cca, max_iter=500)
            X_l_cca, X_l1_cca = cca.fit_transform(X_l_reduced, X_l1_reduced)

            # Compute canonical correlations
            correlations = []
            for i in range(X_l_cca.shape[1]):
                corr = np.corrcoef(X_l_cca[:, i], X_l1_cca[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

            global_mean_corr = np.mean(correlations) if correlations else 0.0

            # Compute per-token "alignment score" using the fitted CCA
            # This is the sum of correlations for each token's projections
            per_token_scores = np.zeros(len(X_l_cca))
            for i in range(X_l_cca.shape[1]):
                # Product of CCA projections (high when both are aligned in that dimension)
                per_token_scores += X_l_cca[:, i] * X_l1_cca[:, i]
            per_token_scores /= X_l_cca.shape[1]  # Normalize

        except Exception as e:
            print(f"        SVCCA failed: {e}")
            continue

        # Split by correctness
        correct_scores = per_token_scores[correct_mask]
        incorrect_scores = per_token_scores[~correct_mask]

        d = cohens_d(correct_scores, incorrect_scores)
        _, p = stats.ttest_ind(correct_scores, incorrect_scores)

        results.append({
            'layer': l,
            'global_mean_corr': float(global_mean_corr),
            'correct_mean_score': float(np.mean(correct_scores)),
            'incorrect_mean_score': float(np.mean(incorrect_scores)),
            'cohens_d': float(d),
            'p_value': float(p)
        })

    return pd.DataFrame(results)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(data_dir, models, output_dir, max_samples=100):
    """Run Procrustes + SVCCA analysis for all model/task combinations."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ['gsm8k', 'humaneval', 'logiqa']
    procrustes_results = []
    svcca_results = []
    procrustes_layer_results = []
    svcca_layer_results = []

    for model in models:
        for task in tasks:
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

            # === PROCRUSTES ANALYSIS ===
            print("    Computing Procrustes analysis...")
            df_proc_layer = procrustes_per_layer(trajectories, labels)
            df_proc_layer['model'] = model
            df_proc_layer['task'] = task
            procrustes_layer_results.append(df_proc_layer)

            # Aggregate across layers
            if len(df_proc_layer) > 0:
                mean_d = df_proc_layer['cohens_d'].mean()
                min_p = df_proc_layer['p_value'].min()
                best_layer = df_proc_layer.loc[df_proc_layer['cohens_d'].abs().idxmax()]

                procrustes_results.append({
                    'model': model,
                    'task': task,
                    'mean_cohens_d': float(mean_d),
                    'best_layer': int(best_layer['layer']),
                    'best_layer_d': float(best_layer['cohens_d']),
                    'best_layer_p': float(best_layer['p_value']),
                    'n_correct': int(n_correct),
                    'n_incorrect': int(n_incorrect)
                })

                print(f"    Procrustes: mean_d={mean_d:.3f}, best_layer={int(best_layer['layer'])} (d={best_layer['cohens_d']:.3f}, p={best_layer['p_value']:.3f})")

            # === SVCCA ANALYSIS ===
            print("    Computing SVCCA analysis...")
            df_svcca_layer = svcca_per_layer(trajectories, labels)
            df_svcca_layer['model'] = model
            df_svcca_layer['task'] = task
            svcca_layer_results.append(df_svcca_layer)

            # Aggregate across layers
            if len(df_svcca_layer) > 0:
                mean_d = df_svcca_layer['cohens_d'].mean()
                min_p = df_svcca_layer['p_value'].min()
                best_layer = df_svcca_layer.loc[df_svcca_layer['cohens_d'].abs().idxmax()]

                svcca_results.append({
                    'model': model,
                    'task': task,
                    'mean_cohens_d': float(mean_d),
                    'best_layer': int(best_layer['layer']),
                    'best_layer_d': float(best_layer['cohens_d']),
                    'best_layer_p': float(best_layer['p_value']),
                    'n_correct': int(n_correct),
                    'n_incorrect': int(n_incorrect)
                })

                print(f"    SVCCA: mean_d={mean_d:.3f}, best_layer={int(best_layer['layer'])} (d={best_layer['cohens_d']:.3f}, p={best_layer['p_value']:.3f})")

    # Save results
    if procrustes_results:
        df_proc = pd.DataFrame(procrustes_results)
        output_path = output_dir / 'h2_procrustes_alignment.csv'
        df_proc.to_csv(output_path, index=False)
        print(f"\n  Saved Procrustes results to {output_path}")

        print("\n  === Procrustes Results ===")
        print(df_proc.to_string(index=False))

    if procrustes_layer_results:
        df_proc_layers = pd.concat(procrustes_layer_results, ignore_index=True)
        layer_path = output_dir / 'h2_procrustes_layers.csv'
        df_proc_layers.to_csv(layer_path, index=False)
        print(f"\n  Saved Procrustes layer results to {layer_path}")

    if svcca_results:
        df_svcca = pd.DataFrame(svcca_results)
        output_path = output_dir / 'h2_svcca.csv'
        df_svcca.to_csv(output_path, index=False)
        print(f"\n  Saved SVCCA results to {output_path}")

        print("\n  === SVCCA Results ===")
        print(df_svcca.to_string(index=False))

    if svcca_layer_results:
        df_svcca_layers = pd.concat(svcca_layer_results, ignore_index=True)
        layer_path = output_dir / 'h2_svcca_layers.csv'
        df_svcca_layers.to_csv(layer_path, index=False)
        print(f"\n  Saved SVCCA layer results to {layer_path}")


def main():
    parser = argparse.ArgumentParser(description='Alignment-Aware Dynamics Analysis')
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
    print("Alignment-Aware Dynamics Analysis (Fast Version)")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Models: {models}")
    print(f"{'='*60}")
    print()
    print("Methods:")
    print("  1. Procrustes: Optimal rotation alignment, measure reconstruction error")
    print("  2. SVCCA: Canonical correlation between layer subspaces")
    print()
    print("Both methods are UNSUPERVISED (no labels used in alignment)")
    print("Using PCA reduction to 256 dims for computational efficiency")
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
