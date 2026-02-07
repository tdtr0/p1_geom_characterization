#!/usr/bin/env python3
"""
Predictive Alignment Analysis

Tests three approaches to recover dynamical signal despite the orthogonality problem
(token-level cos(X_l, X_{l+1}) ≈ 0.10).

Experiments:
1. Per-Sample Prediction Error: Do correct solutions follow global pattern better?
2. Conditional Procrustes: Do correct/incorrect use different rotations?
3. Per-Sample CKA: Is representational similarity higher for correct?

Usage:
    python predictive_alignment.py --data-dir data/trajectories_0shot --model olmo3_rl_zero --task gsm8k
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


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
            np.random.seed(42)
            indices = np.random.choice(len(trajectories), max_samples, replace=False)
            trajectories = trajectories[indices]
            labels = labels[indices]

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
# EXPERIMENT 1: Per-Sample Prediction Error
# =============================================================================

def per_sample_prediction_error(trajectories, labels, n_pca=256):
    """
    Measure how well X_l predicts X_{l+1} for each sample.

    Hypothesis: Correct solutions are more "on-manifold" (lower prediction error).

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean
        n_pca: PCA reduction for tractable least squares

    Returns:
        List of dicts with per-layer results
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape
    n_pca = min(n_pca, n_samples * seq_len - 1, d_model)

    results = []
    for l in range(n_layers - 1):
        # Pool all tokens for fitting predictor
        X_l = trajectories[:, :, l, :].reshape(-1, d_model)
        X_l1 = trajectories[:, :, l+1, :].reshape(-1, d_model)

        # PCA reduction for tractable least squares
        try:
            X_combined = np.vstack([X_l, X_l1])
            X_combined_c = X_combined - X_combined.mean(axis=0)
            U, s, Vt = randomized_svd(X_combined_c, n_components=n_pca, n_iter=3, random_state=42)
            pca_components = Vt

            X_l_pca = (X_l - X_combined.mean(axis=0)) @ pca_components.T
            X_l1_pca = (X_l1 - X_combined.mean(axis=0)) @ pca_components.T
        except Exception as e:
            print(f"  Layer {l}: PCA failed - {e}")
            continue

        # Fit linear predictor on all data
        try:
            W, residuals, rank, s = np.linalg.lstsq(X_l_pca, X_l1_pca, rcond=None)
        except Exception as e:
            print(f"  Layer {l}: Least squares failed - {e}")
            continue

        # Per-sample error (reshape back)
        errors = []
        for i in range(n_samples):
            sample_l = trajectories[i, :, l, :]
            sample_l1 = trajectories[i, :, l+1, :]

            # Project to PCA space
            sample_l_pca = (sample_l - X_combined.mean(axis=0)) @ pca_components.T
            sample_l1_pca = (sample_l1 - X_combined.mean(axis=0)) @ pca_components.T

            # Prediction and error
            pred = sample_l_pca @ W
            err = np.linalg.norm(sample_l1_pca - pred) / (np.linalg.norm(sample_l1_pca) + 1e-10)
            errors.append(err)

        # Compare by correctness
        correct_err = [errors[i] for i in range(n_samples) if labels[i]]
        incorrect_err = [errors[i] for i in range(n_samples) if not labels[i]]

        d = cohens_d(correct_err, incorrect_err)
        _, p = stats.ttest_ind(correct_err, incorrect_err)

        results.append({
            'layer': l,
            'correct_mean_error': float(np.mean(correct_err)),
            'incorrect_mean_error': float(np.mean(incorrect_err)),
            'cohens_d': float(d),
            'p_value': float(p),
            'n_correct': len(correct_err),
            'n_incorrect': len(incorrect_err)
        })

    return results


# =============================================================================
# EXPERIMENT 2: Conditional Procrustes
# =============================================================================

def conditional_procrustes(trajectories, labels, n_pca=256):
    """
    Fit separate rotations for correct/incorrect, measure difference.

    Hypothesis: Correct and incorrect use different rotational paths (R_correct ≠ R_incorrect).

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean
        n_pca: PCA reduction

    Returns:
        List of dicts with per-layer results
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape
    n_pca = min(n_pca, n_samples * seq_len - 1, d_model)

    results = []
    for l in range(n_layers - 1):
        X_l = trajectories[:, :, l, :].reshape(-1, d_model)
        X_l1 = trajectories[:, :, l+1, :].reshape(-1, d_model)
        mask = np.repeat(labels, seq_len)

        # PCA reduction
        try:
            X_combined = np.vstack([X_l, X_l1])
            mean_vec = X_combined.mean(axis=0)
            X_combined_c = X_combined - mean_vec
            U, s, Vt = randomized_svd(X_combined_c, n_components=n_pca, n_iter=3, random_state=42)
            pca_components = Vt

            X_l_pca = (X_l - mean_vec) @ pca_components.T
            X_l1_pca = (X_l1 - mean_vec) @ pca_components.T
        except Exception as e:
            print(f"  Layer {l}: PCA failed - {e}")
            continue

        # Split by correctness
        X_l_correct = X_l_pca[mask]
        X_l1_correct = X_l1_pca[mask]
        X_l_incorrect = X_l_pca[~mask]
        X_l1_incorrect = X_l1_pca[~mask]

        if len(X_l_correct) < 10 or len(X_l_incorrect) < 10:
            print(f"  Layer {l}: Insufficient samples")
            continue

        try:
            # Fit on correct only
            X_l_c = X_l_correct - X_l_correct.mean(axis=0)
            X_l1_c = X_l1_correct - X_l1_correct.mean(axis=0)
            R_correct, scale_correct = orthogonal_procrustes(X_l_c, X_l1_c)

            # Fit on incorrect only
            X_l_i = X_l_incorrect - X_l_incorrect.mean(axis=0)
            X_l1_i = X_l1_incorrect - X_l1_incorrect.mean(axis=0)
            R_incorrect, scale_incorrect = orthogonal_procrustes(X_l_i, X_l1_i)
        except Exception as e:
            print(f"  Layer {l}: Procrustes failed - {e}")
            continue

        # Measure rotation difference (Frobenius norm)
        rotation_diff = np.linalg.norm(R_correct - R_incorrect, 'fro')

        # Normalize by expected norm of difference between two random orthogonal matrices
        # For n×n orthogonal matrices, expected ||R1 - R2||_F ≈ sqrt(2n)
        rotation_diff_normalized = rotation_diff / np.sqrt(2 * n_pca)

        # Cross-application error: use "wrong" rotation on correct samples
        err_correct_with_right_R = np.linalg.norm(X_l1_c - X_l_c @ R_correct * scale_correct, 'fro')
        err_correct_with_wrong_R = np.linalg.norm(X_l1_c - X_l_c @ R_incorrect * scale_incorrect, 'fro')
        extra_error_correct = (err_correct_with_wrong_R - err_correct_with_right_R) / (err_correct_with_right_R + 1e-10)

        # Cross-application error: use "wrong" rotation on incorrect samples
        err_incorrect_with_right_R = np.linalg.norm(X_l1_i - X_l_i @ R_incorrect * scale_incorrect, 'fro')
        err_incorrect_with_wrong_R = np.linalg.norm(X_l1_i - X_l_i @ R_correct * scale_correct, 'fro')
        extra_error_incorrect = (err_incorrect_with_wrong_R - err_incorrect_with_right_R) / (err_incorrect_with_right_R + 1e-10)

        results.append({
            'layer': l,
            'rotation_diff_fro': float(rotation_diff),
            'rotation_diff_normalized': float(rotation_diff_normalized),
            'scale_correct': float(scale_correct),
            'scale_incorrect': float(scale_incorrect),
            'extra_error_correct': float(extra_error_correct),
            'extra_error_incorrect': float(extra_error_incorrect),
            'mean_extra_error': float((extra_error_correct + extra_error_incorrect) / 2)
        })

    return results


# =============================================================================
# EXPERIMENT 3: Per-Sample CKA
# =============================================================================

def linear_cka(K1, K2):
    """
    Compute linear CKA between two Gram matrices.

    CKA = HSIC(K1, K2) / sqrt(HSIC(K1, K1) * HSIC(K2, K2))
    """
    # Center the Gram matrices
    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1_c = H @ K1 @ H
    K2_c = H @ K2 @ H

    # HSIC is just the Frobenius inner product of centered Gram matrices
    hsic_12 = np.sum(K1_c * K2_c)
    hsic_11 = np.sum(K1_c * K1_c)
    hsic_22 = np.sum(K2_c * K2_c)

    if hsic_11 < 1e-10 or hsic_22 < 1e-10:
        return 0.0

    return hsic_12 / np.sqrt(hsic_11 * hsic_22)


def per_sample_cka(trajectories, labels):
    """
    Compute CKA for each sample's trajectory across layers.

    Hypothesis: Correct solutions maintain higher CKA (representational similarity).

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean

    Returns:
        List of dicts with per-layer results
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    results = []
    for l in range(n_layers - 1):
        cka_values = []

        for i in range(n_samples):
            X_l = trajectories[i, :, l, :]      # (seq_len, d_model)
            X_l1 = trajectories[i, :, l+1, :]   # (seq_len, d_model)

            # Compute Gram matrices (token similarity)
            K_l = X_l @ X_l.T      # (seq_len, seq_len)
            K_l1 = X_l1 @ X_l1.T

            cka = linear_cka(K_l, K_l1)
            cka_values.append(cka)

        # Split by correctness
        correct_cka = [cka_values[i] for i in range(n_samples) if labels[i]]
        incorrect_cka = [cka_values[i] for i in range(n_samples) if not labels[i]]

        d = cohens_d(correct_cka, incorrect_cka)
        _, p = stats.ttest_ind(correct_cka, incorrect_cka)

        results.append({
            'layer': l,
            'correct_mean_cka': float(np.mean(correct_cka)),
            'incorrect_mean_cka': float(np.mean(incorrect_cka)),
            'cohens_d': float(d),
            'p_value': float(p),
            'n_correct': len(correct_cka),
            'n_incorrect': len(incorrect_cka)
        })

    return results


# =============================================================================
# MAIN
# =============================================================================

def run_analysis(data_dir, model, task, output_dir, max_samples=None):
    """Run all three experiments."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = data_dir / model / f"{task}_trajectories.h5"
    if not h5_path.exists():
        print(f"ERROR: File not found: {h5_path}")
        return

    print(f"\n{'='*70}")
    print(f"Predictive Alignment Analysis")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Task: {task}")
    print(f"Data: {h5_path}")
    print(f"{'='*70}\n")

    # Load data
    trajectories, labels = load_trajectories(h5_path, max_samples)
    n_samples, seq_len, n_layers, d_model = trajectories.shape
    n_correct = labels.sum()
    n_incorrect = (~labels).sum()

    print(f"Loaded: {n_samples} samples, {seq_len} tokens, {n_layers} layers, d={d_model}")
    print(f"Correct: {n_correct}, Incorrect: {n_incorrect}")
    print()

    all_results = {
        'model': model,
        'task': task,
        'n_samples': int(n_samples),
        'n_correct': int(n_correct),
        'n_incorrect': int(n_incorrect),
        'timestamp': datetime.now().isoformat()
    }

    # =========================
    # Experiment 1: Per-Sample Prediction Error
    # =========================
    print("="*70)
    print("EXPERIMENT 1: Per-Sample Prediction Error")
    print("="*70)
    print("Hypothesis: Correct solutions have lower prediction error (more on-manifold)")
    print()

    pred_results = per_sample_prediction_error(trajectories, labels)
    all_results['prediction_error'] = pred_results

    print(f"{'Layer':<8} {'Correct Err':<14} {'Incorrect Err':<14} {'Cohen d':<10} {'p-value':<10}")
    print("-"*60)
    for r in pred_results:
        print(f"L{r['layer']:<7} {r['correct_mean_error']:<14.4f} {r['incorrect_mean_error']:<14.4f} "
              f"{r['cohens_d']:<10.3f} {r['p_value']:<10.4f}")

    mean_d = np.mean([r['cohens_d'] for r in pred_results])
    sig_layers = sum(1 for r in pred_results if r['p_value'] < 0.05)
    print("-"*60)
    print(f"Mean Cohen's d: {mean_d:.3f}")
    print(f"Significant layers (p<0.05): {sig_layers}/{len(pred_results)}")
    if mean_d < -0.2:
        print(">>> SIGNAL: Correct solutions have LOWER prediction error (more predictable)")
    elif mean_d > 0.2:
        print(">>> SIGNAL: Incorrect solutions have LOWER prediction error (unexpected)")
    else:
        print(">>> NO SIGNAL: Prediction error similar for correct/incorrect")
    print()

    # =========================
    # Experiment 2: Conditional Procrustes
    # =========================
    print("="*70)
    print("EXPERIMENT 2: Conditional Procrustes")
    print("="*70)
    print("Hypothesis: Correct and incorrect use different rotations (R_correct != R_incorrect)")
    print()

    proc_results = conditional_procrustes(trajectories, labels)
    all_results['conditional_procrustes'] = proc_results

    print(f"{'Layer':<8} {'Rot Diff':<12} {'Norm Diff':<12} {'Extra Err C':<12} {'Extra Err I':<12}")
    print("-"*60)
    for r in proc_results:
        print(f"L{r['layer']:<7} {r['rotation_diff_fro']:<12.4f} {r['rotation_diff_normalized']:<12.4f} "
              f"{r['extra_error_correct']:<12.4f} {r['extra_error_incorrect']:<12.4f}")

    mean_rot_diff = np.mean([r['rotation_diff_normalized'] for r in proc_results])
    mean_extra_err = np.mean([r['mean_extra_error'] for r in proc_results])
    print("-"*60)
    print(f"Mean normalized rotation difference: {mean_rot_diff:.4f}")
    print(f"Mean extra error from wrong rotation: {mean_extra_err:.4f}")
    if mean_rot_diff > 0.1:
        print(">>> SIGNAL: Rotations differ significantly between correct/incorrect")
    elif mean_extra_err > 0.05:
        print(">>> SIGNAL: Using wrong rotation causes >5% extra error")
    else:
        print(">>> NO SIGNAL: Rotations are similar for correct/incorrect")
    print()

    # =========================
    # Experiment 3: Per-Sample CKA
    # =========================
    print("="*70)
    print("EXPERIMENT 3: Per-Sample CKA")
    print("="*70)
    print("Hypothesis: Correct solutions maintain higher CKA (information preserved)")
    print()

    cka_results = per_sample_cka(trajectories, labels)
    all_results['per_sample_cka'] = cka_results

    print(f"{'Layer':<8} {'Correct CKA':<14} {'Incorrect CKA':<14} {'Cohen d':<10} {'p-value':<10}")
    print("-"*60)
    for r in cka_results:
        print(f"L{r['layer']:<7} {r['correct_mean_cka']:<14.4f} {r['incorrect_mean_cka']:<14.4f} "
              f"{r['cohens_d']:<10.3f} {r['p_value']:<10.4f}")

    mean_d = np.mean([r['cohens_d'] for r in cka_results])
    sig_layers = sum(1 for r in cka_results if r['p_value'] < 0.05)
    print("-"*60)
    print(f"Mean Cohen's d: {mean_d:.3f}")
    print(f"Significant layers (p<0.05): {sig_layers}/{len(cka_results)}")
    if mean_d > 0.2:
        print(">>> SIGNAL: Correct solutions have HIGHER CKA (better information preservation)")
    elif mean_d < -0.2:
        print(">>> SIGNAL: Incorrect solutions have HIGHER CKA (unexpected)")
    else:
        print(">>> NO SIGNAL: CKA similar for correct/incorrect")
    print()

    # =========================
    # Summary
    # =========================
    print("="*70)
    print("SUMMARY")
    print("="*70)

    pred_mean_d = np.mean([r['cohens_d'] for r in pred_results])
    cka_mean_d = np.mean([r['cohens_d'] for r in cka_results])
    mean_extra_err = np.mean([r['mean_extra_error'] for r in proc_results])

    print(f"\n{'Method':<30} {'Signal Metric':<20} {'Value':<15} {'Interpretation':<20}")
    print("-"*85)
    print(f"{'Per-Sample Prediction':<30} {'Cohen d':<20} {pred_mean_d:<15.3f} "
          f"{'SIGNAL' if abs(pred_mean_d) > 0.2 else 'No signal':<20}")
    print(f"{'Conditional Procrustes':<30} {'Extra Error %':<20} {mean_extra_err*100:<15.2f} "
          f"{'SIGNAL' if mean_extra_err > 0.05 else 'No signal':<20}")
    print(f"{'Per-Sample CKA':<30} {'Cohen d':<20} {cka_mean_d:<15.3f} "
          f"{'SIGNAL' if abs(cka_mean_d) > 0.2 else 'No signal':<20}")

    any_signal = abs(pred_mean_d) > 0.2 or mean_extra_err > 0.05 or abs(cka_mean_d) > 0.2
    print()
    if any_signal:
        print(">>> AT LEAST ONE METHOD SHOWS SIGNAL")
        print("    Predictive alignment may recover dynamical information!")
    else:
        print(">>> NO METHODS SHOW SIGNAL")
        print("    The orthogonality problem affects all predictive measures too.")

    all_results['summary'] = {
        'prediction_error_mean_d': float(pred_mean_d),
        'conditional_procrustes_extra_err': float(mean_extra_err),
        'per_sample_cka_mean_d': float(cka_mean_d),
        'any_signal': any_signal
    }

    # Save results
    output_file = output_dir / f'predictive_alignment_{model}_{task}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Predictive Alignment Analysis')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing trajectory HDF5 files')
    parser.add_argument('--model', type=str, default='olmo3_rl_zero',
                        help='Model name (default: olmo3_rl_zero)')
    parser.add_argument('--task', type=str, default='gsm8k',
                        help='Task name (default: gsm8k)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to use (default: all)')

    args = parser.parse_args()

    run_analysis(
        data_dir=args.data_dir,
        model=args.model,
        task=args.task,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
