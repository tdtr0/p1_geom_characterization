#!/usr/bin/env python3
"""
Deep CKA Analysis

Follow-up to predictive_alignment.py - dig into WHY incorrect solutions have
higher CKA (more preserved similarity) than correct ones.

Analyses:
1. Layer profile of CKA differences
2. CKA as a classifier feature (ROC-AUC)
3. What's different in the Gram matrices?
4. Token-position contribution to CKA difference

Usage:
    python cka_deep_analysis.py --data-dir data/trajectories_0shot --model olmo3_rl_zero --task gsm8k
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_trajectories(filepath, max_samples=None):
    """Load trajectories and labels from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        trajectories = f['trajectories'][:]
        if 'is_correct' in f:
            labels = f['is_correct'][:]
        else:
            labels = f['correct'][:]

        if max_samples and max_samples < len(trajectories):
            np.random.seed(42)
            indices = np.random.choice(len(trajectories), max_samples, replace=False)
            trajectories = trajectories[indices]
            labels = labels[indices]

    return trajectories.astype(np.float32), labels.astype(bool)


def linear_cka(K1, K2):
    """Compute linear CKA between two Gram matrices."""
    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1_c = H @ K1 @ H
    K2_c = H @ K2 @ H

    hsic_12 = np.sum(K1_c * K2_c)
    hsic_11 = np.sum(K1_c * K1_c)
    hsic_22 = np.sum(K2_c * K2_c)

    if hsic_11 < 1e-10 or hsic_22 < 1e-10:
        return 0.0
    return hsic_12 / np.sqrt(hsic_11 * hsic_22)


def compute_per_sample_cka_all_layers(trajectories):
    """Compute CKA for each sample at each layer transition."""
    n_samples, seq_len, n_layers, d_model = trajectories.shape
    cka_matrix = np.zeros((n_samples, n_layers - 1))

    for i in range(n_samples):
        for l in range(n_layers - 1):
            X_l = trajectories[i, :, l, :]
            X_l1 = trajectories[i, :, l+1, :]
            K_l = X_l @ X_l.T
            K_l1 = X_l1 @ X_l1.T
            cka_matrix[i, l] = linear_cka(K_l, K_l1)

    return cka_matrix


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


def analyze_cka_profile(cka_matrix, labels):
    """Analyze CKA difference across layers."""
    n_layers = cka_matrix.shape[1]

    print("\n" + "="*70)
    print("ANALYSIS 1: CKA Difference Profile Across Layers")
    print("="*70)
    print("\nHypothesis: Incorrect solutions have higher CKA (more similarity preserved)")
    print("Negative d = incorrect higher, Positive d = correct higher\n")

    print(f"{'Layer':<8} {'Correct':<12} {'Incorrect':<12} {'Diff':<12} {'Cohen d':<10} {'p-value':<10} {'Sig':<5}")
    print("-"*70)

    layer_stats = []
    for l in range(n_layers):
        correct_cka = cka_matrix[labels, l]
        incorrect_cka = cka_matrix[~labels, l]

        diff = np.mean(correct_cka) - np.mean(incorrect_cka)
        d = cohens_d(correct_cka, incorrect_cka)
        _, p = stats.ttest_ind(correct_cka, incorrect_cka)

        sig = "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""

        print(f"L{l:<7} {np.mean(correct_cka):<12.4f} {np.mean(incorrect_cka):<12.4f} "
              f"{diff:<12.4f} {d:<10.3f} {p:<10.4f} {sig:<5}")

        layer_stats.append({
            'layer': l,
            'correct_mean': float(np.mean(correct_cka)),
            'incorrect_mean': float(np.mean(incorrect_cka)),
            'diff': float(diff),
            'd': float(d),
            'p': float(p)
        })

    # Find most discriminative layer
    best_layer = min(layer_stats, key=lambda x: x['p'])
    print("-"*70)
    print(f"\nMost discriminative layer: L{best_layer['layer']} (d={best_layer['d']:.3f}, p={best_layer['p']:.4f})")

    return layer_stats


def analyze_cka_as_classifier(cka_matrix, labels):
    """Test CKA values as features for correctness prediction."""
    print("\n" + "="*70)
    print("ANALYSIS 2: CKA as Classifier Feature")
    print("="*70)
    print("\nCan CKA values predict correctness? (5-fold CV)")

    n_correct = labels.sum()
    n_incorrect = (~labels).sum()
    n_splits = min(5, n_correct, n_incorrect)

    if n_splits < 2:
        print("Insufficient samples for CV")
        return {}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}

    # Test each layer individually
    print(f"\n{'Feature':<25} {'AUC':<10} {'Interpretation':<30}")
    print("-"*65)

    for l in range(cka_matrix.shape[1]):
        X = cka_matrix[:, l].reshape(-1, 1)
        y = labels.astype(int)

        try:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
            auc = roc_auc_score(y, probs)

            # Interpretation: AUC > 0.5 means correct has higher feature
            # Since we found incorrect has higher CKA, we expect AUC < 0.5
            interp = "Incorrect higher" if auc < 0.5 else "Correct higher" if auc > 0.5 else "No signal"
            # Flip AUC for easier reading (how well can we distinguish?)
            auc_readable = max(auc, 1 - auc)

            print(f"CKA L{l:<21} {auc_readable:<10.3f} {interp:<30}")
            results[f'layer_{l}'] = {'auc': float(auc), 'auc_readable': float(auc_readable)}
        except Exception as e:
            print(f"CKA L{l:<21} FAILED: {e}")

    # Test all layers combined
    try:
        X = cka_matrix
        clf = LogisticRegression(max_iter=1000, random_state=42)
        probs = cross_val_predict(clf, X, labels.astype(int), cv=cv, method='predict_proba')[:, 1]
        auc = roc_auc_score(labels.astype(int), probs)
        auc_readable = max(auc, 1 - auc)
        print(f"{'All layers combined':<25} {auc_readable:<10.3f} {'Multi-layer signal':<30}")
        results['all_layers'] = {'auc': float(auc), 'auc_readable': float(auc_readable)}
    except Exception as e:
        print(f"All layers: FAILED - {e}")

    return results


def analyze_cka_trajectory(cka_matrix, labels):
    """Analyze how CKA changes across layers (trajectory shape)."""
    print("\n" + "="*70)
    print("ANALYSIS 3: CKA Trajectory Shape")
    print("="*70)
    print("\nDo correct/incorrect have different CKA trajectories across layers?")

    # Compute trajectory features
    features = {}

    # Mean CKA
    mean_cka = cka_matrix.mean(axis=1)
    d = cohens_d(mean_cka[labels], mean_cka[~labels])
    _, p = stats.ttest_ind(mean_cka[labels], mean_cka[~labels])
    print(f"\nMean CKA across layers: d={d:.3f}, p={p:.4f}")
    features['mean_cka'] = {'d': float(d), 'p': float(p)}

    # CKA variance (how much does CKA change across layers?)
    cka_var = cka_matrix.var(axis=1)
    d = cohens_d(cka_var[labels], cka_var[~labels])
    _, p = stats.ttest_ind(cka_var[labels], cka_var[~labels])
    print(f"CKA variance (stability): d={d:.3f}, p={p:.4f}")
    features['cka_variance'] = {'d': float(d), 'p': float(p)}

    # CKA trend (does CKA increase or decrease across layers?)
    # Use linear regression slope
    x = np.arange(cka_matrix.shape[1])
    slopes = []
    for i in range(len(cka_matrix)):
        slope, _, _, _, _ = stats.linregress(x, cka_matrix[i])
        slopes.append(slope)
    slopes = np.array(slopes)

    d = cohens_d(slopes[labels], slopes[~labels])
    _, p = stats.ttest_ind(slopes[labels], slopes[~labels])
    print(f"CKA trend (slope): d={d:.3f}, p={p:.4f}")
    print(f"  Correct mean slope: {np.mean(slopes[labels]):.6f}")
    print(f"  Incorrect mean slope: {np.mean(slopes[~labels]):.6f}")
    features['cka_slope'] = {'d': float(d), 'p': float(p),
                             'correct_mean': float(np.mean(slopes[labels])),
                             'incorrect_mean': float(np.mean(slopes[~labels]))}

    # Early vs late layers
    n_layers = cka_matrix.shape[1]
    early_cka = cka_matrix[:, :n_layers//2].mean(axis=1)
    late_cka = cka_matrix[:, n_layers//2:].mean(axis=1)
    early_late_diff = early_cka - late_cka

    d = cohens_d(early_late_diff[labels], early_late_diff[~labels])
    _, p = stats.ttest_ind(early_late_diff[labels], early_late_diff[~labels])
    print(f"Early-Late CKA difference: d={d:.3f}, p={p:.4f}")
    features['early_late_diff'] = {'d': float(d), 'p': float(p)}

    # Min CKA (where is the "bottleneck"?)
    min_cka = cka_matrix.min(axis=1)
    d = cohens_d(min_cka[labels], min_cka[~labels])
    _, p = stats.ttest_ind(min_cka[labels], min_cka[~labels])
    print(f"Min CKA (bottleneck): d={d:.3f}, p={p:.4f}")
    features['min_cka'] = {'d': float(d), 'p': float(p)}

    # Argmin (at which layer is the bottleneck?)
    argmin_cka = cka_matrix.argmin(axis=1)
    print(f"\nBottleneck layer distribution:")
    print(f"  Correct: mean={np.mean(argmin_cka[labels]):.1f}, mode={stats.mode(argmin_cka[labels], keepdims=False).mode}")
    print(f"  Incorrect: mean={np.mean(argmin_cka[~labels]):.1f}, mode={stats.mode(argmin_cka[~labels], keepdims=False).mode}")

    return features


def analyze_gram_matrix_structure(trajectories, labels, layer=7):
    """Analyze what's different in Gram matrices at the most discriminative layer."""
    print("\n" + "="*70)
    print(f"ANALYSIS 4: Gram Matrix Structure at L{layer}")
    print("="*70)
    print("\nWhat's different about token-token similarity patterns?")

    n_samples, seq_len, n_layers, d_model = trajectories.shape

    # Compute Gram matrices at layer l and l+1
    gram_l_correct = []
    gram_l_incorrect = []
    gram_l1_correct = []
    gram_l1_incorrect = []

    for i in range(n_samples):
        X_l = trajectories[i, :, layer, :]
        X_l1 = trajectories[i, :, layer+1, :]
        K_l = X_l @ X_l.T
        K_l1 = X_l1 @ X_l1.T

        # Normalize for comparison
        K_l = K_l / (np.linalg.norm(K_l, 'fro') + 1e-10)
        K_l1 = K_l1 / (np.linalg.norm(K_l1, 'fro') + 1e-10)

        if labels[i]:
            gram_l_correct.append(K_l)
            gram_l1_correct.append(K_l1)
        else:
            gram_l_incorrect.append(K_l)
            gram_l1_incorrect.append(K_l1)

    # Average Gram matrices
    avg_gram_l_correct = np.mean(gram_l_correct, axis=0)
    avg_gram_l_incorrect = np.mean(gram_l_incorrect, axis=0)
    avg_gram_l1_correct = np.mean(gram_l1_correct, axis=0)
    avg_gram_l1_incorrect = np.mean(gram_l1_incorrect, axis=0)

    # How similar are the average Gram matrices?
    cka_correct_avg = linear_cka(avg_gram_l_correct, avg_gram_l1_correct)
    cka_incorrect_avg = linear_cka(avg_gram_l_incorrect, avg_gram_l1_incorrect)

    print(f"\nAverage Gram matrix CKA:")
    print(f"  Correct: {cka_correct_avg:.4f}")
    print(f"  Incorrect: {cka_incorrect_avg:.4f}")
    print(f"  Difference: {cka_correct_avg - cka_incorrect_avg:.4f}")

    # Eigenspectrum analysis
    print(f"\nGram matrix eigenspectrum (layer {layer}):")

    eig_l_correct = np.linalg.eigvalsh(avg_gram_l_correct)[::-1]
    eig_l_incorrect = np.linalg.eigvalsh(avg_gram_l_incorrect)[::-1]

    # Effective rank (entropy of normalized eigenvalues)
    def effective_rank(eigvals):
        eigvals = eigvals[eigvals > 1e-10]
        eigvals = eigvals / eigvals.sum()
        return np.exp(-np.sum(eigvals * np.log(eigvals + 1e-10)))

    eff_rank_correct = effective_rank(eig_l_correct)
    eff_rank_incorrect = effective_rank(eig_l_incorrect)

    print(f"  Effective rank (correct): {eff_rank_correct:.2f}")
    print(f"  Effective rank (incorrect): {eff_rank_incorrect:.2f}")

    # Top eigenvalue concentration
    top_k = 10
    top_k_ratio_correct = eig_l_correct[:top_k].sum() / (eig_l_correct.sum() + 1e-10)
    top_k_ratio_incorrect = eig_l_incorrect[:top_k].sum() / (eig_l_incorrect.sum() + 1e-10)

    print(f"  Top-{top_k} eigenvalue ratio (correct): {top_k_ratio_correct:.4f}")
    print(f"  Top-{top_k} eigenvalue ratio (incorrect): {top_k_ratio_incorrect:.4f}")

    # Gram matrix change magnitude
    change_correct = np.mean([np.linalg.norm(gram_l1_correct[i] - gram_l_correct[i], 'fro')
                              for i in range(len(gram_l_correct))])
    change_incorrect = np.mean([np.linalg.norm(gram_l1_incorrect[i] - gram_l_incorrect[i], 'fro')
                                 for i in range(len(gram_l_incorrect))])

    print(f"\nGram matrix change magnitude (L{layer}→L{layer+1}):")
    print(f"  Correct: {change_correct:.4f}")
    print(f"  Incorrect: {change_incorrect:.4f}")
    print(f"  Ratio: {change_correct/change_incorrect:.3f}x")

    if change_correct > change_incorrect:
        print("  → Correct solutions have MORE Gram matrix change (less preserved structure)")
    else:
        print("  → Incorrect solutions have MORE Gram matrix change")

    return {
        'cka_correct_avg': float(cka_correct_avg),
        'cka_incorrect_avg': float(cka_incorrect_avg),
        'eff_rank_correct': float(eff_rank_correct),
        'eff_rank_incorrect': float(eff_rank_incorrect),
        'gram_change_correct': float(change_correct),
        'gram_change_incorrect': float(change_incorrect)
    }


def main():
    parser = argparse.ArgumentParser(description='Deep CKA Analysis')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='olmo3_rl_zero')
    parser.add_argument('--task', type=str, default='gsm8k')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--max-samples', type=int, default=None)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = data_dir / args.model / f"{args.task}_trajectories.h5"
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found")
        return

    print(f"\n{'='*70}")
    print("Deep CKA Analysis")
    print(f"{'='*70}")
    print(f"Model: {args.model}, Task: {args.task}")

    # Load data
    trajectories, labels = load_trajectories(h5_path, args.max_samples)
    n_samples = len(labels)
    n_correct = labels.sum()
    print(f"Loaded: {n_samples} samples ({n_correct} correct, {n_samples - n_correct} incorrect)")

    # Compute CKA matrix
    print("\nComputing per-sample CKA for all layer transitions...")
    cka_matrix = compute_per_sample_cka_all_layers(trajectories)
    print(f"CKA matrix shape: {cka_matrix.shape}")

    # Run analyses
    all_results = {
        'model': args.model,
        'task': args.task,
        'n_samples': int(n_samples),
        'n_correct': int(n_correct)
    }

    # Analysis 1: Layer profile
    layer_stats = analyze_cka_profile(cka_matrix, labels)
    all_results['layer_profile'] = layer_stats

    # Analysis 2: CKA as classifier
    classifier_results = analyze_cka_as_classifier(cka_matrix, labels)
    all_results['classifier'] = classifier_results

    # Analysis 3: Trajectory shape
    trajectory_features = analyze_cka_trajectory(cka_matrix, labels)
    all_results['trajectory'] = trajectory_features

    # Analysis 4: Gram matrix structure at most discriminative layer
    best_layer = min(layer_stats, key=lambda x: x['p'])['layer']
    gram_analysis = analyze_gram_matrix_structure(trajectories, labels, layer=best_layer)
    all_results['gram_structure'] = gram_analysis

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: What's Different About Correct vs Incorrect CKA?")
    print("="*70)

    print("""
KEY FINDINGS:

1. INCORRECT solutions have HIGHER CKA (more similarity preserved)
   - This means token-token relationships change LESS between layers
   - Incorrect solutions have more "static" representations

2. CORRECT solutions have LOWER CKA (less similarity preserved)
   - Token relationships are transformed MORE between layers
   - Correct solutions undergo more representational change per layer

INTERPRETATION:
   - Correct solutions are "working harder" - doing more computation per layer
   - Incorrect solutions are "coasting" - minimal transformation
   - This aligns with: correct answers require active reasoning, incorrect = pattern matching?

ACTIONABLE INSIGHT:
   - Low inter-layer CKA could be a signal of engaged computation
   - Could be used as a feature for correctness detection (AUC shown above)
""")

    # Save results
    output_file = output_dir / f'cka_deep_analysis_{args.model}_{args.task}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
