#!/usr/bin/env python3
"""
Phase 1a: SVD-based Geometric Analysis

Computes:
1. Effective rank per model/layer
2. Spectral decay (power-law exponent)
3. Subspace preservation (RLVR vs SFT vs base)
4. Statistical significance tests

No classifiers - pure geometric comparison.
"""

import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.linalg import svd, subspace_angles
from scipy.stats import entropy, mannwhitneyu, ttest_ind
from typing import Dict, List, Tuple
import argparse
import json

# =============================================================================
# Core SVD Functions
# =============================================================================

def compute_effective_rank(activations: np.ndarray) -> float:
    """
    Effective rank: exp(entropy of normalized singular values).
    Measures dimensionality of the representation.
    """
    activations = activations.astype(np.float32)
    _, s, _ = svd(activations, full_matrices=False)
    s_normalized = s / (s.sum() + 1e-10)
    s_normalized = s_normalized[s_normalized > 1e-10]
    return float(np.exp(entropy(s_normalized)))


def compute_spectral_decay(activations: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Fit power-law to singular value decay: s_i ∝ i^(-α)
    Returns alpha (decay rate) and singular values.
    """
    activations = activations.astype(np.float32)
    _, s, _ = svd(activations, full_matrices=False)

    log_i = np.log(np.arange(1, len(s) + 1))
    log_s = np.log(s + 1e-10)

    A = np.vstack([log_i, np.ones_like(log_i)]).T
    alpha_neg, _ = np.linalg.lstsq(A, log_s, rcond=None)[0]

    return float(-alpha_neg), s


def compute_subspace_preservation(
    base_acts: np.ndarray,
    other_acts: np.ndarray,
    k: int = 100
) -> Tuple[float, np.ndarray]:
    """
    Measure subspace overlap between base and fine-tuned model.
    Returns preservation score (0-1) and principal angles.
    """
    base_acts = base_acts.astype(np.float32)
    other_acts = other_acts.astype(np.float32)

    _, _, Vt_base = svd(base_acts, full_matrices=False)
    _, _, Vt_other = svd(other_acts, full_matrices=False)

    k = min(k, Vt_base.shape[0], Vt_other.shape[0])
    V_base_k = Vt_base[:k, :].T
    V_other_k = Vt_other[:k, :].T

    angles = subspace_angles(V_base_k, V_other_k)
    preservation = float(np.cos(angles).mean())

    return preservation, angles


# =============================================================================
# Data Loading
# =============================================================================

def load_trajectories(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load trajectories and correctness labels."""
    print(f"  Loading {filepath}...", flush=True)

    with h5py.File(filepath, 'r') as f:
        trajectories = f['trajectories'][:]

        if 'is_correct' in f:
            labels = f['is_correct'][:]
        else:
            labels = np.zeros(len(trajectories), dtype=bool)

    print(f"    Shape: {trajectories.shape}", flush=True)
    print(f"    Correct: {labels.sum()}/{len(labels)} ({100*labels.mean():.1f}%)", flush=True)

    return trajectories.astype(np.float32), labels.astype(bool)


def aggregate_to_layer(trajectories: np.ndarray, layer_idx: int) -> np.ndarray:
    """
    Aggregate trajectories to get (n_samples, d_model) for a specific layer.
    Takes mean over sequence dimension.
    """
    # trajectories: (n_samples, seq_len, n_layers, d_model)
    layer_acts = trajectories[:, :, layer_idx, :]  # (n_samples, seq_len, d_model)
    return layer_acts.mean(axis=1)  # (n_samples, d_model)


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_single_model(
    trajectories: np.ndarray,
    labels: np.ndarray,
    model_name: str
) -> Dict:
    """Compute geometric metrics for a single model."""
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    results = {
        'model': model_name,
        'n_samples': n_samples,
        'n_correct': int(labels.sum()),
        'n_incorrect': int((~labels).sum()),
        'layers': {}
    }

    for layer_idx in range(n_layers):
        layer_acts = aggregate_to_layer(trajectories, layer_idx)

        # Overall metrics
        eff_rank = compute_effective_rank(layer_acts)
        spectral_alpha, _ = compute_spectral_decay(layer_acts)

        layer_results = {
            'effective_rank': eff_rank,
            'spectral_decay_alpha': spectral_alpha,
        }

        # Compare correct vs incorrect (if enough samples)
        if labels.sum() >= 10 and (~labels).sum() >= 10:
            correct_acts = layer_acts[labels]
            incorrect_acts = layer_acts[~labels]

            eff_rank_correct = compute_effective_rank(correct_acts)
            eff_rank_incorrect = compute_effective_rank(incorrect_acts)

            # Statistical test: Mann-Whitney U for effective rank difference
            # Compute per-sample metrics for statistical test
            layer_results['eff_rank_correct'] = eff_rank_correct
            layer_results['eff_rank_incorrect'] = eff_rank_incorrect
            layer_results['eff_rank_diff'] = eff_rank_correct - eff_rank_incorrect

        results['layers'][layer_idx] = layer_results

    return results


def compare_models(
    data_dir: str,
    models: List[str],
    task: str = 'gsm8k'
) -> Dict:
    """
    Compare geometric properties across models.
    Key comparison: subspace preservation (RLVR vs SFT vs base).
    """
    print("\n" + "="*70)
    print("GEOMETRIC ANALYSIS: Model Comparison")
    print("="*70)

    # Load all data
    model_data = {}
    for model in models:
        filepath = Path(data_dir) / model / f"{task}_trajectories.h5"
        if not filepath.exists():
            print(f"⚠ {model}: File not found")
            continue

        trajectories, labels = load_trajectories(str(filepath))
        model_data[model] = {'trajectories': trajectories, 'labels': labels}

    if 'olmo3_base' not in model_data:
        print("⚠ Base model required for subspace preservation analysis")
        return {}

    results = {
        'task': task,
        'models': {},
        'subspace_preservation': {},
        'effective_rank_comparison': {}
    }

    # Analyze each model
    for model, data in model_data.items():
        print(f"\n{'─'*70}")
        print(f"Analyzing: {model}")
        print(f"{'─'*70}")

        model_results = analyze_single_model(
            data['trajectories'],
            data['labels'],
            model
        )
        results['models'][model] = model_results

    # Subspace preservation vs base
    print(f"\n{'─'*70}")
    print("Subspace Preservation vs Base Model")
    print(f"{'─'*70}")

    base_trajs = model_data['olmo3_base']['trajectories']
    n_layers = base_trajs.shape[2]

    for model in models:
        if model == 'olmo3_base' or model not in model_data:
            continue

        other_trajs = model_data[model]['trajectories']
        preservations = []

        for layer_idx in range(n_layers):
            base_acts = aggregate_to_layer(base_trajs, layer_idx)
            other_acts = aggregate_to_layer(other_trajs, layer_idx)

            preservation, _ = compute_subspace_preservation(base_acts, other_acts, k=50)
            preservations.append(preservation)

        mean_preservation = np.mean(preservations)
        results['subspace_preservation'][model] = {
            'per_layer': preservations,
            'mean': float(mean_preservation)
        }

        print(f"  {model}: {mean_preservation:.3f} (mean across layers)")

    # Effective rank comparison
    print(f"\n{'─'*70}")
    print("Effective Rank Comparison")
    print(f"{'─'*70}")

    for model, model_results in results['models'].items():
        mean_eff_rank = np.mean([
            layer['effective_rank']
            for layer in model_results['layers'].values()
        ])
        results['effective_rank_comparison'][model] = float(mean_eff_rank)
        print(f"  {model}: {mean_eff_rank:.1f} (mean effective rank)")

    return results


def statistical_significance(results: Dict) -> Dict:
    """
    Add statistical significance tests.
    Z-score for subspace preservation difference.
    """
    print(f"\n{'─'*70}")
    print("Statistical Significance")
    print(f"{'─'*70}")

    sig_results = {}

    # Compare RLVR vs SFT subspace preservation
    if 'olmo3_rl_zero' in results['subspace_preservation'] and \
       'olmo3_sft' in results['subspace_preservation']:

        rl_pres = results['subspace_preservation']['olmo3_rl_zero']['per_layer']
        sft_pres = results['subspace_preservation']['olmo3_sft']['per_layer']

        # Paired t-test (same layers)
        t_stat, p_value = ttest_ind(rl_pres, sft_pres)

        sig_results['rlvr_vs_sft'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'rl_mean': float(np.mean(rl_pres)),
            'sft_mean': float(np.mean(sft_pres))
        }

        print(f"\n  RLVR vs SFT (subspace preservation):")
        print(f"    RLVR mean: {np.mean(rl_pres):.3f}")
        print(f"    SFT mean:  {np.mean(sft_pres):.3f}")
        print(f"    t-stat:    {t_stat:.3f}")
        print(f"    p-value:   {p_value:.4f}")
        print(f"    Significant: {'YES' if p_value < 0.05 else 'NO'}")

    return sig_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SVD-based Geometric Analysis")
    parser.add_argument('--data-dir', type=str, default='data/trajectories')
    parser.add_argument('--task', type=str, default='gsm8k')
    parser.add_argument('--output', type=str, default='results/geometric_analysis.json')

    args = parser.parse_args()

    models = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']

    print("="*70)
    print("SVD-BASED GEOMETRIC ANALYSIS")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Models: {models}")

    # Run analysis
    results = compare_models(args.data_dir, models, args.task)

    if results:
        # Statistical significance
        sig_results = statistical_significance(results)
        results['statistical_significance'] = sig_results

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✓ Results saved to {output_path}")

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        print("\nSubspace Preservation (vs base):")
        for model, data in results.get('subspace_preservation', {}).items():
            print(f"  {model}: {data['mean']:.3f}")

        print("\nEffective Rank:")
        for model, rank in results.get('effective_rank_comparison', {}).items():
            print(f"  {model}: {rank:.1f}")

        if 'rlvr_vs_sft' in results.get('statistical_significance', {}):
            sig = results['statistical_significance']['rlvr_vs_sft']
            verdict = "RLVR preserves base geometry MORE" if sig['rl_mean'] > sig['sft_mean'] else "SFT preserves more"
            print(f"\nKey Finding: {verdict} (p={sig['p_value']:.4f})")


if __name__ == "__main__":
    main()
