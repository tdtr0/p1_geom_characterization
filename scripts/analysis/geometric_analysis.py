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

def analyze_layer(args) -> Tuple[int, Dict]:
    """Analyze a single layer (for parallel execution)."""
    layer_idx, layer_acts, labels = args

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

        layer_results['eff_rank_correct'] = eff_rank_correct
        layer_results['eff_rank_incorrect'] = eff_rank_incorrect
        layer_results['eff_rank_diff'] = eff_rank_correct - eff_rank_incorrect

    return layer_idx, layer_results


def analyze_single_model(
    trajectories: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    n_jobs: int = 8
) -> Dict:
    """Compute geometric metrics for a single model (parallelized)."""
    from joblib import Parallel, delayed

    n_samples, seq_len, n_layers, d_model = trajectories.shape

    results = {
        'model': model_name,
        'n_samples': n_samples,
        'n_correct': int(labels.sum()),
        'n_incorrect': int((~labels).sum()),
        'layers': {}
    }

    print(f"    Analyzing {n_layers} layers with {n_jobs} workers...", flush=True)

    # Prepare layer data
    layer_data = [
        (layer_idx, aggregate_to_layer(trajectories, layer_idx), labels)
        for layer_idx in range(n_layers)
    ]

    # Parallel SVD computation
    layer_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(analyze_layer)(args) for args in layer_data
    )

    for layer_idx, layer_result in layer_results:
        results['layers'][layer_idx] = layer_result

    return results


def compute_base_layer_svd(trajectories: np.ndarray, n_jobs: int = 8) -> Dict[int, np.ndarray]:
    """
    Compute and store Vt matrices for base model (for subspace preservation).
    Returns dict of layer_idx -> Vt matrix (top-k right singular vectors).
    """
    from joblib import Parallel, delayed

    n_samples, seq_len, n_layers, d_model = trajectories.shape
    k = 50  # Keep top-k singular vectors

    def get_layer_vt(layer_idx):
        layer_acts = aggregate_to_layer(trajectories, layer_idx)
        _, _, Vt = svd(layer_acts.astype(np.float32), full_matrices=False)
        return layer_idx, Vt[:k, :]  # Only keep top-k

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(get_layer_vt)(i) for i in range(n_layers)
    )

    return {layer_idx: vt for layer_idx, vt in results}


def compare_models(
    data_dir: str,
    models: List[str],
    task: str = 'gsm8k',
    n_jobs: int = 8
) -> Dict:
    """
    Compare geometric properties across models.
    Memory-efficient: processes one model at a time.
    """
    import gc

    print("\n" + "="*70)
    print("GEOMETRIC ANALYSIS: Model Comparison")
    print("="*70)

    results = {
        'task': task,
        'models': {},
        'subspace_preservation': {},
        'effective_rank_comparison': {}
    }

    # First pass: analyze base model and cache its SVD
    base_path = Path(data_dir) / 'olmo3_base' / f"{task}_trajectories.h5"
    if not base_path.exists():
        print("⚠ Base model required for subspace preservation analysis")
        return {}

    print(f"\n{'─'*70}")
    print("Analyzing: olmo3_base (caching SVD for subspace comparison)")
    print(f"{'─'*70}")

    base_trajs, base_labels = load_trajectories(str(base_path))
    n_layers = base_trajs.shape[2]

    # Analyze base model
    base_results = analyze_single_model(base_trajs, base_labels, 'olmo3_base', n_jobs)
    results['models']['olmo3_base'] = base_results

    # Cache base Vt matrices
    print("    Caching base model SVD matrices...", flush=True)
    base_vt = compute_base_layer_svd(base_trajs, n_jobs)

    # Free base trajectories
    del base_trajs, base_labels
    gc.collect()
    print("    Base model processed, memory freed.", flush=True)

    # Second pass: analyze other models one at a time
    for model in models:
        if model == 'olmo3_base':
            continue

        filepath = Path(data_dir) / model / f"{task}_trajectories.h5"
        if not filepath.exists():
            print(f"⚠ {model}: File not found")
            continue

        print(f"\n{'─'*70}")
        print(f"Analyzing: {model}")
        print(f"{'─'*70}")

        trajectories, labels = load_trajectories(str(filepath))

        # Analyze model
        model_results = analyze_single_model(trajectories, labels, model, n_jobs)
        results['models'][model] = model_results

        # Compute subspace preservation vs base
        print(f"    Computing subspace preservation vs base...", flush=True)
        preservations = []

        for layer_idx in range(n_layers):
            other_acts = aggregate_to_layer(trajectories, layer_idx)
            _, _, Vt_other = svd(other_acts.astype(np.float32), full_matrices=False)

            # Compare with cached base Vt
            k = min(50, base_vt[layer_idx].shape[0], Vt_other.shape[0])
            V_base_k = base_vt[layer_idx][:k, :].T
            V_other_k = Vt_other[:k, :].T

            angles = subspace_angles(V_base_k, V_other_k)
            preservation = float(np.cos(angles).mean())
            preservations.append(preservation)

        mean_preservation = np.mean(preservations)
        results['subspace_preservation'][model] = {
            'per_layer': preservations,
            'mean': float(mean_preservation)
        }
        print(f"    Subspace preservation: {mean_preservation:.3f}")

        # Free memory
        del trajectories, labels
        gc.collect()
        print(f"    {model} processed, memory freed.", flush=True)

    # Summary: Effective rank comparison
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

    # Summary: Subspace preservation
    print(f"\n{'─'*70}")
    print("Subspace Preservation vs Base Model")
    print(f"{'─'*70}")

    for model, data in results.get('subspace_preservation', {}).items():
        print(f"  {model}: {data['mean']:.3f} (mean across layers)")

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
