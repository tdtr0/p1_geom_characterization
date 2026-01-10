#!/usr/bin/env python3
"""
Geometric Analysis Pipeline

Computes geometric measures (effective rank, spectral decay, subspace preservation)
for all collected activation data.
"""

import sys
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.linalg import svd, subspace_angles
from scipy.stats import entropy
from typing import Dict, List, Tuple
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def compute_effective_rank(activations: np.ndarray) -> float:
    """
    Effective rank: exponential of entropy of normalized singular values.

    Measures how many dimensions are "actively used" in the representation.
    Low effective rank = concentrated on few dimensions (sparse/compressed)
    High effective rank = spread across many dimensions (distributed)

    Args:
        activations: (n_samples, d_model)

    Returns:
        Effective rank (scalar between 1 and min(n_samples, d_model))
    """
    # SVD
    _, s, _ = svd(activations, full_matrices=False)

    # Normalize to probability distribution
    s_normalized = s / s.sum()

    # Remove zeros for entropy calculation
    s_normalized = s_normalized[s_normalized > 1e-10]

    # Effective rank = exp(entropy)
    return np.exp(entropy(s_normalized))


def compute_spectral_decay(activations: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Fit power-law to singular value decay: s_i ∝ i^(-α)

    Higher α = faster decay = more concentrated spectrum
    Lower α = slower decay = more distributed spectrum

    Returns:
        alpha: Power-law exponent
        singular_values: Raw singular values for inspection
    """
    _, s, _ = svd(activations, full_matrices=False)

    # Fit log-log regression: log(s) = -α * log(i) + c
    log_i = np.log(np.arange(1, len(s) + 1))
    log_s = np.log(s + 1e-10)  # Add epsilon for numerical stability

    # Least squares fit
    A = np.vstack([log_i, np.ones_like(log_i)]).T
    alpha_neg, _ = np.linalg.lstsq(A, log_s, rcond=None)[0]

    return -alpha_neg, s


def compute_subspace_preservation(
    base_activations: np.ndarray,
    finetuned_activations: np.ndarray,
    k: int = 100
) -> Tuple[float, np.ndarray]:
    """
    Measure how much of base model's top-k subspace is preserved after fine-tuning.

    Uses principal angles between subspaces.
    Preservation = 1: Perfect preservation (identical subspaces)
    Preservation = 0: Orthogonal subspaces (no overlap)

    Args:
        base_activations: (n_samples, d_model) from base model
        finetuned_activations: (n_samples, d_model) from fine-tuned model
        k: Number of top singular vectors to compare

    Returns:
        preservation_score: Mean cosine of principal angles
        angles: Individual principal angles (in radians)
    """
    # Get top-k right singular vectors
    _, _, Vt_base = svd(base_activations, full_matrices=False)
    _, _, Vt_ft = svd(finetuned_activations, full_matrices=False)

    V_base_k = Vt_base[:k, :].T  # (d_model, k)
    V_ft_k = Vt_ft[:k, :].T

    # Compute principal angles using scipy
    angles = subspace_angles(V_base_k, V_ft_k)

    # Preservation score
    preservation = np.cos(angles).mean()

    return preservation, angles


def analyze_activations(
    data_dir: str,
    models: List[str],
    tasks: List[str],
    k_subspace: int = 100,
    layers_to_analyze: List[int] = None
) -> pd.DataFrame:
    """
    Compute all geometric measures for all model/task combinations.

    Returns:
        DataFrame with columns:
        - model, task, layer
        - effective_rank, spectral_decay_alpha
        - preservation_vs_base (only for fine-tuned models)
    """
    results = []

    # Load base model activations for preservation comparison
    print("Loading base model activations...")
    base_activations = {}  # task -> layer -> activations

    for task in tasks:
        base_path = Path(data_dir) / "olmo3_base" / f"{task}.h5"
        if not base_path.exists():
            print(f"⚠ Base activations not found for {task}")
            continue

        with h5py.File(base_path, 'r') as f:
            base_activations[task] = {}
            for key in f.keys():
                if key.startswith('layer_') and not key.startswith('metadata'):
                    layer = int(key.split('_')[1])
                    base_activations[task][layer] = f[key][:]

        print(f"  ✓ Loaded {len(base_activations[task])} layers for {task}")

    # Analyze each model
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*60}")

        model_dir = Path(data_dir) / model_name
        if not model_dir.exists():
            print(f"⚠ Model directory not found: {model_dir}")
            continue

        for task in tasks:
            filepath = model_dir / f"{task}.h5"
            if not filepath.exists():
                print(f"  ⚠ {task}: File not found")
                continue

            print(f"\n  Task: {task}")

            with h5py.File(filepath, 'r') as f:
                # Get all layer keys
                layer_keys = [k for k in f.keys() if k.startswith('layer_')]

                for key in layer_keys:
                    layer = int(key.split('_')[1])

                    if layers_to_analyze and layer not in layers_to_analyze:
                        continue

                    activations = f[key][:]

                    # Compute measures
                    try:
                        eff_rank = compute_effective_rank(activations)
                        alpha, _ = compute_spectral_decay(activations)

                        # Subspace preservation (vs base model)
                        preservation = None
                        if model_name != "olmo3_base" and task in base_activations:
                            base_acts = base_activations[task].get(layer)
                            if base_acts is not None and base_acts.shape == activations.shape:
                                preservation, _ = compute_subspace_preservation(
                                    base_acts, activations, k=min(k_subspace, base_acts.shape[0], base_acts.shape[1])
                                )

                        results.append({
                            'model': model_name,
                            'task': task,
                            'layer': layer,
                            'effective_rank': eff_rank,
                            'spectral_decay_alpha': alpha,
                            'preservation_vs_base': preservation,
                            'n_samples': activations.shape[0],
                            'd_model': activations.shape[1],
                        })

                    except Exception as e:
                        print(f"    ⚠ Layer {layer}: {e}")
                        continue

            print(f"    ✓ Analyzed {len(layer_keys)} layers")

    return pd.DataFrame(results)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-layer results to model-level summaries.
    """
    # Group by model and task
    grouped = df.groupby(['model', 'task']).agg({
        'effective_rank': ['mean', 'std'],
        'spectral_decay_alpha': ['mean', 'std'],
        'preservation_vs_base': ['mean', 'std'],
        'n_samples': 'first',
        'd_model': 'first',
    })

    grouped.columns = ['_'.join(col).strip() if col[1] else col[0]
                      for col in grouped.columns.values]
    return grouped.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Run geometric analysis on collected activations")
    parser.add_argument('--data-dir', type=str, default='data/activations',
                       help='Directory containing activation data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save analysis results')
    parser.add_argument('--k-subspace', type=int, default=100,
                       help='Number of top singular vectors for subspace preservation')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer indices to analyze (default: all)')

    args = parser.parse_args()

    # Parse layers
    layers_to_analyze = None
    if args.layers:
        layers_to_analyze = [int(x) for x in args.layers.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Models and tasks
    models = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']
    tasks = ['gsm8k', 'humaneval']

    print("="*60)
    print("GEOMETRIC ANALYSIS PIPELINE")
    print("="*60)
    print(f"\nModels: {models}")
    print(f"Tasks: {tasks}")
    print(f"k_subspace: {args.k_subspace}")
    if layers_to_analyze:
        print(f"Analyzing layers: {layers_to_analyze}")
    else:
        print("Analyzing all layers")

    # Run analysis
    print("\n" + "="*60)
    print("Computing geometric measures...")
    print("="*60)

    df = analyze_activations(
        args.data_dir,
        models,
        tasks,
        k_subspace=args.k_subspace,
        layers_to_analyze=layers_to_analyze
    )

    # Save detailed results
    detail_file = output_dir / 'geometric_analysis_detailed.csv'
    df.to_csv(detail_file, index=False)
    print(f"\n✓ Saved detailed results to {detail_file}")

    # Aggregate and save
    print("\n" + "="*60)
    print("Aggregating results...")
    print("="*60)

    agg_df = aggregate_results(df)
    agg_file = output_dir / 'geometric_analysis_summary.csv'
    agg_df.to_csv(agg_file, index=False)
    print(f"✓ Saved summary results to {agg_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print("\nEffective Rank (mean ± std):")
    print(agg_df[['model', 'task', 'effective_rank_mean', 'effective_rank_std']])

    print("\nSubspace Preservation vs Base (mean ± std):")
    preservation_cols = agg_df[['model', 'task', 'preservation_vs_base_mean', 'preservation_vs_base_std']]
    print(preservation_cols[agg_df['model'] != 'olmo3_base'])

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
