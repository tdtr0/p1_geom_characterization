#!/usr/bin/env python3
"""
Analyze Wynroe-style error detection direction from clean/corrupted pairs.

This script:
1. Loads clean/corrupted trajectory pairs
2. Computes error-detection direction (mean difference)
3. Tests phase transition sharpness at error token
4. Compares across models (Base vs SFT vs RL-Zero vs Think)

Usage:
    python analyze_wynroe_direction.py \
        --input data/wynroe_replication/wynroe_trajectories.h5 \
        --output results/wynroe/
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm


# ============================================================================
# Direction Extraction
# ============================================================================

def compute_error_direction(
    clean_activations: np.ndarray,
    corrupted_activations: np.ndarray,
) -> np.ndarray:
    """
    Compute error-detection direction as mean difference.

    Args:
        clean_activations: (n_samples, n_layers, hidden_dim)
        corrupted_activations: (n_samples, n_layers, hidden_dim)

    Returns:
        direction: (n_layers, hidden_dim) - normalized direction vector
    """
    # Mean difference
    diff = corrupted_activations - clean_activations  # (n_samples, n_layers, hidden)
    direction = diff.mean(axis=0)  # (n_layers, hidden)

    # Normalize per layer
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction_normalized = direction / (norms + 1e-8)

    return direction_normalized


def project_onto_direction(
    activations: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Project activations onto error-detection direction.

    Args:
        activations: (n_samples, n_layers, hidden_dim) or (n_layers, hidden_dim)
        direction: (n_layers, hidden_dim)

    Returns:
        projections: (n_samples, n_layers) or (n_layers,)
    """
    if activations.ndim == 2:
        # Single sample
        projections = np.sum(activations * direction, axis=-1)
    else:
        # Multiple samples
        projections = np.sum(activations * direction[None, :, :], axis=-1)
    return projections


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_effect_size(clean_proj: np.ndarray, corrupt_proj: np.ndarray) -> dict:
    """
    Compute effect size statistics for error direction.

    Args:
        clean_proj: (n_samples, n_layers) projections for clean traces
        corrupt_proj: (n_samples, n_layers) projections for corrupted traces

    Returns:
        Dict with statistics per layer
    """
    n_layers = clean_proj.shape[1]
    results = []

    for layer in range(n_layers):
        clean = clean_proj[:, layer]
        corrupt = corrupt_proj[:, layer]

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(clean) + np.var(corrupt)) / 2)
        d = (np.mean(corrupt) - np.mean(clean)) / (pooled_std + 1e-8)

        # t-test
        t_stat, p_val = stats.ttest_rel(corrupt, clean)

        results.append({
            'layer': layer,
            'clean_mean': float(np.mean(clean)),
            'clean_std': float(np.std(clean)),
            'corrupt_mean': float(np.mean(corrupt)),
            'corrupt_std': float(np.std(corrupt)),
            'effect_size_d': float(d),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
        })

    return results


def analyze_model(
    h5file,
    model_key: str,
    direction: np.ndarray = None,
) -> dict:
    """
    Analyze a single model's clean/corrupted pairs.

    Args:
        h5file: Open HDF5 file
        model_key: Model name in file
        direction: Pre-computed direction (if None, compute from this model)

    Returns:
        Analysis results dict
    """
    clean_group = h5file[f'{model_key}/clean']
    corrupt_group = h5file[f'{model_key}/corrupted']

    # Load all pairs
    clean_acts = []
    corrupt_acts = []

    n_pairs = len(clean_group.keys())

    for i in range(n_pairs):
        key = f'pair_{i}'
        if key in clean_group and key in corrupt_group:
            clean_acts.append(clean_group[key][:])
            corrupt_acts.append(corrupt_group[key][:])

    clean_acts = np.array(clean_acts)  # (n_pairs, n_layers, hidden)
    corrupt_acts = np.array(corrupt_acts)

    print(f"  Loaded {len(clean_acts)} pairs, shape: {clean_acts.shape}")

    # Compute direction if not provided
    if direction is None:
        direction = compute_error_direction(clean_acts, corrupt_acts)
        computed_direction = True
    else:
        computed_direction = False

    # Project onto direction
    clean_proj = project_onto_direction(clean_acts, direction)
    corrupt_proj = project_onto_direction(corrupt_acts, direction)

    # Compute effect sizes
    effect_results = compute_effect_size(clean_proj, corrupt_proj)

    # Find best layer
    best_layer = max(effect_results, key=lambda x: abs(x['effect_size_d']))

    return {
        'model': model_key,
        'n_pairs': len(clean_acts),
        'direction_computed': computed_direction,
        'per_layer': effect_results,
        'best_layer': best_layer['layer'],
        'best_effect_size': best_layer['effect_size_d'],
        'direction': direction if computed_direction else None,
        'clean_projections': clean_proj,
        'corrupt_projections': corrupt_proj,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_layer_profile(results: dict, output_dir: str):
    """Plot effect size across layers for all models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    markers = ['o', 's', '^', 'D']

    for i, (model_key, result) in enumerate(results.items()):
        layers = [r['layer'] for r in result['per_layer']]
        effect_sizes = [r['effect_size_d'] for r in result['per_layer']]

        ax.plot(layers, effect_sizes, f'-{markers[i % len(markers)]}',
                color=colors[i % len(colors)], label=model_key, linewidth=2, markersize=8)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='d=0.5 (medium effect)')
    ax.axhline(y=-0.5, color='green', linestyle=':', alpha=0.5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title('Error Detection Direction: Effect Size by Layer', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'layer_profile.png', dpi=150)
    plt.close()


def plot_projection_distributions(results: dict, output_dir: str):
    """Plot projection distributions for clean vs corrupted."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_key, result) in zip(axes, results.items()):
        best_layer = result['best_layer']

        clean_proj = result['clean_projections'][:, best_layer]
        corrupt_proj = result['corrupt_projections'][:, best_layer]

        ax.hist(clean_proj, bins=30, alpha=0.6, label='Clean', color='#3498db')
        ax.hist(corrupt_proj, bins=30, alpha=0.6, label='Corrupted', color='#e74c3c')

        ax.axvline(np.mean(clean_proj), color='#2980b9', linestyle='--', linewidth=2)
        ax.axvline(np.mean(corrupt_proj), color='#c0392b', linestyle='--', linewidth=2)

        ax.set_xlabel('Projection onto Error Direction', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{model_key} (Layer {best_layer})\nd = {result["best_effect_size"]:.2f}', fontsize=12)
        ax.legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'projection_distributions.png', dpi=150)
    plt.close()


def plot_model_comparison(results: dict, output_dir: str):
    """Bar chart comparing models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    effect_sizes = [results[m]['best_effect_size'] for m in models]
    best_layers = [results[m]['best_layer'] for m in models]

    colors = ['#3498db' if e > 0.5 else '#95a5a6' for e in effect_sizes]

    bars = ax.bar(models, effect_sizes, color=colors, edgecolor='black', linewidth=1)

    # Add layer annotations
    for bar, layer in zip(bars, best_layers):
        height = bar.get_height()
        ax.annotate(f'L{layer}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='d=0.5 (medium)')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='d=0.8 (large)')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel("Best Layer Effect Size (Cohen's d)", fontsize=12)
    ax.set_title('Error Detection Direction: Model Comparison', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'model_comparison.png', dpi=150)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze Wynroe-style error detection direction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to wynroe_trajectories.h5'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/wynroe/',
        help='Output directory'
    )
    parser.add_argument(
        '--reference-model',
        type=str,
        default='rl_zero',
        help='Model to use for computing reference direction (default: rl_zero)'
    )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading data from: {args.input}")

    with h5py.File(args.input, 'r') as f:
        models = f.attrs['models']
        n_pairs = f.attrs['n_pairs']

        print(f"Models: {models}")
        print(f"Total pairs: {n_pairs}")

        all_results = {}
        reference_direction = None

        # First pass: compute direction from reference model
        if args.reference_model in f:
            print(f"\nComputing direction from reference model: {args.reference_model}")
            ref_result = analyze_model(f, args.reference_model)
            reference_direction = ref_result['direction']
            all_results[args.reference_model] = ref_result
        else:
            print(f"Warning: Reference model {args.reference_model} not found")

        # Second pass: analyze all models using reference direction
        for model_key in models:
            if model_key == args.reference_model:
                continue  # Already analyzed

            if model_key not in f:
                print(f"Warning: Model {model_key} not in file, skipping")
                continue

            print(f"\nAnalyzing model: {model_key}")
            result = analyze_model(f, model_key, direction=reference_direction)
            all_results[model_key] = result

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for model_key, result in all_results.items():
        print(f"\n{model_key}:")
        print(f"  Pairs: {result['n_pairs']}")
        print(f"  Best layer: {result['best_layer']}")
        print(f"  Effect size (d): {result['best_effect_size']:.3f}")

        best = result['per_layer'][result['best_layer']]
        if best['p_value'] < 0.001:
            sig = "***"
        elif best['p_value'] < 0.01:
            sig = "**"
        elif best['p_value'] < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        print(f"  p-value: {best['p_value']:.4f} {sig}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    for model_key, result in all_results.items():
        d = result['best_effect_size']
        if abs(d) > 0.8:
            strength = "STRONG"
        elif abs(d) > 0.5:
            strength = "MEDIUM"
        elif abs(d) > 0.2:
            strength = "WEAK"
        else:
            strength = "NONE"

        print(f"{model_key}: {strength} error-detection signal (d={d:.2f})")

    # Generate plots
    print(f"\nGenerating plots in: {args.output}")

    # Remove numpy arrays before saving JSON
    results_for_json = {}
    for model_key, result in all_results.items():
        results_for_json[model_key] = {
            'model': result['model'],
            'n_pairs': result['n_pairs'],
            'best_layer': result['best_layer'],
            'best_effect_size': result['best_effect_size'],
            'per_layer': result['per_layer'],
        }

    plot_layer_profile(all_results, args.output)
    plot_projection_distributions(all_results, args.output)
    plot_model_comparison(all_results, args.output)

    # Save results
    results_file = Path(args.output) / 'wynroe_analysis.json'
    with open(results_file, 'w') as f:
        json.dump({
            'reference_model': args.reference_model,
            'models': results_for_json,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Plots saved to: {args.output}")


if __name__ == '__main__':
    main()
