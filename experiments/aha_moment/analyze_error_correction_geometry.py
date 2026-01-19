#!/usr/bin/env python3
"""
Analyze geometry of error correction activations.

This script:
1. Loads error-detection direction from Wynroe experiment (Exp A)
2. Loads activations from error correction experiment (Exp C)
3. Computes:
   - Projection onto error-detection direction at corrupted position
   - Trajectory velocity during generation
   - Curvature during generation
   - Compare base vs think model geometry

Usage:
    python analyze_error_correction_geometry.py
"""

import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Configuration
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
WYNROE_DATA_DIR = SCRIPT_DIR / "data" / "wynroe_replication"
ACTIVATION_DIR = RESULTS_DIR / "error_correction_activations"
OUTPUT_DIR = RESULTS_DIR / "error_correction_geometry"

# Layers to analyze (matching collection)
LAYERS = list(range(0, 32, 2))  # [0, 2, 4, ..., 30]


def compute_error_detection_direction(wynroe_file: Path) -> dict:
    """
    Compute error-detection direction from Wynroe trajectory data.

    Direction = mean(corrupt_activations) - mean(clean_activations) per layer.
    Normalized to unit length.
    """
    directions = {}

    with h5py.File(wynroe_file, 'r') as f:
        n_samples = f.attrs.get('n_samples', len([k for k in f.keys() if k.startswith('sample_')]))

        # Collect clean and corrupt activations per layer
        clean_by_layer = {l: [] for l in LAYERS}
        corrupt_by_layer = {l: [] for l in LAYERS}

        for i in range(n_samples):
            grp = f[f'sample_{i:04d}']

            # Get activations at error position
            if 'clean_activation' in grp:
                clean_act = grp['clean_activation'][:]  # (n_layers, d_model)
                corrupt_act = grp['corrupt_activation'][:]
            elif 'activations' in grp:
                # Alternative format: activations at specific position
                clean_act = grp['activations']['clean'][:]
                corrupt_act = grp['activations']['corrupt'][:]
            else:
                continue

            for layer_idx, layer in enumerate(LAYERS):
                if layer_idx < clean_act.shape[0]:
                    clean_by_layer[layer].append(clean_act[layer_idx])
                    corrupt_by_layer[layer].append(corrupt_act[layer_idx])

        # Compute direction for each layer
        for layer in LAYERS:
            if clean_by_layer[layer] and corrupt_by_layer[layer]:
                clean_mean = np.mean(clean_by_layer[layer], axis=0)
                corrupt_mean = np.mean(corrupt_by_layer[layer], axis=0)

                direction = corrupt_mean - clean_mean
                direction = direction / (np.linalg.norm(direction) + 1e-8)

                directions[layer] = direction.astype(np.float32)

    return directions


def compute_error_detection_direction_from_pairs(h5_file: Path) -> dict:
    """
    Alternative: compute direction from paired trajectory data.

    Expects HDF5 with format:
    - sample_XXXX/clean_trajectory: (seq_len, n_layers, d_model)
    - sample_XXXX/corrupt_trajectory: (seq_len, n_layers, d_model)
    - sample_XXXX/error_position: int
    """
    directions = {}

    with h5py.File(h5_file, 'r') as f:
        clean_by_layer = {l_idx: [] for l_idx in range(len(LAYERS))}
        corrupt_by_layer = {l_idx: [] for l_idx in range(len(LAYERS))}

        for key in f.keys():
            if not key.startswith('sample_'):
                continue

            grp = f[key]

            # Try different data formats
            if 'clean_trajectory' in grp and 'corrupt_trajectory' in grp:
                clean_traj = grp['clean_trajectory'][:]  # (seq, layers, dim)
                corrupt_traj = grp['corrupt_trajectory'][:]
                error_pos = grp.attrs.get('error_position', -1)

                if error_pos >= 0:
                    for l_idx in range(min(clean_traj.shape[1], len(LAYERS))):
                        clean_by_layer[l_idx].append(clean_traj[error_pos, l_idx])
                        corrupt_by_layer[l_idx].append(corrupt_traj[error_pos, l_idx])

            elif 'clean_activations' in grp:
                # Another possible format
                clean_act = grp['clean_activations'][:]
                corrupt_act = grp['corrupt_activations'][:]

                for l_idx in range(min(clean_act.shape[0], len(LAYERS))):
                    clean_by_layer[l_idx].append(clean_act[l_idx])
                    corrupt_by_layer[l_idx].append(corrupt_act[l_idx])

        # Compute directions
        for l_idx, layer in enumerate(LAYERS):
            if clean_by_layer[l_idx] and corrupt_by_layer[l_idx]:
                clean_mean = np.mean(clean_by_layer[l_idx], axis=0)
                corrupt_mean = np.mean(corrupt_by_layer[l_idx], axis=0)

                direction = corrupt_mean - clean_mean
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm

                directions[layer] = direction.astype(np.float32)

    return directions


def project_onto_direction(activations: np.ndarray, direction: np.ndarray) -> float:
    """Project activation onto error-detection direction."""
    return float(np.dot(activations.flatten(), direction.flatten()))


def compute_trajectory_velocity(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute velocity (magnitude of change between consecutive tokens).

    Args:
        trajectory: (n_tokens, n_layers, d_model)

    Returns:
        velocities: (n_tokens-1,) average velocity across layers
    """
    if trajectory.shape[0] < 2:
        return np.array([])

    # Compute differences
    diffs = trajectory[1:] - trajectory[:-1]  # (n_tokens-1, n_layers, d_model)

    # Compute magnitude per layer, then average
    velocities = np.linalg.norm(diffs, axis=2).mean(axis=1)  # (n_tokens-1,)

    return velocities


def compute_trajectory_curvature(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute Menger curvature along trajectory.

    For points p1, p2, p3, curvature = 4 * area(triangle) / (|p1-p2| * |p2-p3| * |p1-p3|)

    Args:
        trajectory: (n_tokens, n_layers, d_model)

    Returns:
        curvatures: (n_tokens-2,) curvature at each middle point
    """
    if trajectory.shape[0] < 3:
        return np.array([])

    # Flatten layers for curvature computation
    flat_traj = trajectory.reshape(trajectory.shape[0], -1)  # (n_tokens, n_layers*d_model)

    curvatures = []
    for i in range(1, flat_traj.shape[0] - 1):
        p1, p2, p3 = flat_traj[i-1], flat_traj[i], flat_traj[i+1]

        # Edge lengths
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        # Semi-perimeter
        s = (a + b + c) / 2

        # Area via Heron's formula
        area_sq = s * (s - a) * (s - b) * (s - c)
        area = np.sqrt(max(area_sq, 0))

        # Menger curvature
        denom = a * b * c
        if denom > 1e-8:
            curvature = 4 * area / denom
        else:
            curvature = 0

        curvatures.append(curvature)

    return np.array(curvatures)


def analyze_model_activations(model_name: str, directions: dict) -> dict:
    """
    Analyze activations for a single model.

    Returns statistics about:
    - Projection onto error-detection direction
    - Velocity during generation
    - Curvature during generation
    """
    activation_file = ACTIVATION_DIR / f"{model_name}_activations.h5"

    if not activation_file.exists():
        print(f"Warning: {activation_file} not found")
        return None

    results = {
        'model': model_name,
        'samples': [],
        'aggregate': {}
    }

    # Per-layer projections at last prompt token (where error is)
    projections_by_layer = {l: [] for l in LAYERS}

    all_velocities = []
    all_curvatures = []

    with h5py.File(activation_file, 'r') as f:
        for key in sorted(f.keys()):
            if not key.startswith('sample_'):
                continue

            grp = f[key]

            sample_result = {
                'problem_id': grp.attrs.get('problem_id', key),
                'ground_truth': float(grp.attrs.get('ground_truth', 0)),
                'corrupted_value': float(grp.attrs.get('corrupted_value', 0)),
            }

            # Get activations
            prompt_act = grp['prompt_activations'][:]  # (prompt_len, n_layers, d_model)
            gen_act = grp['gen_activations'][:]  # (gen_len, n_layers, d_model)

            # Project last prompt token (where corrupted calculation is)
            if prompt_act.shape[0] > 0:
                last_prompt_act = prompt_act[-1]  # (n_layers, d_model)

                for l_idx, layer in enumerate(LAYERS):
                    if l_idx < last_prompt_act.shape[0] and layer in directions:
                        proj = project_onto_direction(last_prompt_act[l_idx], directions[layer])
                        projections_by_layer[layer].append(proj)

            # Compute velocity and curvature on generation
            if gen_act.shape[0] > 1:
                velocity = compute_trajectory_velocity(gen_act)
                all_velocities.extend(velocity.tolist())
                sample_result['mean_velocity'] = float(velocity.mean()) if len(velocity) > 0 else 0

            if gen_act.shape[0] > 2:
                curvature = compute_trajectory_curvature(gen_act)
                all_curvatures.extend(curvature.tolist())
                sample_result['mean_curvature'] = float(curvature.mean()) if len(curvature) > 0 else 0

            results['samples'].append(sample_result)

    # Aggregate statistics
    results['aggregate'] = {
        'n_samples': len(results['samples']),
        'projection_by_layer': {},
        'velocity': {
            'mean': float(np.mean(all_velocities)) if all_velocities else 0,
            'std': float(np.std(all_velocities)) if all_velocities else 0,
        },
        'curvature': {
            'mean': float(np.mean(all_curvatures)) if all_curvatures else 0,
            'std': float(np.std(all_curvatures)) if all_curvatures else 0,
        }
    }

    for layer in LAYERS:
        if projections_by_layer[layer]:
            results['aggregate']['projection_by_layer'][layer] = {
                'mean': float(np.mean(projections_by_layer[layer])),
                'std': float(np.std(projections_by_layer[layer])),
            }

    return results


def compare_models(results: dict) -> dict:
    """Compare geometric measures between models."""
    comparisons = {}

    models = list(results.keys())

    # Velocity comparison
    if len(models) >= 2:
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                if results[m1] and results[m2]:
                    v1 = results[m1]['aggregate']['velocity']
                    v2 = results[m2]['aggregate']['velocity']

                    comparisons[f'{m1}_vs_{m2}_velocity'] = {
                        f'{m1}_mean': v1['mean'],
                        f'{m2}_mean': v2['mean'],
                        'diff': v1['mean'] - v2['mean'],
                    }

                    c1 = results[m1]['aggregate']['curvature']
                    c2 = results[m2]['aggregate']['curvature']

                    comparisons[f'{m1}_vs_{m2}_curvature'] = {
                        f'{m1}_mean': c1['mean'],
                        f'{m2}_mean': c2['mean'],
                        'diff': c1['mean'] - c2['mean'],
                    }

    return comparisons


def plot_results(results: dict, output_dir: Path):
    """Generate visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m for m in results.keys() if results[m] is not None]

    if not models:
        print("No models to plot")
        return

    # Plot 1: Projection by layer
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        layers = []
        means = []
        stds = []
        for layer in LAYERS:
            if layer in results[model]['aggregate']['projection_by_layer']:
                layers.append(layer)
                means.append(results[model]['aggregate']['projection_by_layer'][layer]['mean'])
                stds.append(results[model]['aggregate']['projection_by_layer'][layer]['std'])

        if layers:
            ax.errorbar(layers, means, yerr=stds, label=model, marker='o', capsize=3)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Projection onto Error-Detection Direction')
    ax.set_title('Error Detection Signal at Corrupted Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'projection_by_layer.png', dpi=150)
    plt.close()

    # Plot 2: Velocity comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    velocities = [results[m]['aggregate']['velocity']['mean'] for m in models]
    velocity_stds = [results[m]['aggregate']['velocity']['std'] for m in models]
    bars = ax.bar(models, velocities, yerr=velocity_stds, capsize=5)
    ax.set_ylabel('Mean Trajectory Velocity')
    ax.set_title('Velocity During Generation')
    plt.tight_layout()
    plt.savefig(output_dir / 'velocity_comparison.png', dpi=150)
    plt.close()

    # Plot 3: Curvature comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    curvatures = [results[m]['aggregate']['curvature']['mean'] for m in models]
    curvature_stds = [results[m]['aggregate']['curvature']['std'] for m in models]
    bars = ax.bar(models, curvatures, yerr=curvature_stds, capsize=5)
    ax.set_ylabel('Mean Trajectory Curvature')
    ax.set_title('Curvature During Generation')
    plt.tight_layout()
    plt.savefig(output_dir / 'curvature_comparison.png', dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load or compute error-detection direction
    print("Computing error-detection direction from Wynroe data...")

    wynroe_file = WYNROE_DATA_DIR / "wynroe_trajectories.h5"
    if wynroe_file.exists():
        directions = compute_error_detection_direction_from_pairs(wynroe_file)
        print(f"Loaded directions for {len(directions)} layers")
    else:
        print(f"Warning: {wynroe_file} not found")
        print("Looking for alternative data sources...")

        # Try to find any h5 file in the data directory
        for h5_file in SCRIPT_DIR.glob("data/**/*.h5"):
            print(f"Trying {h5_file}...")
            directions = compute_error_detection_direction_from_pairs(h5_file)
            if directions:
                print(f"Found directions from {h5_file}")
                break
        else:
            print("No direction data found. Creating synthetic direction for testing.")
            # Create random directions for testing (will be replaced with real data)
            directions = {l: np.random.randn(4096).astype(np.float32) for l in LAYERS}
            for l in directions:
                directions[l] /= np.linalg.norm(directions[l])

    # Step 2: Analyze each model
    print("\nAnalyzing model activations...")
    results = {}

    for model_name in ['base', 'rl_zero', 'think']:
        print(f"  Analyzing {model_name}...")
        results[model_name] = analyze_model_activations(model_name, directions)

    # Step 3: Compare models
    print("\nComparing models...")
    comparisons = compare_models(results)

    # Step 4: Save results
    output_file = OUTPUT_DIR / "geometry_analysis.json"
    output_data = {
        'models': {k: v for k, v in results.items() if v is not None},
        'comparisons': comparisons,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_file}")

    # Step 5: Plot
    print("\nGenerating plots...")
    plot_results(results, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model_name, model_results in results.items():
        if model_results:
            agg = model_results['aggregate']
            print(f"\n{model_name}:")
            print(f"  Samples: {agg['n_samples']}")
            print(f"  Velocity: {agg['velocity']['mean']:.4f} ± {agg['velocity']['std']:.4f}")
            print(f"  Curvature: {agg['curvature']['mean']:.6f} ± {agg['curvature']['std']:.6f}")

            # Best layer projection
            best_layer = max(
                agg['projection_by_layer'].keys(),
                key=lambda l: abs(agg['projection_by_layer'][l]['mean']),
                default=None
            )
            if best_layer:
                proj = agg['projection_by_layer'][best_layer]
                print(f"  Best projection (layer {best_layer}): {proj['mean']:.4f} ± {proj['std']:.4f}")


if __name__ == "__main__":
    main()
