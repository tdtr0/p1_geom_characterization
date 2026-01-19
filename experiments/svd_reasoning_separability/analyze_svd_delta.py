#!/usr/bin/env python3
"""
SVD Reasoning Separability Analysis

Compares eigenvector changes between base model and RLVR model to test
whether reasoning capabilities are separable or entangled with knowledge.

Key metric: delta_k = 1 - |cos(U_base[:,k], U_rlvr[:,k])|
- High delta = eigenvector changed significantly
- Low delta = eigenvector stayed similar
"""

import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.linalg import svd
import json
from datetime import datetime

# Configuration
DATA_DIR = Path("/data/thanhdo/trajectories_0shot")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Models to compare
BASE_MODEL = "olmo3_base"
RLVR_MODEL = "olmo3_rl_zero"

# Tasks to analyze
TASKS = ["gsm8k", "humaneval", "logiqa"]

# Number of top eigenvectors to analyze
TOP_K = 100


def load_trajectories(model: str, task: str) -> np.ndarray:
    """Load trajectory data from HDF5 file."""
    filepath = DATA_DIR / model / f"{task}_trajectories.h5"
    print(f"Loading {filepath}...")

    with h5py.File(filepath, 'r') as f:
        # Shape: (n_samples, seq_len, n_layers, d_model)
        trajectories = f['trajectories'][:]
        print(f"  Shape: {trajectories.shape}, dtype: {trajectories.dtype}")

    return trajectories.astype(np.float32)  # Convert from float16 for numerical stability


def compute_layer_svd(activations: np.ndarray, layer_idx: int, max_samples: int = 100) -> tuple:
    """
    Compute SVD for a single layer's activations.

    Args:
        activations: (n_samples, seq_len, n_layers, d_model)
        layer_idx: which layer to analyze
        max_samples: limit samples to manage memory

    Returns:
        U, S, Vh from SVD
    """
    # Extract layer: (n_samples, seq_len, d_model)
    layer_acts = activations[:max_samples, :, layer_idx, :]
    n_samples, seq_len, d_model = layer_acts.shape

    # Flatten to (n_samples * seq_len, d_model)
    flattened = layer_acts.reshape(-1, d_model)
    print(f"    Layer {layer_idx}: flattened shape {flattened.shape}")

    # Center the data
    flattened = flattened - flattened.mean(axis=0, keepdims=True)

    # Compute SVD (only need U and S for our analysis)
    U, S, Vh = svd(flattened, full_matrices=False)

    return U, S, Vh


def compute_eigenvector_delta(U_base: np.ndarray, U_rlvr: np.ndarray,
                               Vh_base: np.ndarray, Vh_rlvr: np.ndarray,
                               top_k: int = TOP_K) -> np.ndarray:
    """
    Compute delta between eigenvectors of base and RLVR models.

    Uses right singular vectors (Vh) which represent the principal directions
    in the d_model space.

    delta_k = 1 - |cos(v_base_k, v_rlvr_k)|
    """
    deltas = []

    for k in range(min(top_k, Vh_base.shape[0], Vh_rlvr.shape[0])):
        v_base = Vh_base[k, :]  # k-th principal direction from base
        v_rlvr = Vh_rlvr[k, :]  # k-th principal direction from RLVR

        # Cosine similarity (vectors are already unit norm from SVD)
        cos_sim = np.abs(np.dot(v_base, v_rlvr))
        delta = 1 - cos_sim
        deltas.append(delta)

    return np.array(deltas)


def analyze_task(task: str) -> dict:
    """Run SVD analysis for a single task."""
    print(f"\n{'='*60}")
    print(f"Analyzing task: {task}")
    print(f"{'='*60}")

    # Load trajectories
    base_traj = load_trajectories(BASE_MODEL, task)
    rlvr_traj = load_trajectories(RLVR_MODEL, task)

    n_layers = base_traj.shape[2]
    results = {
        'task': task,
        'n_layers': n_layers,
        'layers': {}
    }

    # Analyze each layer
    for layer_idx in range(n_layers):
        print(f"  Processing layer {layer_idx}...")

        # Compute SVD for both models
        U_base, S_base, Vh_base = compute_layer_svd(base_traj, layer_idx)
        U_rlvr, S_rlvr, Vh_rlvr = compute_layer_svd(rlvr_traj, layer_idx)

        # Compute eigenvector deltas
        deltas = compute_eigenvector_delta(U_base, U_rlvr, Vh_base, Vh_rlvr)

        # Store results
        results['layers'][layer_idx] = {
            'deltas': deltas.tolist(),
            'singular_values_base': S_base[:TOP_K].tolist(),
            'singular_values_rlvr': S_rlvr[:TOP_K].tolist(),
            'mean_delta_top10': float(np.mean(deltas[:10])),
            'mean_delta_top50': float(np.mean(deltas[:50])),
            'mean_delta_tail50': float(np.mean(deltas[50:])),
        }

        print(f"    Top-10 mean delta: {results['layers'][layer_idx]['mean_delta_top10']:.4f}")
        print(f"    Tail-50 mean delta: {results['layers'][layer_idx]['mean_delta_tail50']:.4f}")

    return results


def plot_results(all_results: dict):
    """Generate visualization of results."""
    n_tasks = len(all_results)

    # Figure 1: Delta vs Eigenvalue Rank (per task, averaged across layers)
    fig1, axes1 = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 4))
    if n_tasks == 1:
        axes1 = [axes1]

    for ax, (task, results) in zip(axes1, all_results.items()):
        # Average deltas across layers
        all_deltas = np.array([results['layers'][l]['deltas'] for l in results['layers']])
        mean_deltas = all_deltas.mean(axis=0)
        std_deltas = all_deltas.std(axis=0)

        x = np.arange(len(mean_deltas))
        ax.fill_between(x, mean_deltas - std_deltas, mean_deltas + std_deltas, alpha=0.3)
        ax.plot(x, mean_deltas, 'b-', linewidth=1)
        ax.set_xlabel('Eigenvalue Rank (k)')
        ax.set_ylabel('Delta (1 - |cos|)')
        ax.set_title(f'{task.upper()}\nEigenvector Change vs Rank')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline')
        ax.legend()

    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / 'delta_vs_rank.png', dpi=150)
    print(f"\nSaved: {RESULTS_DIR / 'delta_vs_rank.png'}")

    # Figure 2: Layer-wise analysis (heatmap of deltas)
    fig2, axes2 = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 6))
    if n_tasks == 1:
        axes2 = [axes2]

    for ax, (task, results) in zip(axes2, all_results.items()):
        all_deltas = np.array([results['layers'][l]['deltas'] for l in sorted(results['layers'].keys())])
        im = ax.imshow(all_deltas, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xlabel('Eigenvalue Rank (k)')
        ax.set_ylabel('Layer Index')
        ax.set_title(f'{task.upper()}\nDelta Heatmap')
        plt.colorbar(im, ax=ax, label='Delta')

    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'delta_heatmap.png', dpi=150)
    print(f"Saved: {RESULTS_DIR / 'delta_heatmap.png'}")

    # Figure 3: Top vs Tail comparison
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    tasks = list(all_results.keys())
    x = np.arange(len(tasks))
    width = 0.35

    top10_means = []
    tail50_means = []

    for task in tasks:
        results = all_results[task]
        top10 = np.mean([results['layers'][l]['mean_delta_top10'] for l in results['layers']])
        tail50 = np.mean([results['layers'][l]['mean_delta_tail50'] for l in results['layers']])
        top10_means.append(top10)
        tail50_means.append(tail50)

    bars1 = ax3.bar(x - width/2, top10_means, width, label='Top-10 Eigenvectors', color='coral')
    bars2 = ax3.bar(x + width/2, tail50_means, width, label='Tail-50 Eigenvectors', color='steelblue')

    ax3.set_ylabel('Mean Delta')
    ax3.set_title('RLVR Training: Top vs Tail Eigenvector Changes')
    ax3.set_xticks(x)
    ax3.set_xticklabels([t.upper() for t in tasks])
    ax3.legend()
    ax3.set_ylim(0, 1)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / 'top_vs_tail.png', dpi=150)
    print(f"Saved: {RESULTS_DIR / 'top_vs_tail.png'}")

    plt.close('all')


def main():
    print("="*60)
    print("SVD REASONING SEPARABILITY ANALYSIS")
    print(f"Base model: {BASE_MODEL}")
    print(f"RLVR model: {RLVR_MODEL}")
    print(f"Tasks: {TASKS}")
    print("="*60)

    all_results = {}

    for task in TASKS:
        try:
            results = analyze_task(task)
            all_results[task] = results
        except Exception as e:
            print(f"Error analyzing {task}: {e}")
            continue

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"svd_analysis_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw results: {results_file}")

    # Generate plots
    if all_results:
        plot_results(all_results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for task, results in all_results.items():
        top10 = np.mean([results['layers'][l]['mean_delta_top10'] for l in results['layers']])
        tail50 = np.mean([results['layers'][l]['mean_delta_tail50'] for l in results['layers']])
        ratio = top10 / tail50 if tail50 > 0 else float('inf')

        print(f"\n{task.upper()}:")
        print(f"  Mean delta (top-10 eigenvectors): {top10:.4f}")
        print(f"  Mean delta (tail-50 eigenvectors): {tail50:.4f}")
        print(f"  Ratio (top/tail): {ratio:.2f}")

        if ratio > 1.5:
            print(f"  -> SUGGESTS SEPARABLE (top changes more than tail)")
        elif ratio < 0.67:
            print(f"  -> SUGGESTS TAIL-HEAVY (tail changes more - unexpected)")
        else:
            print(f"  -> SUGGESTS ENTANGLED (roughly equal changes)")


if __name__ == "__main__":
    main()
