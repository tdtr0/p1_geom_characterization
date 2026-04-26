#!/usr/bin/env python3
"""
SVD Reasoning Separability Analysis (Memory-Efficient Version)

Compares eigenvector changes between base model and RLVR model to test
whether reasoning capabilities are separable or entangled with knowledge.

Key metric: delta_k = 1 - |cos(U_base[:,k], U_rlvr[:,k])|
- High delta = eigenvector changed significantly
- Low delta = eigenvector stayed similar
"""

import h5py
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
from datetime import datetime
import gc
import sys
import time

# Configuration
DATA_DIR = Path("/data/thanhdo/trajectories_0shot")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Status file for dashboard
STATUS_FILE = Path(__file__).parent / "status.json"

# Models to compare
BASE_MODEL = "olmo3_base"
RLVR_MODEL = "olmo3_rl_zero"

# Tasks to analyze (start with smaller ones)
TASKS = ["humaneval", "gsm8k"]  # logiqa is too large (13GB)

# Analysis parameters
MAX_SAMPLES = 50  # Reduced for memory
MAX_SEQ_LEN = 256  # Truncate sequences
TOP_K = 100


def log(msg: str):
    """Print with timestamp and flush immediately (keepalive)."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def update_status(status: dict):
    """Write status to file for dashboard."""
    status['last_update'] = datetime.now().isoformat()
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)


def load_layer_data(filepath: Path, layer_idx: int, max_samples: int = MAX_SAMPLES,
                    max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Load only a single layer's data from HDF5 file (memory efficient)."""
    with h5py.File(filepath, 'r') as f:
        # Shape: (n_samples, seq_len, n_layers, d_model)
        n_samples = min(max_samples, f['trajectories'].shape[0])
        seq_len = min(max_seq_len, f['trajectories'].shape[1])

        # Load only the specific layer slice
        layer_data = f['trajectories'][:n_samples, :seq_len, layer_idx, :]

    return layer_data.astype(np.float32)


def get_file_info(filepath: Path) -> dict:
    """Get metadata about HDF5 file without loading data."""
    with h5py.File(filepath, 'r') as f:
        return {
            'n_samples': f['trajectories'].shape[0],
            'seq_len': f['trajectories'].shape[1],
            'n_layers': f['trajectories'].shape[2],
            'd_model': f['trajectories'].shape[3],
        }


def compute_svd_for_layer(layer_data: np.ndarray) -> tuple:
    """
    Compute SVD for a layer's activations using randomized SVD (much faster).

    Args:
        layer_data: (n_samples, seq_len, d_model)

    Returns:
        S (singular values), Vh (right singular vectors)
    """
    from sklearn.decomposition import TruncatedSVD

    n_samples, seq_len, d_model = layer_data.shape

    # Flatten to (n_samples * seq_len, d_model)
    flattened = layer_data.reshape(-1, d_model)

    # Center the data
    flattened = flattened - flattened.mean(axis=0, keepdims=True)

    # Use randomized SVD - MUCH faster for large matrices
    # algorithm='randomized' is O(n * k^2) instead of O(n * d^2) for full SVD
    svd = TruncatedSVD(n_components=TOP_K, algorithm='randomized', n_iter=5, random_state=42)
    svd.fit(flattened)

    return svd.singular_values_, svd.components_


def compute_eigenvector_delta(Vh_base: np.ndarray, Vh_rlvr: np.ndarray) -> np.ndarray:
    """
    Compute delta between eigenvectors of base and RLVR models.

    delta_k = 1 - |cos(v_base_k, v_rlvr_k)|
    """
    deltas = []
    k_max = min(TOP_K, Vh_base.shape[0], Vh_rlvr.shape[0])

    for k in range(k_max):
        v_base = Vh_base[k, :]
        v_rlvr = Vh_rlvr[k, :]

        # Normalize (should already be unit norm, but ensure)
        v_base = v_base / (np.linalg.norm(v_base) + 1e-10)
        v_rlvr = v_rlvr / (np.linalg.norm(v_rlvr) + 1e-10)

        # Cosine similarity
        cos_sim = np.abs(np.dot(v_base, v_rlvr))
        delta = 1 - cos_sim
        deltas.append(delta)

    return np.array(deltas)


def analyze_task(task: str, task_idx: int, total_tasks: int) -> dict:
    """Run SVD analysis for a single task."""
    log(f"")
    log(f"{'='*60}")
    log(f"TASK {task_idx+1}/{total_tasks}: {task.upper()}")
    log(f"{'='*60}")

    base_path = DATA_DIR / BASE_MODEL / f"{task}_trajectories.h5"
    rlvr_path = DATA_DIR / RLVR_MODEL / f"{task}_trajectories.h5"

    # Get file info
    info = get_file_info(base_path)
    log(f"File shape: {info['n_samples']} samples × {info['seq_len']} seq × {info['n_layers']} layers × {info['d_model']} dim")
    log(f"Using: {MAX_SAMPLES} samples × {MAX_SEQ_LEN} seq")

    n_layers = info['n_layers']
    results = {
        'task': task,
        'n_layers': n_layers,
        'max_samples': MAX_SAMPLES,
        'max_seq_len': MAX_SEQ_LEN,
        'layers': {}
    }

    layer_times = []
    start_time = time.time()

    # Analyze each layer separately (memory efficient)
    for layer_idx in range(n_layers):
        layer_start = time.time()

        # Update status for dashboard
        update_status({
            'state': 'running',
            'task': task,
            'task_idx': task_idx + 1,
            'total_tasks': total_tasks,
            'layer': layer_idx,
            'total_layers': n_layers,
            'progress_pct': round((task_idx * n_layers + layer_idx) / (total_tasks * n_layers) * 100, 1)
        })

        log(f"  Layer {layer_idx:2d}/{n_layers-1} | Loading data...", )

        # Load layer data for both models
        base_data = load_layer_data(base_path, layer_idx)
        rlvr_data = load_layer_data(rlvr_path, layer_idx)

        log(f"  Layer {layer_idx:2d}/{n_layers-1} | Computing SVD (base)...")

        # Compute SVD
        S_base, Vh_base = compute_svd_for_layer(base_data)
        del base_data
        gc.collect()

        log(f"  Layer {layer_idx:2d}/{n_layers-1} | Computing SVD (rlvr)...")

        S_rlvr, Vh_rlvr = compute_svd_for_layer(rlvr_data)
        del rlvr_data
        gc.collect()

        # Compute eigenvector deltas
        deltas = compute_eigenvector_delta(Vh_base, Vh_rlvr)

        del Vh_base, Vh_rlvr
        gc.collect()

        layer_time = time.time() - layer_start
        layer_times.append(layer_time)

        # Store results
        results['layers'][layer_idx] = {
            'deltas': deltas.tolist(),
            'singular_values_base': S_base.tolist(),
            'singular_values_rlvr': S_rlvr.tolist(),
            'mean_delta_top10': float(np.mean(deltas[:10])),
            'mean_delta_top50': float(np.mean(deltas[:50])) if len(deltas) >= 50 else float(np.mean(deltas)),
            'mean_delta_tail50': float(np.mean(deltas[50:])) if len(deltas) > 50 else float(np.mean(deltas[len(deltas)//2:])),
        }

        # Estimate remaining time
        avg_time = np.mean(layer_times)
        remaining_layers = n_layers - layer_idx - 1
        remaining_tasks_layers = (total_tasks - task_idx - 1) * n_layers
        eta_seconds = (remaining_layers + remaining_tasks_layers) * avg_time
        eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 60 else f"{int(eta_seconds)}s"

        log(f"  Layer {layer_idx:2d}/{n_layers-1} | DONE in {layer_time:.1f}s | "
            f"top10={results['layers'][layer_idx]['mean_delta_top10']:.3f}, "
            f"tail={results['layers'][layer_idx]['mean_delta_tail50']:.3f} | "
            f"ETA: {eta_str}")

    task_time = time.time() - start_time
    log(f"Task {task.upper()} completed in {task_time:.1f}s ({task_time/60:.1f}m)")

    return results


def plot_results(all_results: dict):
    """Generate visualization of results."""
    log("Generating plots...")
    n_tasks = len(all_results)
    if n_tasks == 0:
        return

    # Figure 1: Delta vs Eigenvalue Rank (per task, averaged across layers)
    fig1, axes1 = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 4))
    if n_tasks == 1:
        axes1 = [axes1]

    for ax, (task, results) in zip(axes1, all_results.items()):
        # Average deltas across layers
        all_deltas = np.array([results['layers'][str(l)]['deltas']
                              for l in range(results['n_layers'])])
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
    log(f"Saved: {RESULTS_DIR / 'delta_vs_rank.png'}")

    # Figure 2: Layer-wise analysis (heatmap of deltas)
    fig2, axes2 = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 6))
    if n_tasks == 1:
        axes2 = [axes2]

    for ax, (task, results) in zip(axes2, all_results.items()):
        all_deltas = np.array([results['layers'][str(l)]['deltas']
                              for l in range(results['n_layers'])])
        im = ax.imshow(all_deltas, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xlabel('Eigenvalue Rank (k)')
        ax.set_ylabel('Layer Index')
        ax.set_title(f'{task.upper()}\nDelta Heatmap')
        plt.colorbar(im, ax=ax, label='Delta')

    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'delta_heatmap.png', dpi=150)
    log(f"Saved: {RESULTS_DIR / 'delta_heatmap.png'}")

    # Figure 3: Top vs Tail comparison
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    tasks = list(all_results.keys())
    x = np.arange(len(tasks))
    width = 0.35

    top10_means = []
    tail50_means = []

    for task in tasks:
        results = all_results[task]
        top10 = np.mean([results['layers'][str(l)]['mean_delta_top10']
                        for l in range(results['n_layers'])])
        tail50 = np.mean([results['layers'][str(l)]['mean_delta_tail50']
                         for l in range(results['n_layers'])])
        top10_means.append(top10)
        tail50_means.append(tail50)

    bars1 = ax3.bar(x - width/2, top10_means, width, label='Top-10 Eigenvectors', color='coral')
    bars2 = ax3.bar(x + width/2, tail50_means, width, label='Tail-50 Eigenvectors', color='steelblue')

    ax3.set_ylabel('Mean Delta')
    ax3.set_title('RLVR Training: Top vs Tail Eigenvector Changes\n(Base → RL-Zero)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([t.upper() for t in tasks])
    ax3.legend()
    ax3.set_ylim(0, 1)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

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
    log(f"Saved: {RESULTS_DIR / 'top_vs_tail.png'}")

    plt.close('all')


def main():
    total_start = time.time()

    log("="*60)
    log("SVD REASONING SEPARABILITY ANALYSIS")
    log("="*60)
    log(f"Base model:  {BASE_MODEL}")
    log(f"RLVR model:  {RLVR_MODEL}")
    log(f"Tasks:       {TASKS}")
    log(f"Max samples: {MAX_SAMPLES}")
    log(f"Max seq len: {MAX_SEQ_LEN}")
    log(f"Top-K:       {TOP_K}")
    log("="*60)

    update_status({
        'state': 'starting',
        'tasks': TASKS,
        'progress_pct': 0
    })

    all_results = {}

    for task_idx, task in enumerate(TASKS):
        try:
            results = analyze_task(task, task_idx, len(TASKS))
            all_results[task] = results

            # Save intermediate results
            gc.collect()
        except Exception as e:
            log(f"ERROR analyzing {task}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"svd_analysis_{timestamp}.json"

    # Convert int keys to strings for JSON
    json_results = {}
    for task, res in all_results.items():
        json_results[task] = {
            'task': res['task'],
            'n_layers': res['n_layers'],
            'max_samples': res['max_samples'],
            'max_seq_len': res['max_seq_len'],
            'layers': {str(k): v for k, v in res['layers'].items()}
        }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"Saved raw results: {results_file}")

    # Generate plots
    if json_results:
        plot_results(json_results)

    # Print summary
    log("")
    log("="*60)
    log("SUMMARY")
    log("="*60)

    for task, results in json_results.items():
        top10 = np.mean([results['layers'][str(l)]['mean_delta_top10']
                        for l in range(results['n_layers'])])
        tail50 = np.mean([results['layers'][str(l)]['mean_delta_tail50']
                         for l in range(results['n_layers'])])
        ratio = top10 / tail50 if tail50 > 0 else float('inf')

        log(f"")
        log(f"{task.upper()}:")
        log(f"  Mean delta (top-10 eigenvectors): {top10:.4f}")
        log(f"  Mean delta (tail-50 eigenvectors): {tail50:.4f}")
        log(f"  Ratio (top/tail): {ratio:.2f}")

        if ratio > 1.5:
            log(f"  --> SUGGESTS SEPARABLE (top changes more than tail)")
        elif ratio < 0.67:
            log(f"  --> SUGGESTS TAIL-HEAVY (tail changes more - unexpected)")
        else:
            log(f"  --> SUGGESTS ENTANGLED (roughly equal changes)")

    total_time = time.time() - total_start
    log("")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")

    update_status({
        'state': 'completed',
        'total_time_seconds': total_time,
        'results_file': str(results_file),
        'progress_pct': 100
    })

    log("="*60)
    log("DONE")
    log("="*60)


if __name__ == "__main__":
    main()
