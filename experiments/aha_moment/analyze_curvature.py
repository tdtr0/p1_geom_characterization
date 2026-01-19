#!/usr/bin/env python3
"""
Analyze trajectory curvature at pivot points.

Computes:
1. Menger curvature: Uses 3 consecutive points to estimate local curvature
2. Gaussian curvature proxy: Uses local PCA to estimate intrinsic curvature

Usage:
    python analyze_curvature.py \
        --input experiments/aha_moment/data/pivot_collection/pivot_trajectories.h5 \
        --output experiments/aha_moment/results/pivot_analysis/
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm


# Pivot detection patterns
PIVOT_PATTERNS = [
    (r'\bBUT WAIT\b', 'BUT_WAIT'),
    (r'\bWait,\s*(no|that|I)\b', 'Wait_correction'),
    (r'\bActually,\s*(no|that|I|let)\b', 'Actually_correction'),
    (r'\bI was wrong\b', 'I_was_wrong'),
    (r'\blet me reconsider\b', 'reconsider'),
    (r'\bno,\s*that\'s (not|wrong)\b', 'no_thats_wrong'),
    (r'\bhmm,?\s*(wait|no|that)\b', 'hmm_correction'),
    (r'\bBUT\b', 'BUT'),
    (r'\bWait\b', 'Wait'),
    (r'\bactually\b', 'actually'),
    (r'\bhowever\b', 'however'),
]


def detect_pivots(text: str) -> list[dict]:
    """Detect pivot positions in generated text."""
    pivots = []
    for pattern, label in PIVOT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            pivots.append({
                'char_pos': match.start(),
                'pattern': label,
                'match': match.group(0),
                'context': text[start:end],
            })

    # Remove duplicates
    pivots.sort(key=lambda x: x['char_pos'])
    seen = set()
    unique = []
    for p in pivots:
        if p['char_pos'] not in seen:
            unique.append(p)
            seen.add(p['char_pos'])
    return unique


def char_to_token_position(char_pos: int, text: str, n_tokens: int) -> int:
    """Map character position to token position."""
    chars_per_token = len(text) / max(n_tokens, 1)
    return min(int(char_pos / chars_per_token), n_tokens - 1)


def compute_menger_curvature(trajectory: np.ndarray, window: int = 1) -> np.ndarray:
    """
    Compute Menger curvature at each point.

    Menger curvature uses 3 points (p1, p2, p3) and computes:
    κ = 4 * area(triangle) / (|p1-p2| * |p2-p3| * |p3-p1|)

    This is the curvature of the circumscribed circle.

    Args:
        trajectory: (n_tokens, n_layers, hidden_dim) or (n_tokens, hidden_dim)
        window: Distance between points (default 1 = consecutive)

    Returns:
        (n_tokens,) curvature values
    """
    # Average across layers if needed
    if trajectory.ndim == 3:
        traj = trajectory.mean(axis=1)  # (n_tokens, hidden_dim)
    else:
        traj = trajectory

    n_tokens = traj.shape[0]
    curvature = np.zeros(n_tokens)

    for i in range(window, n_tokens - window):
        p1 = traj[i - window]
        p2 = traj[i]
        p3 = traj[i + window]

        # Side lengths
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        # Avoid division by zero
        if a < 1e-8 or b < 1e-8 or c < 1e-8:
            curvature[i] = 0
            continue

        # Semi-perimeter
        s = (a + b + c) / 2

        # Area by Heron's formula
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq < 0:
            area_sq = 0
        area = np.sqrt(area_sq)

        # Menger curvature
        curvature[i] = 4 * area / (a * b * c)

    return curvature


def compute_gaussian_curvature_proxy(trajectory: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute a proxy for Gaussian curvature using local PCA.

    For a local neighborhood, we compute PCA and use the ratio of eigenvalues
    to estimate how "curved" the local manifold is.

    High ratio (λ1 >> λ2) = trajectory is nearly linear (low curvature)
    Low ratio (λ1 ≈ λ2) = trajectory spans multiple directions (high curvature)

    We return 1 - (λ1 / sum(λ)) as curvature proxy.

    Args:
        trajectory: (n_tokens, n_layers, hidden_dim) or (n_tokens, hidden_dim)
        window: Size of local neighborhood

    Returns:
        (n_tokens,) curvature proxy values
    """
    if trajectory.ndim == 3:
        traj = trajectory.mean(axis=1)
    else:
        traj = trajectory

    n_tokens = traj.shape[0]
    curvature = np.zeros(n_tokens)

    for i in range(window, n_tokens - window):
        # Get local neighborhood
        local = traj[i - window:i + window + 1]

        # Center the data
        local = local - local.mean(axis=0)

        # SVD to get eigenvalues
        try:
            _, s, _ = np.linalg.svd(local, full_matrices=False)

            # Curvature proxy: how much variance is NOT in the first component
            total_var = np.sum(s ** 2)
            if total_var > 1e-8:
                curvature[i] = 1 - (s[0] ** 2 / total_var)
            else:
                curvature[i] = 0
        except:
            curvature[i] = 0

    return curvature


def analyze_sample(trajectory: np.ndarray, text: str, seq_len: int) -> dict:
    """Analyze curvature at pivot vs random positions."""
    traj = trajectory[:seq_len].astype(np.float32)

    # Detect pivots
    pivots = detect_pivots(text)
    for p in pivots:
        p['token_pos'] = char_to_token_position(p['char_pos'], text, seq_len)

    # Filter valid pivots
    valid_pivots = [p for p in pivots if 10 <= p['token_pos'] < seq_len - 10]

    if not valid_pivots:
        return None

    # Compute curvatures
    menger = compute_menger_curvature(traj)
    gaussian = compute_gaussian_curvature_proxy(traj)

    # Get pivot positions
    pivot_positions = [p['token_pos'] for p in valid_pivots]

    # Get random positions
    all_positions = set(range(15, seq_len - 15))
    pivot_set = set(pivot_positions)
    for pos in pivot_positions:
        for offset in range(-5, 6):
            pivot_set.add(pos + offset)
    available = list(all_positions - pivot_set)

    if len(available) < len(pivot_positions):
        return None

    np.random.shuffle(available)
    random_positions = available[:len(pivot_positions)]

    return {
        'n_pivots': len(valid_pivots),
        'pivot_positions': pivot_positions,
        'pivot_patterns': [p['pattern'] for p in valid_pivots],
        'pivot_contexts': [p['context'] for p in valid_pivots],
        'pivot_menger': [menger[p] for p in pivot_positions],
        'pivot_gaussian': [gaussian[p] for p in pivot_positions],
        'random_menger': [menger[p] for p in random_positions],
        'random_gaussian': [gaussian[p] for p in random_positions],
        'seq_len': seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze curvature at pivots")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--examples", type=int, default=5, help="Number of examples to save")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {args.input}")
    with h5py.File(args.input, 'r') as f:
        trajectories = f['trajectories'][:]
        seq_lengths = f['sequence_lengths'][:]
        texts = [t.decode('utf-8') if isinstance(t, bytes) else t
                 for t in f['generated_texts'][:]]
        model = f.attrs.get('model', 'unknown')

    print(f"Model: {model}")
    print(f"Samples: {len(trajectories)}")

    # Analyze
    print("\nAnalyzing curvature...")
    results = []
    examples = []

    for i in tqdm(range(len(trajectories)), desc="Analyzing"):
        result = analyze_sample(trajectories[i], texts[i], seq_lengths[i])
        if result is not None:
            result['sample_idx'] = i
            result['text'] = texts[i][:500]  # First 500 chars
            results.append(result)

            # Save examples with pivots
            if len(examples) < args.examples and result['n_pivots'] > 0:
                examples.append({
                    'sample_idx': i,
                    'text': texts[i],
                    'pivots': [{
                        'pattern': result['pivot_patterns'][j],
                        'context': result['pivot_contexts'][j],
                        'menger_curvature': float(result['pivot_menger'][j]),
                        'gaussian_curvature': float(result['pivot_gaussian'][j]),
                    } for j in range(len(result['pivot_patterns']))],
                })

    print(f"\nSamples with pivots: {len(results)}/{len(trajectories)}")

    if not results:
        print("No pivots detected!")
        return

    # Aggregate
    all_pivot_menger = []
    all_random_menger = []
    all_pivot_gaussian = []
    all_random_gaussian = []

    for r in results:
        all_pivot_menger.extend(r['pivot_menger'])
        all_random_menger.extend(r['random_menger'])
        all_pivot_gaussian.extend(r['pivot_gaussian'])
        all_random_gaussian.extend(r['random_gaussian'])

    # Statistics
    print("\n" + "=" * 60)
    print("CURVATURE RESULTS")
    print("=" * 60)

    # Menger curvature
    t_menger, p_menger = stats.ttest_ind(all_pivot_menger, all_random_menger)
    d_menger = (np.mean(all_pivot_menger) - np.mean(all_random_menger)) / \
               np.sqrt((np.std(all_pivot_menger)**2 + np.std(all_random_menger)**2) / 2)

    print(f"\nMenger Curvature:")
    print(f"  Pivot mean: {np.mean(all_pivot_menger):.6f} (std: {np.std(all_pivot_menger):.6f})")
    print(f"  Random mean: {np.mean(all_random_menger):.6f} (std: {np.std(all_random_menger):.6f})")
    print(f"  Effect size (d): {d_menger:.3f}")
    print(f"  t-statistic: {t_menger:.3f}, p-value: {p_menger:.4f}")

    # Gaussian curvature proxy
    t_gauss, p_gauss = stats.ttest_ind(all_pivot_gaussian, all_random_gaussian)
    d_gauss = (np.mean(all_pivot_gaussian) - np.mean(all_random_gaussian)) / \
              np.sqrt((np.std(all_pivot_gaussian)**2 + np.std(all_random_gaussian)**2) / 2)

    print(f"\nGaussian Curvature Proxy:")
    print(f"  Pivot mean: {np.mean(all_pivot_gaussian):.6f} (std: {np.std(all_pivot_gaussian):.6f})")
    print(f"  Random mean: {np.mean(all_random_gaussian):.6f} (std: {np.std(all_random_gaussian):.6f})")
    print(f"  Effect size (d): {d_gauss:.3f}")
    print(f"  t-statistic: {t_gauss:.3f}, p-value: {p_gauss:.4f}")

    # Save results
    summary = {
        'model': model,
        'n_samples': len(trajectories),
        'n_samples_with_pivots': len(results),
        'total_pivots': len(all_pivot_menger),
        'menger_curvature': {
            'pivot_mean': float(np.mean(all_pivot_menger)),
            'pivot_std': float(np.std(all_pivot_menger)),
            'random_mean': float(np.mean(all_random_menger)),
            'random_std': float(np.std(all_random_menger)),
            'effect_size_d': float(d_menger),
            'p_value': float(p_menger),
        },
        'gaussian_curvature_proxy': {
            'pivot_mean': float(np.mean(all_pivot_gaussian)),
            'pivot_std': float(np.std(all_pivot_gaussian)),
            'random_mean': float(np.mean(all_random_gaussian)),
            'random_std': float(np.std(all_random_gaussian)),
            'effect_size_d': float(d_gauss),
            'p_value': float(p_gauss),
        },
    }

    with open(output_dir / "curvature_analysis.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save examples
    with open(output_dir / "pivot_examples.json", 'w') as f:
        json.dump(examples, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(all_random_menger, bins=30, alpha=0.5, label='Random', density=True)
    axes[0].hist(all_pivot_menger, bins=30, alpha=0.5, label='Pivot', density=True)
    axes[0].set_xlabel('Menger Curvature')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Menger Curvature\n(d={d_menger:.2f}, p={p_menger:.4f})')
    axes[0].legend()

    axes[1].hist(all_random_gaussian, bins=30, alpha=0.5, label='Random', density=True)
    axes[1].hist(all_pivot_gaussian, bins=30, alpha=0.5, label='Pivot', density=True)
    axes[1].set_xlabel('Gaussian Curvature Proxy')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Gaussian Curvature Proxy\n(d={d_gauss:.2f}, p={p_gauss:.4f})')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "curvature_distributions.png", dpi=150)
    plt.close()

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
