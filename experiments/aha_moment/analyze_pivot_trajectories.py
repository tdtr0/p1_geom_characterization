#!/usr/bin/env python3
"""
Analyze trajectory dynamics at pivot points.

This script:
1. Loads generation trajectories from collect_pivot_trajectories.py
2. Detects pivot points in generated text (self-correction moments)
3. Computes trajectory metrics (velocity, direction change, Lyapunov)
4. Compares pivot vs random token dynamics

Usage:
    python analyze_pivot_trajectories.py \
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
STRONG_PIVOT_PATTERNS = [
    (r'\bBUT WAIT\b', 'BUT_WAIT'),
    (r'\bWait,\s*(no|that|I)\b', 'Wait_correction'),
    (r'\bActually,\s*(no|that|I|let)\b', 'Actually_correction'),
    (r'\bI was wrong\b', 'I_was_wrong'),
    (r'\blet me reconsider\b', 'reconsider'),
    (r'\bno,\s*that\'s (not|wrong)\b', 'no_thats_wrong'),
    (r'\bhmm,?\s*(wait|no|that)\b', 'hmm_correction'),
]

WEAK_PIVOT_PATTERNS = [
    (r'\bBUT\b', 'BUT'),
    (r'\bWait\b', 'Wait'),
    (r'\bactually\b', 'actually'),
    (r'\bhowever\b', 'however'),
    (r'\bhmm\b', 'hmm'),
]


def detect_pivots(text: str, use_strong_only: bool = True) -> list[dict]:
    """
    Detect pivot positions in generated text.

    Returns list of dicts with:
        - char_pos: Character position of pivot
        - token_pos: Will be filled in later
        - pattern: Which pattern matched
        - context: Surrounding text
    """
    patterns = STRONG_PIVOT_PATTERNS if use_strong_only else STRONG_PIVOT_PATTERNS + WEAK_PIVOT_PATTERNS
    pivots = []

    for pattern, label in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Get context
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]

            pivots.append({
                'char_pos': match.start(),
                'pattern': label,
                'match': match.group(0),
                'context': context,
            })

    # Sort by position and remove duplicates (keep first pattern match)
    pivots.sort(key=lambda x: x['char_pos'])
    seen_positions = set()
    unique_pivots = []
    for p in pivots:
        if p['char_pos'] not in seen_positions:
            unique_pivots.append(p)
            seen_positions.add(p['char_pos'])

    return unique_pivots


def char_to_token_position(char_pos: int, text: str, n_tokens: int) -> int:
    """
    Approximate mapping from character position to token position.

    Uses simple heuristic: chars_per_token ~= 4 for English text.
    """
    # Estimate: average 4 chars per token
    chars_per_token = len(text) / max(n_tokens, 1)
    token_pos = int(char_pos / chars_per_token)
    return min(token_pos, n_tokens - 1)


def compute_velocity(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute velocity magnitude at each token position.

    trajectory: (n_tokens, n_layers, hidden_dim)
    Returns: (n_tokens,) velocity magnitudes
    """
    # Average across layers
    avg_traj = trajectory.mean(axis=1)  # (n_tokens, hidden_dim)

    # Compute differences
    diffs = np.diff(avg_traj, axis=0)  # (n_tokens-1, hidden_dim)

    # Velocity magnitude
    velocities = np.linalg.norm(diffs, axis=1)  # (n_tokens-1,)

    # Pad to match original length
    velocities = np.concatenate([[velocities[0]], velocities])

    return velocities


def compute_direction_change(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute direction change (curvature) at each token position.

    Uses cosine similarity between consecutive velocity vectors.
    Returns: (n_tokens,) direction change values (0 = straight, 1 = 90 deg, 2 = reversal)
    """
    # Average across layers
    avg_traj = trajectory.mean(axis=1)  # (n_tokens, hidden_dim)

    # Velocity vectors
    v = np.diff(avg_traj, axis=0)  # (n_tokens-1, hidden_dim)

    if len(v) < 2:
        return np.zeros(trajectory.shape[0])

    # Normalize
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v_norm = v / (norms + 1e-8)

    # Cosine similarity between consecutive velocities
    cos_sim = np.sum(v_norm[:-1] * v_norm[1:], axis=1)  # (n_tokens-2,)

    # Direction change = 1 - cos_sim (0 = same direction, 2 = opposite)
    dir_change = 1 - cos_sim

    # Pad to match original length
    dir_change = np.concatenate([[0], [0], dir_change])

    return dir_change


def compute_local_lyapunov(trajectory: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute local Lyapunov exponent estimate at each position.

    Uses SVD of local trajectory segment to estimate expansion rate.
    Returns: (n_tokens,) Lyapunov estimates
    """
    n_tokens = trajectory.shape[0]
    lyapunov = np.zeros(n_tokens)

    # Average across layers
    avg_traj = trajectory.mean(axis=1).astype(np.float32)  # (n_tokens, hidden_dim)

    for i in range(window, n_tokens - window):
        # Get local segment
        segment = avg_traj[i-window:i+window+1]  # (2*window+1, hidden_dim)

        # Center the segment
        segment = segment - segment.mean(axis=0)

        # SVD
        try:
            _, s, _ = np.linalg.svd(segment, full_matrices=False)
            # Lyapunov estimate: log of ratio of largest to smallest singular value
            if s[-1] > 1e-8:
                lyapunov[i] = np.log(s[0] / s[-1]) / (2 * window)
            else:
                lyapunov[i] = np.nan
        except Exception:
            lyapunov[i] = np.nan

    return lyapunov


def analyze_sample(
    trajectory: np.ndarray,
    text: str,
    seq_len: int,
) -> dict:
    """
    Analyze a single sample's trajectory at pivot vs random positions.
    """
    # Only use actual sequence (not padding)
    traj = trajectory[:seq_len].astype(np.float32)

    # Detect pivots
    pivots = detect_pivots(text, use_strong_only=False)

    # Convert to token positions
    for p in pivots:
        p['token_pos'] = char_to_token_position(p['char_pos'], text, seq_len)

    # Filter pivots within valid range (skip first/last 5 tokens)
    valid_pivots = [p for p in pivots if 5 <= p['token_pos'] < seq_len - 5]

    if not valid_pivots:
        return None

    # Compute metrics
    velocity = compute_velocity(traj)
    dir_change = compute_direction_change(traj)
    lyapunov = compute_local_lyapunov(traj)

    # Get pivot positions
    pivot_positions = [p['token_pos'] for p in valid_pivots]

    # Get random positions (same count, avoiding pivots)
    all_positions = set(range(10, seq_len - 10))
    pivot_set = set(pivot_positions)
    # Also exclude neighbors of pivots
    for pos in pivot_positions:
        for offset in range(-3, 4):
            pivot_set.add(pos + offset)
    available = list(all_positions - pivot_set)

    if len(available) < len(pivot_positions):
        return None

    np.random.shuffle(available)
    random_positions = available[:len(pivot_positions)]

    # Collect metrics at pivot vs random
    pivot_metrics = {
        'velocity': [velocity[p] for p in pivot_positions],
        'dir_change': [dir_change[p] for p in pivot_positions],
        'lyapunov': [lyapunov[p] for p in pivot_positions if not np.isnan(lyapunov[p])],
    }

    random_metrics = {
        'velocity': [velocity[p] for p in random_positions],
        'dir_change': [dir_change[p] for p in random_positions],
        'lyapunov': [lyapunov[p] for p in random_positions if not np.isnan(lyapunov[p])],
    }

    return {
        'n_pivots': len(valid_pivots),
        'pivot_positions': pivot_positions,
        'pivot_patterns': [p['pattern'] for p in valid_pivots],
        'pivot_metrics': pivot_metrics,
        'random_metrics': random_metrics,
        'seq_len': seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze pivot trajectories")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    # Create output directory
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
        print(f"Trajectory shape: {trajectories.shape}")

    # Analyze each sample
    print("\nAnalyzing samples...")
    results = []

    for i in tqdm(range(len(trajectories)), desc="Analyzing"):
        result = analyze_sample(
            trajectories[i],
            texts[i],
            seq_lengths[i],
        )
        if result is not None:
            result['sample_idx'] = i
            results.append(result)

    print(f"\nSamples with pivots: {len(results)}/{len(trajectories)}")

    if not results:
        print("No pivots detected! Try with --use_weak_patterns")
        return

    # Aggregate results
    all_pivot_velocity = []
    all_random_velocity = []
    all_pivot_dir = []
    all_random_dir = []
    all_pivot_lyap = []
    all_random_lyap = []

    for r in results:
        all_pivot_velocity.extend(r['pivot_metrics']['velocity'])
        all_random_velocity.extend(r['random_metrics']['velocity'])
        all_pivot_dir.extend(r['pivot_metrics']['dir_change'])
        all_random_dir.extend(r['random_metrics']['dir_change'])
        all_pivot_lyap.extend(r['pivot_metrics']['lyapunov'])
        all_random_lyap.extend(r['random_metrics']['lyapunov'])

    # Statistical tests
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Velocity
    if all_pivot_velocity and all_random_velocity:
        t_vel, p_vel = stats.ttest_ind(all_pivot_velocity, all_random_velocity)
        d_vel = (np.mean(all_pivot_velocity) - np.mean(all_random_velocity)) / \
                np.sqrt((np.std(all_pivot_velocity)**2 + np.std(all_random_velocity)**2) / 2)
        print(f"\nVelocity:")
        print(f"  Pivot mean: {np.mean(all_pivot_velocity):.3f} (std: {np.std(all_pivot_velocity):.3f})")
        print(f"  Random mean: {np.mean(all_random_velocity):.3f} (std: {np.std(all_random_velocity):.3f})")
        print(f"  Effect size (d): {d_vel:.3f}")
        print(f"  t-statistic: {t_vel:.3f}, p-value: {p_vel:.4f}")

    # Direction change
    if all_pivot_dir and all_random_dir:
        t_dir, p_dir = stats.ttest_ind(all_pivot_dir, all_random_dir)
        d_dir = (np.mean(all_pivot_dir) - np.mean(all_random_dir)) / \
                np.sqrt((np.std(all_pivot_dir)**2 + np.std(all_random_dir)**2) / 2)
        print(f"\nDirection Change:")
        print(f"  Pivot mean: {np.mean(all_pivot_dir):.3f} (std: {np.std(all_pivot_dir):.3f})")
        print(f"  Random mean: {np.mean(all_random_dir):.3f} (std: {np.std(all_random_dir):.3f})")
        print(f"  Effect size (d): {d_dir:.3f}")
        print(f"  t-statistic: {t_dir:.3f}, p-value: {p_dir:.4f}")

    # Lyapunov
    if all_pivot_lyap and all_random_lyap:
        t_lyap, p_lyap = stats.ttest_ind(all_pivot_lyap, all_random_lyap)
        d_lyap = (np.mean(all_pivot_lyap) - np.mean(all_random_lyap)) / \
                 np.sqrt((np.std(all_pivot_lyap)**2 + np.std(all_random_lyap)**2) / 2)
        print(f"\nLocal Lyapunov:")
        print(f"  Pivot mean: {np.mean(all_pivot_lyap):.3f} (std: {np.std(all_pivot_lyap):.3f})")
        print(f"  Random mean: {np.mean(all_random_lyap):.3f} (std: {np.std(all_random_lyap):.3f})")
        print(f"  Effect size (d): {d_lyap:.3f}")
        print(f"  t-statistic: {t_lyap:.3f}, p-value: {p_lyap:.4f}")

    # Save results
    summary = {
        'model': model,
        'n_samples': len(trajectories),
        'n_samples_with_pivots': len(results),
        'total_pivots': len(all_pivot_velocity),
        'metrics': {
            'velocity': {
                'pivot_mean': float(np.mean(all_pivot_velocity)) if all_pivot_velocity else None,
                'pivot_std': float(np.std(all_pivot_velocity)) if all_pivot_velocity else None,
                'random_mean': float(np.mean(all_random_velocity)) if all_random_velocity else None,
                'random_std': float(np.std(all_random_velocity)) if all_random_velocity else None,
                'effect_size_d': float(d_vel) if all_pivot_velocity else None,
                'p_value': float(p_vel) if all_pivot_velocity else None,
            },
            'direction_change': {
                'pivot_mean': float(np.mean(all_pivot_dir)) if all_pivot_dir else None,
                'pivot_std': float(np.std(all_pivot_dir)) if all_pivot_dir else None,
                'random_mean': float(np.mean(all_random_dir)) if all_random_dir else None,
                'random_std': float(np.std(all_random_dir)) if all_random_dir else None,
                'effect_size_d': float(d_dir) if all_pivot_dir else None,
                'p_value': float(p_dir) if all_pivot_dir else None,
            },
            'lyapunov': {
                'pivot_mean': float(np.mean(all_pivot_lyap)) if all_pivot_lyap else None,
                'pivot_std': float(np.std(all_pivot_lyap)) if all_pivot_lyap else None,
                'random_mean': float(np.mean(all_random_lyap)) if all_random_lyap else None,
                'random_std': float(np.std(all_random_lyap)) if all_random_lyap else None,
                'effect_size_d': float(d_lyap) if all_pivot_lyap else None,
                'p_value': float(p_lyap) if all_pivot_lyap else None,
            },
        },
        'pivot_patterns': {},
    }

    # Count pivot patterns
    for r in results:
        for pattern in r['pivot_patterns']:
            summary['pivot_patterns'][pattern] = summary['pivot_patterns'].get(pattern, 0) + 1

    # Save summary
    with open(output_dir / "pivot_analysis.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Velocity distribution
    axes[0].hist(all_random_velocity, bins=30, alpha=0.5, label='Random', density=True)
    axes[0].hist(all_pivot_velocity, bins=30, alpha=0.5, label='Pivot', density=True)
    axes[0].set_xlabel('Velocity Magnitude')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Velocity at Pivot vs Random\n(d={d_vel:.2f}, p={p_vel:.4f})')
    axes[0].legend()

    # Direction change distribution
    axes[1].hist(all_random_dir, bins=30, alpha=0.5, label='Random', density=True)
    axes[1].hist(all_pivot_dir, bins=30, alpha=0.5, label='Pivot', density=True)
    axes[1].set_xlabel('Direction Change')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Direction Change at Pivot vs Random\n(d={d_dir:.2f}, p={p_dir:.4f})')
    axes[1].legend()

    # Lyapunov distribution
    if all_pivot_lyap and all_random_lyap:
        axes[2].hist(all_random_lyap, bins=30, alpha=0.5, label='Random', density=True)
        axes[2].hist(all_pivot_lyap, bins=30, alpha=0.5, label='Pivot', density=True)
        axes[2].set_xlabel('Local Lyapunov Exponent')
        axes[2].set_ylabel('Density')
        axes[2].set_title(f'Lyapunov at Pivot vs Random\n(d={d_lyap:.2f}, p={p_lyap:.4f})')
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "pivot_distributions.png", dpi=150)
    plt.close()

    print(f"\nResults saved to: {output_dir}")
    print(f"  - pivot_analysis.json")
    print(f"  - pivot_distributions.png")


if __name__ == "__main__":
    main()
