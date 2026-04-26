#!/usr/bin/env python3
"""
MI Pivot Token Analysis

Hypothesis: RLVR models show MI peaks at pivot tokens ("wait", "but", "hmm", etc.),
while SFT models show flat MI profiles.

This script:
1. Loads trajectories from Phase 2 0shot GSM8K data
2. Tokenizes outputs to identify pivot tokens
3. Trains linear probes to predict correctness from hidden states
4. Computes MI proxy (probe confidence) at each token position
5. Compares pivot vs non-pivot positions across models
"""

import os
import json
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import re

# Configuration
PIVOT_TOKENS = [
    "but", "wait", "hmm", "however", "actually", "?",
    "alternatively", "let me", "so", "no", "wrong", "mistake",
    "correction", "oops", "sorry", "instead", "rather", "think"
]

# Case-insensitive pattern for pivot detection
PIVOT_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in PIVOT_TOKENS) + r')\b',
    re.IGNORECASE
)

DATA_DIR = Path("/data/thanhdo/trajectories_0shot")
MODELS = ["olmo3_base", "olmo3_sft", "olmo3_rl_zero"]
MIDDLE_LAYER_IDX = 8  # Layer 16 out of 32 (even layers 0,2,...,30 -> index 8 = layer 16)

def load_trajectories(model_name: str, task: str = "gsm8k"):
    """Load trajectories and metadata from HDF5 file."""
    h5_path = DATA_DIR / model_name / f"{task}_trajectories.h5"
    print(f"Loading {h5_path}...")

    with h5py.File(h5_path, 'r') as f:
        data = {
            'trajectories': f['trajectories'][:],  # (N, 512, 16, 4096)
            'model_outputs': [x.decode() if isinstance(x, bytes) else x for x in f['model_outputs'][:]],
            'is_correct': f['is_correct'][:],
            'sequence_lengths': f['sequence_lengths'][:],
            'prompts': [x.decode() if isinstance(x, bytes) else x for x in f['prompts'][:]],
        }

    print(f"  Loaded {len(data['is_correct'])} samples, {data['is_correct'].sum()} correct ({100*data['is_correct'].mean():.1f}%)")
    return data


def find_pivot_positions(text: str, n_positions: int = 512) -> list:
    """Find positions of pivot tokens in text.

    Returns list of (position_estimate, pivot_word) tuples.
    Position is estimated as character position / avg_chars_per_token.
    """
    pivots = []
    avg_chars_per_token = 4  # Rough estimate for subword tokenization

    for match in PIVOT_PATTERN.finditer(text):
        char_pos = match.start()
        token_pos = min(char_pos // avg_chars_per_token, n_positions - 1)
        pivots.append((token_pos, match.group().lower()))

    return pivots


def compute_mi_profile(
    hidden_states: np.ndarray,  # (512, d_model)
    is_correct: bool,
    probe: LogisticRegression,
    seq_len: int
) -> np.ndarray:
    """Compute MI proxy (probe confidence) at each token position."""
    # Only look at valid positions (up to sequence length)
    valid_len = min(seq_len, hidden_states.shape[0])

    mi_profile = np.zeros(valid_len)

    for t in range(valid_len):
        h_t = hidden_states[t].reshape(1, -1).astype(np.float32)
        # MI proxy: probability assigned to correct class
        probs = probe.predict_proba(h_t)[0]
        correct_prob = probs[1] if is_correct else probs[0]
        mi_profile[t] = correct_prob

    return mi_profile


def train_correctness_probe(trajectories: np.ndarray, is_correct: np.ndarray, layer_idx: int = MIDDLE_LAYER_IDX):
    """Train a linear probe to predict correctness from hidden states.

    Uses mean-pooled hidden states across sequence positions.
    """
    print(f"  Training probe on layer {layer_idx}...")

    # Mean pool across sequence (use first 256 positions to avoid padding)
    X = trajectories[:, :256, layer_idx, :].mean(axis=1).astype(np.float32)  # (N, d_model)
    y = is_correct.astype(int)

    probe = LogisticRegression(max_iter=1000, C=0.1, random_state=42)

    # Cross-validation score
    scores = cross_val_score(probe, X, y, cv=5, scoring='accuracy')
    print(f"  Probe CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    # Train on all data for MI estimation
    probe.fit(X, y)

    return probe


def analyze_model(model_name: str, task: str = "gsm8k"):
    """Analyze MI profiles for a single model."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print('='*60)

    data = load_trajectories(model_name, task)

    # Train probe
    probe = train_correctness_probe(
        data['trajectories'],
        data['is_correct'],
        layer_idx=MIDDLE_LAYER_IDX
    )

    results = {
        'model': model_name,
        'n_samples': len(data['is_correct']),
        'accuracy': float(data['is_correct'].mean()),
        'pivot_mi': [],  # MI at pivot positions
        'nonpivot_mi': [],  # MI at non-pivot positions
        'mi_at_pivots': defaultdict(list),  # MI by pivot word
        'mi_profiles': [],  # Full MI profiles for correct samples
        'pivot_positions': [],  # Where pivots occur
    }

    print(f"\nComputing MI profiles...")

    for i in range(len(data['is_correct'])):
        output = data['model_outputs'][i]
        is_correct = data['is_correct'][i]
        seq_len = data['sequence_lengths'][i]
        hidden_states = data['trajectories'][i, :, MIDDLE_LAYER_IDX, :]  # (512, d_model)

        # Find pivots
        pivots = find_pivot_positions(output, n_positions=512)
        pivot_positions = set(p[0] for p in pivots)

        # Compute MI profile
        mi_profile = compute_mi_profile(hidden_states, is_correct, probe, seq_len)

        # Collect MI at pivot vs non-pivot positions
        for t in range(len(mi_profile)):
            if t in pivot_positions:
                results['pivot_mi'].append(mi_profile[t])
                # Also track by pivot word
                for pos, word in pivots:
                    if pos == t:
                        results['mi_at_pivots'][word].append(mi_profile[t])
            else:
                results['nonpivot_mi'].append(mi_profile[t])

        # Store for later analysis
        if is_correct:
            results['mi_profiles'].append(mi_profile)
        results['pivot_positions'].extend([p[0] for p in pivots])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(data['is_correct'])} samples")

    # Compute summary statistics
    results['pivot_mi_mean'] = float(np.mean(results['pivot_mi'])) if results['pivot_mi'] else 0
    results['pivot_mi_std'] = float(np.std(results['pivot_mi'])) if results['pivot_mi'] else 0
    results['nonpivot_mi_mean'] = float(np.mean(results['nonpivot_mi']))
    results['nonpivot_mi_std'] = float(np.std(results['nonpivot_mi']))
    results['n_pivots'] = len(results['pivot_mi'])

    # Effect size (Cohen's d)
    if results['pivot_mi'] and len(results['pivot_mi']) > 1:
        pooled_std = np.sqrt((results['pivot_mi_std']**2 + results['nonpivot_mi_std']**2) / 2)
        if pooled_std > 0:
            results['cohens_d'] = (results['pivot_mi_mean'] - results['nonpivot_mi_mean']) / pooled_std
        else:
            results['cohens_d'] = 0
    else:
        results['cohens_d'] = 0

    print(f"\nResults for {model_name}:")
    print(f"  Pivot MI:     {results['pivot_mi_mean']:.4f} +/- {results['pivot_mi_std']:.4f} (n={results['n_pivots']})")
    print(f"  Non-pivot MI: {results['nonpivot_mi_mean']:.4f} +/- {results['nonpivot_mi_std']:.4f}")
    print(f"  Cohen's d:    {results['cohens_d']:.4f}")

    # Clean up for JSON serialization
    results['pivot_mi'] = []  # Don't save raw values
    results['nonpivot_mi'] = []
    results['mi_profiles'] = []
    results['mi_at_pivots'] = {k: float(np.mean(v)) for k, v in results['mi_at_pivots'].items()}

    return results


def plot_comparison(all_results: dict, output_dir: Path):
    """Plot MI comparison across models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = list(all_results.keys())

    # Plot 1: Pivot vs Non-pivot MI by model
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.35

    pivot_means = [all_results[m]['pivot_mi_mean'] for m in models]
    pivot_stds = [all_results[m]['pivot_mi_std'] for m in models]
    nonpivot_means = [all_results[m]['nonpivot_mi_mean'] for m in models]
    nonpivot_stds = [all_results[m]['nonpivot_mi_std'] for m in models]

    ax.bar(x - width/2, pivot_means, width, yerr=pivot_stds, label='Pivot', alpha=0.8, capsize=3)
    ax.bar(x + width/2, nonpivot_means, width, yerr=nonpivot_stds, label='Non-pivot', alpha=0.8, capsize=3)

    ax.set_xlabel('Model')
    ax.set_ylabel('MI Proxy (Probe Confidence)')
    ax.set_title('MI at Pivot vs Non-Pivot Tokens')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('olmo3_', '') for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Effect sizes (Cohen's d)
    ax = axes[1]
    cohens_d = [all_results[m]['cohens_d'] for m in models]
    colors = ['green' if d > 0.2 else 'orange' if d > 0 else 'red' for d in cohens_d]

    bars = ax.bar(models, cohens_d, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')

    ax.set_xlabel('Model')
    ax.set_ylabel("Cohen's d (Pivot - Non-pivot)")
    ax.set_title('Effect Size: MI Spike at Pivot Tokens')
    ax.set_xticklabels([m.replace('olmo3_', '') for m in models], rotation=45)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Accuracy vs MI effect
    ax = axes[2]
    accuracies = [all_results[m]['accuracy'] * 100 for m in models]

    for i, m in enumerate(models):
        color = 'blue' if 'rl' in m else 'orange' if 'sft' in m else 'gray'
        ax.scatter(accuracies[i], cohens_d[i], s=100, c=color, label=m.replace('olmo3_', ''))
        ax.annotate(m.replace('olmo3_', ''), (accuracies[i], cohens_d[i]),
                   textcoords="offset points", xytext=(5,5), fontsize=9)

    ax.set_xlabel('Model Accuracy (%)')
    ax.set_ylabel("Cohen's d (Pivot MI Effect)")
    ax.set_title('Accuracy vs Pivot MI Effect')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'mi_pivot_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    plt.close()


def main():
    print("="*60)
    print("MI Pivot Token Analysis")
    print("Hypothesis: RLVR shows MI peaks at pivot tokens, SFT doesn't")
    print("="*60)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    for model in MODELS:
        results = analyze_model(model)
        all_results[model] = results

    # Save results
    results_path = output_dir / 'mi_pivot_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Plot comparison
    plot_comparison(all_results, output_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Accuracy':>10} {'Pivot MI':>10} {'Non-pivot':>10} {'Cohen d':>10}")
    print("-"*60)
    for m in MODELS:
        r = all_results[m]
        print(f"{m.replace('olmo3_', ''):<15} {r['accuracy']*100:>9.1f}% {r['pivot_mi_mean']:>10.4f} {r['nonpivot_mi_mean']:>10.4f} {r['cohens_d']:>10.4f}")

    # Hypothesis test
    print("\n" + "="*60)
    print("HYPOTHESIS TEST")
    print("="*60)

    rl_effect = all_results['olmo3_rl_zero']['cohens_d']
    sft_effect = all_results['olmo3_sft']['cohens_d']
    base_effect = all_results['olmo3_base']['cohens_d']

    print(f"RLVR pivot effect (Cohen's d): {rl_effect:.4f}")
    print(f"SFT pivot effect (Cohen's d):  {sft_effect:.4f}")
    print(f"Base pivot effect (Cohen's d): {base_effect:.4f}")

    if rl_effect > sft_effect and rl_effect > 0.2:
        print("\n[SUPPORTED] RLVR shows stronger MI peaks at pivot tokens than SFT")
    elif rl_effect > 0.2:
        print("\n[PARTIAL] RLVR shows MI peaks at pivots, but so does SFT")
    else:
        print("\n[NOT SUPPORTED] No clear MI peaks at pivot tokens for RLVR")


if __name__ == "__main__":
    main()
