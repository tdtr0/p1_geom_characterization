#!/usr/bin/env python3
"""
MI Pivot Token Analysis v2

Refined approach that controls for baseline probe accuracy by looking at:
1. Within-sample MI changes at pivot positions (delta from local average)
2. MI trajectory slope leading up to pivots
3. Separate analysis for correct vs incorrect samples
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
from scipy import stats

# Configuration
PIVOT_TOKENS = [
    "but", "wait", "hmm", "however", "actually", "?",
    "alternatively", "let me", "so", "no", "wrong", "mistake",
    "correction", "oops", "sorry", "instead", "rather", "think"
]

PIVOT_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in PIVOT_TOKENS) + r')\b',
    re.IGNORECASE
)

DATA_DIR = Path("/data/thanhdo/trajectories_0shot")
MODELS = ["olmo3_base", "olmo3_sft", "olmo3_rl_zero"]
MIDDLE_LAYER_IDX = 8  # Layer 16 out of 32

def load_trajectories(model_name: str, task: str = "gsm8k"):
    """Load trajectories and metadata from HDF5 file."""
    h5_path = DATA_DIR / model_name / f"{task}_trajectories.h5"
    print(f"Loading {h5_path}...")

    with h5py.File(h5_path, 'r') as f:
        data = {
            'trajectories': f['trajectories'][:],
            'model_outputs': [x.decode() if isinstance(x, bytes) else x for x in f['model_outputs'][:]],
            'is_correct': f['is_correct'][:],
            'sequence_lengths': f['sequence_lengths'][:],
            'prompts': [x.decode() if isinstance(x, bytes) else x for x in f['prompts'][:]],
        }

    print(f"  Loaded {len(data['is_correct'])} samples, {data['is_correct'].sum()} correct ({100*data['is_correct'].mean():.1f}%)")
    return data


def find_pivot_positions(text: str, n_positions: int = 512) -> list:
    """Find positions of pivot tokens in text."""
    pivots = []
    avg_chars_per_token = 4

    for match in PIVOT_PATTERN.finditer(text):
        char_pos = match.start()
        token_pos = min(char_pos // avg_chars_per_token, n_positions - 1)
        pivots.append((token_pos, match.group().lower()))

    return pivots


def compute_mi_profile_normalized(
    hidden_states: np.ndarray,
    probe: LogisticRegression,
    seq_len: int
) -> np.ndarray:
    """Compute MI proxy (probe probability for positive class) at each position."""
    valid_len = min(seq_len, hidden_states.shape[0])
    mi_profile = np.zeros(valid_len)

    for t in range(valid_len):
        h_t = hidden_states[t].reshape(1, -1).astype(np.float32)
        probs = probe.predict_proba(h_t)[0]
        mi_profile[t] = probs[1]  # P(correct)

    return mi_profile


def compute_pivot_deltas(mi_profile: np.ndarray, pivot_positions: list, window: int = 10):
    """
    Compute MI delta at pivot positions relative to local context.

    Returns:
        deltas: MI at pivot minus average MI in surrounding window
        slopes: Slope of MI leading up to pivot
    """
    deltas = []
    slopes = []
    post_pivot_changes = []

    for pos, _ in pivot_positions:
        if pos < window or pos >= len(mi_profile) - window:
            continue

        # Local context
        pre_window = mi_profile[max(0, pos-window):pos]
        post_window = mi_profile[pos+1:min(len(mi_profile), pos+window+1)]

        if len(pre_window) < 3 or len(post_window) < 3:
            continue

        # Delta: MI at pivot vs surrounding average
        local_avg = (pre_window.mean() + post_window.mean()) / 2
        delta = mi_profile[pos] - local_avg
        deltas.append(delta)

        # Slope leading up to pivot (increasing = building confidence)
        slope = np.polyfit(range(len(pre_window)), pre_window, 1)[0]
        slopes.append(slope)

        # Post-pivot change
        post_change = post_window.mean() - mi_profile[pos]
        post_pivot_changes.append(post_change)

    return deltas, slopes, post_pivot_changes


def train_correctness_probe(trajectories: np.ndarray, is_correct: np.ndarray, layer_idx: int = MIDDLE_LAYER_IDX):
    """Train linear probe on mean-pooled hidden states."""
    print(f"  Training probe on layer {layer_idx}...")

    X = trajectories[:, :256, layer_idx, :].mean(axis=1).astype(np.float32)
    y = is_correct.astype(int)

    probe = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    scores = cross_val_score(probe, X, y, cv=5, scoring='accuracy')
    print(f"  Probe CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    probe.fit(X, y)
    return probe, scores.mean()


def analyze_model_v2(model_name: str, task: str = "gsm8k"):
    """Analyze with delta-based MI changes at pivots."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name} (v2 - delta-based)")
    print('='*60)

    data = load_trajectories(model_name, task)
    probe, probe_acc = train_correctness_probe(
        data['trajectories'],
        data['is_correct'],
        layer_idx=MIDDLE_LAYER_IDX
    )

    results = {
        'model': model_name,
        'n_samples': len(data['is_correct']),
        'accuracy': float(data['is_correct'].mean()),
        'probe_accuracy': float(probe_acc),
        # Correct samples
        'correct_pivot_deltas': [],
        'correct_pivot_slopes': [],
        'correct_post_changes': [],
        # Incorrect samples
        'incorrect_pivot_deltas': [],
        'incorrect_pivot_slopes': [],
        'incorrect_post_changes': [],
        # All samples
        'all_pivot_deltas': [],
        'all_pivot_slopes': [],
    }

    print(f"\nComputing MI profiles and pivot deltas...")

    for i in range(len(data['is_correct'])):
        output = data['model_outputs'][i]
        is_correct = data['is_correct'][i]
        seq_len = data['sequence_lengths'][i]
        hidden_states = data['trajectories'][i, :, MIDDLE_LAYER_IDX, :]

        pivots = find_pivot_positions(output, n_positions=512)
        if not pivots:
            continue

        mi_profile = compute_mi_profile_normalized(hidden_states, probe, seq_len)
        deltas, slopes, post_changes = compute_pivot_deltas(mi_profile, pivots)

        results['all_pivot_deltas'].extend(deltas)
        results['all_pivot_slopes'].extend(slopes)

        if is_correct:
            results['correct_pivot_deltas'].extend(deltas)
            results['correct_pivot_slopes'].extend(slopes)
            results['correct_post_changes'].extend(post_changes)
        else:
            results['incorrect_pivot_deltas'].extend(deltas)
            results['incorrect_pivot_slopes'].extend(slopes)
            results['incorrect_post_changes'].extend(post_changes)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(data['is_correct'])} samples")

    # Compute summary statistics
    def summarize(arr, name):
        if len(arr) == 0:
            return {'mean': 0, 'std': 0, 'n': 0}
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'n': len(arr)
        }

    results['correct_delta_stats'] = summarize(results['correct_pivot_deltas'], 'correct_deltas')
    results['incorrect_delta_stats'] = summarize(results['incorrect_pivot_deltas'], 'incorrect_deltas')
    results['all_delta_stats'] = summarize(results['all_pivot_deltas'], 'all_deltas')

    results['correct_slope_stats'] = summarize(results['correct_pivot_slopes'], 'correct_slopes')
    results['incorrect_slope_stats'] = summarize(results['incorrect_pivot_slopes'], 'incorrect_slopes')
    results['all_slope_stats'] = summarize(results['all_pivot_slopes'], 'all_slopes')

    # Statistical test: Do correct samples have different pivot deltas than incorrect?
    if len(results['correct_pivot_deltas']) > 5 and len(results['incorrect_pivot_deltas']) > 5:
        t_stat, p_val = stats.ttest_ind(
            results['correct_pivot_deltas'],
            results['incorrect_pivot_deltas']
        )
        results['delta_ttest'] = {'t_stat': float(t_stat), 'p_value': float(p_val)}
    else:
        results['delta_ttest'] = {'t_stat': 0, 'p_value': 1.0}

    # Effect size for pivot deltas
    if results['all_delta_stats']['std'] > 0:
        results['delta_effect_size'] = results['all_delta_stats']['mean'] / results['all_delta_stats']['std']
    else:
        results['delta_effect_size'] = 0

    print(f"\nResults for {model_name}:")
    print(f"  Probe accuracy: {results['probe_accuracy']:.3f}")
    print(f"  All pivot deltas: {results['all_delta_stats']['mean']:.4f} +/- {results['all_delta_stats']['std']:.4f} (n={results['all_delta_stats']['n']})")
    print(f"  Correct deltas:   {results['correct_delta_stats']['mean']:.4f} +/- {results['correct_delta_stats']['std']:.4f} (n={results['correct_delta_stats']['n']})")
    print(f"  Incorrect deltas: {results['incorrect_delta_stats']['mean']:.4f} +/- {results['incorrect_delta_stats']['std']:.4f} (n={results['incorrect_delta_stats']['n']})")
    print(f"  Delta effect size: {results['delta_effect_size']:.4f}")
    if results['delta_ttest']['p_value'] < 0.05:
        print(f"  ** Correct vs Incorrect significant: t={results['delta_ttest']['t_stat']:.2f}, p={results['delta_ttest']['p_value']:.4f}")

    # Clean up for JSON
    for key in ['correct_pivot_deltas', 'incorrect_pivot_deltas', 'all_pivot_deltas',
                'correct_pivot_slopes', 'incorrect_pivot_slopes', 'all_pivot_slopes',
                'correct_post_changes', 'incorrect_post_changes']:
        results[key] = []

    return results


def plot_comparison_v2(all_results: dict, output_dir: Path):
    """Plot delta-based comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = list(all_results.keys())

    # Plot 1: Pivot delta by correctness
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.35

    correct_means = [all_results[m]['correct_delta_stats']['mean'] for m in models]
    correct_stds = [all_results[m]['correct_delta_stats']['std'] for m in models]
    incorrect_means = [all_results[m]['incorrect_delta_stats']['mean'] for m in models]
    incorrect_stds = [all_results[m]['incorrect_delta_stats']['std'] for m in models]

    ax.bar(x - width/2, correct_means, width, yerr=correct_stds, label='Correct', alpha=0.8, color='green', capsize=3)
    ax.bar(x + width/2, incorrect_means, width, yerr=incorrect_stds, label='Incorrect', alpha=0.8, color='red', capsize=3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Model')
    ax.set_ylabel('MI Delta at Pivot (vs local avg)')
    ax.set_title('Pivot Token MI Delta by Correctness')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('olmo3_', '') for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Overall delta effect by model
    ax = axes[1]
    all_deltas = [all_results[m]['all_delta_stats']['mean'] for m in models]
    all_stds = [all_results[m]['all_delta_stats']['std'] for m in models]

    colors = ['blue' if 'rl' in m else 'orange' if 'sft' in m else 'gray' for m in models]
    bars = ax.bar(models, all_deltas, yerr=all_stds, color=colors, alpha=0.7, capsize=3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Model')
    ax.set_ylabel('MI Delta at Pivot')
    ax.set_title('Overall Pivot MI Delta')
    ax.set_xticklabels([m.replace('olmo3_', '') for m in models], rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Probe accuracy vs delta effect
    ax = axes[2]
    probe_accs = [all_results[m]['probe_accuracy'] * 100 for m in models]
    delta_effects = [all_results[m]['delta_effect_size'] for m in models]

    for i, m in enumerate(models):
        color = 'blue' if 'rl' in m else 'orange' if 'sft' in m else 'gray'
        ax.scatter(probe_accs[i], delta_effects[i], s=100, c=color, label=m.replace('olmo3_', ''))
        ax.annotate(m.replace('olmo3_', ''), (probe_accs[i], delta_effects[i]),
                   textcoords="offset points", xytext=(5,5), fontsize=9)

    ax.set_xlabel('Probe Accuracy (%)')
    ax.set_ylabel('Delta Effect Size (mean/std)')
    ax.set_title('Probe Accuracy vs Pivot Effect')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'mi_pivot_comparison_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    plt.close()


def main():
    print("="*60)
    print("MI Pivot Token Analysis v2 (Delta-based)")
    print("Controls for baseline by measuring local MI changes at pivots")
    print("="*60)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    for model in MODELS:
        results = analyze_model_v2(model)
        all_results[model] = results

    # Save results
    results_path = output_dir / 'mi_pivot_results_v2.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Plot comparison
    plot_comparison_v2(all_results, output_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (v2 - Delta-based)")
    print("="*60)
    print(f"{'Model':<15} {'Probe Acc':>10} {'All Delta':>12} {'Correct':>12} {'Incorrect':>12} {'p-value':>10}")
    print("-"*75)
    for m in MODELS:
        r = all_results[m]
        print(f"{m.replace('olmo3_', ''):<15} {r['probe_accuracy']*100:>9.1f}% {r['all_delta_stats']['mean']:>11.4f} "
              f"{r['correct_delta_stats']['mean']:>11.4f} {r['incorrect_delta_stats']['mean']:>11.4f} "
              f"{r['delta_ttest']['p_value']:>9.4f}")

    # Hypothesis test
    print("\n" + "="*60)
    print("HYPOTHESIS TEST (v2)")
    print("="*60)

    rl_delta = all_results['olmo3_rl_zero']['all_delta_stats']['mean']
    sft_delta = all_results['olmo3_sft']['all_delta_stats']['mean']
    base_delta = all_results['olmo3_base']['all_delta_stats']['mean']

    print(f"RLVR pivot delta: {rl_delta:.4f}")
    print(f"SFT pivot delta:  {sft_delta:.4f}")
    print(f"Base pivot delta: {base_delta:.4f}")

    # Check if correct vs incorrect samples differ at pivots
    print("\nCorrect vs Incorrect at pivots:")
    for m in MODELS:
        r = all_results[m]
        p = r['delta_ttest']['p_value']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        c_mean = r['correct_delta_stats']['mean']
        i_mean = r['incorrect_delta_stats']['mean']
        print(f"  {m.replace('olmo3_', '')}: correct={c_mean:.4f}, incorrect={i_mean:.4f}, p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
