#!/usr/bin/env python3
"""
MI Pivot Token Analysis v3

Fixed version that:
1. Truncates outputs at contamination markers (Passage:, Answer:, etc.)
2. Uses more specific pivot tokens for self-correction
3. Looks at MI trajectory around pivots
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

# More specific self-correction pivot tokens (not "so" which is too common)
CORRECTION_TOKENS = [
    "wait", "hmm", "however", "actually", "alternatively",
    "no,", "wrong", "mistake", "correction", "oops", "sorry",
    "instead", "rather", "let me think", "let me reconsider",
    "but wait", "hold on", "actually,"
]

# Broader deliberation tokens
DELIBERATION_TOKENS = [
    "but", "?", "think", "first", "then", "therefore", "thus",
    "because", "since", "so,", "hence"
]

CORRECTION_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in CORRECTION_TOKENS) + r')',
    re.IGNORECASE
)

DELIBERATION_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in DELIBERATION_TOKENS) + r')',
    re.IGNORECASE
)

# Truncation markers - cut output here
TRUNCATE_MARKERS = [
    "Passage:", "The 2010 United States Census",
    "According to the passage", "\n\nThe ", "\n\nIn ",
]

DATA_DIR = Path("/data/thanhdo/trajectories_0shot")
MODELS = ["olmo3_base", "olmo3_sft", "olmo3_rl_zero"]
MIDDLE_LAYER_IDX = 8


def truncate_output(text: str) -> str:
    """Remove contaminated portions of output."""
    for marker in TRUNCATE_MARKERS:
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()


def load_trajectories(model_name: str, task: str = "gsm8k"):
    """Load trajectories and metadata from HDF5 file."""
    h5_path = DATA_DIR / model_name / f"{task}_trajectories.h5"
    print(f"Loading {h5_path}...")

    with h5py.File(h5_path, 'r') as f:
        data = {
            'trajectories': f['trajectories'][:],
            'model_outputs': [truncate_output(x.decode() if isinstance(x, bytes) else x)
                             for x in f['model_outputs'][:]],
            'is_correct': f['is_correct'][:],
            'sequence_lengths': f['sequence_lengths'][:],
            'prompts': [x.decode() if isinstance(x, bytes) else x for x in f['prompts'][:]],
        }

    print(f"  Loaded {len(data['is_correct'])} samples, {data['is_correct'].sum()} correct ({100*data['is_correct'].mean():.1f}%)")
    return data


def find_tokens(text: str, pattern, n_positions: int = 512) -> list:
    """Find positions of tokens matching pattern."""
    tokens = []
    avg_chars_per_token = 4

    for match in pattern.finditer(text):
        char_pos = match.start()
        token_pos = min(char_pos // avg_chars_per_token, n_positions - 1)
        tokens.append((token_pos, match.group().lower()))

    return tokens


def compute_mi_profile(hidden_states: np.ndarray, probe: LogisticRegression, seq_len: int) -> np.ndarray:
    """Compute MI proxy (P(correct)) at each position."""
    valid_len = min(seq_len, hidden_states.shape[0])
    mi_profile = np.zeros(valid_len)

    for t in range(valid_len):
        h_t = hidden_states[t].reshape(1, -1).astype(np.float32)
        probs = probe.predict_proba(h_t)[0]
        mi_profile[t] = probs[1]

    return mi_profile


def compute_token_effects(mi_profile: np.ndarray, token_positions: list, window: int = 5):
    """
    Compute effects at token positions.

    Returns:
        deltas: MI change (post - pre)
        pre_vals: Average MI before token
        post_vals: Average MI after token
    """
    deltas = []
    pre_vals = []
    post_vals = []

    for pos, _ in token_positions:
        if pos < window or pos >= len(mi_profile) - window:
            continue

        pre = mi_profile[pos-window:pos].mean()
        post = mi_profile[pos+1:pos+window+1].mean()
        delta = post - pre

        deltas.append(delta)
        pre_vals.append(pre)
        post_vals.append(post)

    return deltas, pre_vals, post_vals


def train_probe(trajectories: np.ndarray, is_correct: np.ndarray, layer_idx: int = MIDDLE_LAYER_IDX):
    """Train correctness probe."""
    print(f"  Training probe on layer {layer_idx}...")

    X = trajectories[:, :256, layer_idx, :].mean(axis=1).astype(np.float32)
    y = is_correct.astype(int)

    probe = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    scores = cross_val_score(probe, X, y, cv=5, scoring='accuracy')
    print(f"  Probe CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    probe.fit(X, y)
    return probe, scores.mean()


def analyze_model_v3(model_name: str, task: str = "gsm8k"):
    """Analyze with cleaner token detection."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name} (v3 - cleaned)")
    print('='*60)

    data = load_trajectories(model_name, task)
    probe, probe_acc = train_probe(data['trajectories'], data['is_correct'])

    results = {
        'model': model_name,
        'n_samples': len(data['is_correct']),
        'accuracy': float(data['is_correct'].mean()),
        'probe_accuracy': float(probe_acc),
        # Token counts
        'n_samples_with_correction': 0,
        'n_samples_with_deliberation': 0,
        'total_correction_tokens': 0,
        'total_deliberation_tokens': 0,
        # Correction token effects
        'correction_deltas': [],
        'correction_deltas_correct': [],
        'correction_deltas_incorrect': [],
        # Deliberation token effects
        'deliberation_deltas': [],
        'deliberation_deltas_correct': [],
        'deliberation_deltas_incorrect': [],
    }

    print(f"\nComputing token effects...")

    for i in range(len(data['is_correct'])):
        output = data['model_outputs'][i]
        is_correct = data['is_correct'][i]
        seq_len = min(data['sequence_lengths'][i], len(output) // 4 + 10)  # Estimate
        hidden_states = data['trajectories'][i, :, MIDDLE_LAYER_IDX, :]

        correction_tokens = find_tokens(output, CORRECTION_PATTERN)
        deliberation_tokens = find_tokens(output, DELIBERATION_PATTERN)

        if correction_tokens:
            results['n_samples_with_correction'] += 1
            results['total_correction_tokens'] += len(correction_tokens)
        if deliberation_tokens:
            results['n_samples_with_deliberation'] += 1
            results['total_deliberation_tokens'] += len(deliberation_tokens)

        if not correction_tokens and not deliberation_tokens:
            continue

        mi_profile = compute_mi_profile(hidden_states, probe, seq_len)

        # Correction tokens
        if correction_tokens:
            deltas, _, _ = compute_token_effects(mi_profile, correction_tokens)
            results['correction_deltas'].extend(deltas)
            if is_correct:
                results['correction_deltas_correct'].extend(deltas)
            else:
                results['correction_deltas_incorrect'].extend(deltas)

        # Deliberation tokens
        if deliberation_tokens:
            deltas, _, _ = compute_token_effects(mi_profile, deliberation_tokens)
            results['deliberation_deltas'].extend(deltas)
            if is_correct:
                results['deliberation_deltas_correct'].extend(deltas)
            else:
                results['deliberation_deltas_incorrect'].extend(deltas)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(data['is_correct'])} samples")

    # Summary stats
    def stats_summary(arr):
        if len(arr) == 0:
            return {'mean': 0, 'std': 0, 'n': 0}
        return {'mean': float(np.mean(arr)), 'std': float(np.std(arr)), 'n': len(arr)}

    results['correction_stats'] = stats_summary(results['correction_deltas'])
    results['correction_correct_stats'] = stats_summary(results['correction_deltas_correct'])
    results['correction_incorrect_stats'] = stats_summary(results['correction_deltas_incorrect'])

    results['deliberation_stats'] = stats_summary(results['deliberation_deltas'])
    results['deliberation_correct_stats'] = stats_summary(results['deliberation_deltas_correct'])
    results['deliberation_incorrect_stats'] = stats_summary(results['deliberation_deltas_incorrect'])

    print(f"\nResults for {model_name}:")
    print(f"  Samples with correction tokens: {results['n_samples_with_correction']} ({results['total_correction_tokens']} total)")
    print(f"  Samples with deliberation tokens: {results['n_samples_with_deliberation']} ({results['total_deliberation_tokens']} total)")
    print(f"  Correction delta: {results['correction_stats']['mean']:.4f} +/- {results['correction_stats']['std']:.4f}")
    print(f"  Deliberation delta: {results['deliberation_stats']['mean']:.4f} +/- {results['deliberation_stats']['std']:.4f}")

    # Cleanup for JSON
    for key in ['correction_deltas', 'correction_deltas_correct', 'correction_deltas_incorrect',
                'deliberation_deltas', 'deliberation_deltas_correct', 'deliberation_deltas_incorrect']:
        results[key] = []

    return results


def plot_results(all_results: dict, output_dir: Path):
    """Plot comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = list(all_results.keys())

    # Plot 1: Token counts
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.35

    correction_counts = [all_results[m]['n_samples_with_correction'] for m in models]
    deliberation_counts = [all_results[m]['n_samples_with_deliberation'] for m in models]

    ax.bar(x - width/2, correction_counts, width, label='Correction', alpha=0.8, color='red')
    ax.bar(x + width/2, deliberation_counts, width, label='Deliberation', alpha=0.8, color='blue')

    ax.set_xlabel('Model')
    ax.set_ylabel('# Samples with Tokens')
    ax.set_title('Token Prevalence')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('olmo3_', '') for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Correction token deltas
    ax = axes[1]
    correction_means = [all_results[m]['correction_stats']['mean'] for m in models]
    correction_stds = [all_results[m]['correction_stats']['std'] for m in models]

    colors = ['blue' if 'rl' in m else 'orange' if 'sft' in m else 'gray' for m in models]
    ax.bar(range(len(models)), correction_means, yerr=correction_stds, color=colors, alpha=0.7, capsize=3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Model')
    ax.set_ylabel('MI Delta (post - pre)')
    ax.set_title('MI Change at Correction Tokens')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('olmo3_', '') for m in models])
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Deliberation token deltas
    ax = axes[2]
    delib_means = [all_results[m]['deliberation_stats']['mean'] for m in models]
    delib_stds = [all_results[m]['deliberation_stats']['std'] for m in models]

    ax.bar(range(len(models)), delib_means, yerr=delib_stds, color=colors, alpha=0.7, capsize=3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Model')
    ax.set_ylabel('MI Delta (post - pre)')
    ax.set_title('MI Change at Deliberation Tokens')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('olmo3_', '') for m in models])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'mi_pivot_comparison_v3.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    plt.close()


def main():
    print("="*60)
    print("MI Pivot Token Analysis v3 (Cleaned)")
    print("- Truncates at contamination markers")
    print("- Uses specific correction vs deliberation tokens")
    print("="*60)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}
    for model in MODELS:
        results = analyze_model_v3(model)
        all_results[model] = results

    # Save
    results_path = output_dir / 'mi_pivot_results_v3.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    plot_results(all_results, output_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (v3)")
    print("="*60)
    print(f"{'Model':<12} {'Acc':>6} {'#Corr':>6} {'#Delib':>6} {'Corr Δ':>10} {'Delib Δ':>10}")
    print("-"*60)
    for m in MODELS:
        r = all_results[m]
        print(f"{m.replace('olmo3_', ''):<12} {r['accuracy']*100:>5.1f}% {r['n_samples_with_correction']:>6} "
              f"{r['n_samples_with_deliberation']:>6} {r['correction_stats']['mean']:>10.4f} "
              f"{r['deliberation_stats']['mean']:>10.4f}")

    # Key finding
    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)

    rl_corr = all_results['olmo3_rl_zero']['correction_stats']['mean']
    sft_corr = all_results['olmo3_sft']['correction_stats']['mean']

    rl_delib = all_results['olmo3_rl_zero']['deliberation_stats']['mean']
    sft_delib = all_results['olmo3_sft']['deliberation_stats']['mean']

    rl_n_corr = all_results['olmo3_rl_zero']['n_samples_with_correction']
    sft_n_corr = all_results['olmo3_sft']['n_samples_with_correction']

    print(f"Correction token prevalence: RLVR={rl_n_corr}, SFT={sft_n_corr}")
    print(f"Correction MI delta: RLVR={rl_corr:.4f}, SFT={sft_corr:.4f}")
    print(f"Deliberation MI delta: RLVR={rl_delib:.4f}, SFT={sft_delib:.4f}")

    if rl_n_corr > sft_n_corr:
        print("\n[FINDING] RLVR uses MORE correction tokens than SFT")
    else:
        print("\n[FINDING] SFT uses MORE correction tokens than RLVR")

    if abs(rl_corr) > abs(sft_corr) and rl_corr > 0:
        print("[FINDING] RLVR shows POSITIVE MI change after correction (hypothesis supported)")
    else:
        print("[FINDING] No clear MI spike pattern at correction tokens")


if __name__ == "__main__":
    main()
