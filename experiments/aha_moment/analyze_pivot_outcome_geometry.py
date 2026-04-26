#!/usr/bin/env python3
"""
Analyze if pivot velocity predicts correction success.

H1: Pivots that lead to successful corrections have different geometry
    than pivots that lead to failed corrections.

Uses existing pivot_trajectories.h5 data (200 samples).
"""

import re
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy import stats


# Pivot detection patterns
PIVOT_PATTERNS = [
    (r'\bWait\b', 'Wait'),
    (r'\bBUT\b', 'BUT'),
    (r'\bactually\b', 'actually'),
    (r'\bhowever\b', 'however'),
    (r'\bhmm\b', 'hmm'),
    (r'\bActually\b', 'Actually'),
]


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numerical answer from GSM8K format.

    Looks for patterns like:
    - #### 42
    - \\boxed{42}
    - answer is 42
    - = 42 (at end of reasoning)
    """
    # Try #### format first
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return float(match.group(1).replace(',', ''))

    # Try \boxed{} format
    match = re.search(r'\\boxed\{(-?[\d,]+\.?\d*)\}', text)
    if match:
        return float(match.group(1).replace(',', ''))

    # Try "answer is X" format
    match = re.search(r'(?:answer|total|result)\s+(?:is|=|:)\s*\$?(-?[\d,]+\.?\d*)', text, re.I)
    if match:
        return float(match.group(1).replace(',', ''))

    # Try last number in text
    numbers = re.findall(r'(-?[\d,]+\.?\d*)', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass

    return None


def extract_ground_truth(gt_str: str) -> Optional[float]:
    """Extract numerical answer from ground truth string."""
    # GSM8K ground truth format: "... #### 42"
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', gt_str)
    if match:
        return float(match.group(1).replace(',', ''))

    # Try direct number
    try:
        return float(gt_str.strip().replace(',', ''))
    except:
        pass

    return None


def detect_pivots(text: str, tokenizer=None) -> List[Dict]:
    """Detect pivot positions in text.

    Returns list of {pattern, char_pos, token_pos} dicts.
    """
    pivots = []

    for pattern, name in PIVOT_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            pivots.append({
                'pattern': name,
                'char_pos': match.start(),
                'text_around': text[max(0, match.start()-30):match.end()+30]
            })

    return pivots


def compute_velocity(trajectory: np.ndarray) -> np.ndarray:
    """Compute velocity (L2 norm of difference) along trajectory.

    Args:
        trajectory: (seq_len, n_layers, hidden_dim)

    Returns:
        velocities: (seq_len-1,) - velocity at each position
    """
    # Average across layers
    avg_traj = trajectory.mean(axis=1)  # (seq_len, hidden_dim)

    # Compute differences
    diffs = np.diff(avg_traj, axis=0)  # (seq_len-1, hidden_dim)

    # L2 norm
    velocities = np.linalg.norm(diffs, axis=1)  # (seq_len-1,)

    return velocities


def char_to_token_pos(text: str, char_pos: int, tokenizer) -> int:
    """Convert character position to token position."""
    # Tokenize with offset mapping
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    for idx, (start, end) in enumerate(encoding.offset_mapping):
        if start <= char_pos < end:
            return idx

    # Fallback: approximate
    return int(char_pos / len(text) * len(encoding.input_ids))


def analyze_sample(
    sample_idx: int,
    generated_text: str,
    ground_truth: str,
    trajectory: np.ndarray,
    seq_length: int,
) -> Optional[Dict]:
    """Analyze a single sample.

    Returns dict with correctness, pivot info, velocities.
    """
    # Extract answers
    predicted = extract_gsm8k_answer(generated_text)
    gt = extract_ground_truth(ground_truth)

    if predicted is None or gt is None:
        return None

    # Determine correctness
    is_correct = abs(predicted - gt) < 0.01  # Allow small floating point error

    # Detect pivots
    pivots = detect_pivots(generated_text)

    if not pivots:
        return None  # Need pivots to analyze

    # Compute velocities
    traj = trajectory[:seq_length]  # Trim to actual sequence
    velocities = compute_velocity(traj)

    # For each pivot, estimate token position and get velocity
    # Approximate: char_pos / len(text) * seq_length
    pivot_velocities = []
    text_len = len(generated_text)

    for pivot in pivots:
        # Approximate token position
        token_pos = int(pivot['char_pos'] / text_len * seq_length)

        # Get velocity at pivot (and nearby tokens)
        if 0 < token_pos < len(velocities) - 1:
            # Average velocity in window around pivot
            window = velocities[max(0, token_pos-2):min(len(velocities), token_pos+3)]
            pivot_velocity = float(np.mean(window))
            pivot_velocities.append({
                'pattern': pivot['pattern'],
                'token_pos': token_pos,
                'velocity': pivot_velocity,
            })

    if not pivot_velocities:
        return None

    # Get random token velocities for comparison
    n_random = len(pivot_velocities) * 3
    random_positions = np.random.choice(
        range(10, len(velocities) - 10),  # Avoid edges
        size=min(n_random, len(velocities) - 20),
        replace=False
    )
    random_velocities = [float(velocities[pos]) for pos in random_positions]

    return {
        'sample_idx': sample_idx,
        'is_correct': is_correct,
        'predicted': float(predicted),
        'ground_truth': float(gt),
        'n_pivots': len(pivot_velocities),
        'pivot_velocities': pivot_velocities,
        'mean_pivot_velocity': float(np.mean([p['velocity'] for p in pivot_velocities])),
        'random_velocities': random_velocities,
        'mean_random_velocity': float(np.mean(random_velocities)) if random_velocities else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='experiments/aha_moment/data/pivot_collection/pivot_trajectories.h5')
    parser.add_argument('--output', type=str,
                        default='experiments/aha_moment/results/pivot_outcome/')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}...")

    # Load data
    with h5py.File(input_path, 'r') as f:
        generated_texts = [t.decode() if isinstance(t, bytes) else t for t in f['generated_texts'][:]]
        ground_truths = [t.decode() if isinstance(t, bytes) else t for t in f['ground_truth'][:]]
        trajectories = f['trajectories'][:]
        seq_lengths = f['sequence_lengths'][:]

    print(f"Loaded {len(generated_texts)} samples")

    # Analyze each sample
    results = []
    for i in range(len(generated_texts)):
        result = analyze_sample(
            i, generated_texts[i], ground_truths[i],
            trajectories[i], seq_lengths[i]
        )
        if result:
            results.append(result)

    print(f"\nAnalyzed {len(results)} samples with pivots")

    # Split by correctness
    correct_samples = [r for r in results if r['is_correct']]
    incorrect_samples = [r for r in results if not r['is_correct']]

    print(f"  Correct: {len(correct_samples)} ({100*len(correct_samples)/len(results):.1f}%)")
    print(f"  Incorrect: {len(incorrect_samples)} ({100*len(incorrect_samples)/len(results):.1f}%)")

    # Extract velocities by group
    correct_pivot_velocities = [r['mean_pivot_velocity'] for r in correct_samples]
    incorrect_pivot_velocities = [r['mean_pivot_velocity'] for r in incorrect_samples]

    print(f"\n=== PIVOT VELOCITY BY OUTCOME ===")
    print(f"Correct samples:   mean={np.mean(correct_pivot_velocities):.3f}, std={np.std(correct_pivot_velocities):.3f}")
    print(f"Incorrect samples: mean={np.mean(incorrect_pivot_velocities):.3f}, std={np.std(incorrect_pivot_velocities):.3f}")

    # Statistical test
    if len(correct_pivot_velocities) >= 5 and len(incorrect_pivot_velocities) >= 5:
        t_stat, p_value = stats.ttest_ind(correct_pivot_velocities, incorrect_pivot_velocities)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(correct_pivot_velocities)-1) * np.var(correct_pivot_velocities) +
             (len(incorrect_pivot_velocities)-1) * np.var(incorrect_pivot_velocities)) /
            (len(correct_pivot_velocities) + len(incorrect_pivot_velocities) - 2)
        )
        cohens_d = (np.mean(correct_pivot_velocities) - np.mean(incorrect_pivot_velocities)) / pooled_std

        print(f"\nt-test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"Cohen's d: {cohens_d:.3f}")

        # Interpretation
        if p_value < 0.05:
            if cohens_d > 0:
                print("\n** FINDING: Correct solutions have HIGHER pivot velocity (p < 0.05) **")
            else:
                print("\n** FINDING: Correct solutions have LOWER pivot velocity (p < 0.05) **")
        else:
            print("\n** FINDING: No significant difference in pivot velocity by outcome **")
    else:
        t_stat, p_value, cohens_d = None, None, None
        print("\nInsufficient samples for statistical test")

    # Additional analysis: Do low-velocity pivots predict success?
    print(f"\n=== VELOCITY → SUCCESS PREDICTION ===")

    all_velocities = [(r['mean_pivot_velocity'], r['is_correct']) for r in results]
    all_velocities.sort(key=lambda x: x[0])

    # Split into low/high velocity groups
    median_velocity = np.median([v for v, _ in all_velocities])
    low_velocity = [v for v, c in all_velocities if v < median_velocity]
    low_velocity_correct = [c for v, c in all_velocities if v < median_velocity]
    high_velocity = [v for v, c in all_velocities if v >= median_velocity]
    high_velocity_correct = [c for v, c in all_velocities if v >= median_velocity]

    low_success_rate = np.mean(low_velocity_correct) if low_velocity_correct else 0
    high_success_rate = np.mean(high_velocity_correct) if high_velocity_correct else 0

    print(f"Median velocity: {median_velocity:.3f}")
    print(f"Low velocity group: {len(low_velocity)} samples, {100*low_success_rate:.1f}% correct")
    print(f"High velocity group: {len(high_velocity)} samples, {100*high_success_rate:.1f}% correct")

    # Chi-square test for velocity → success
    if len(low_velocity) >= 5 and len(high_velocity) >= 5:
        contingency = [
            [sum(low_velocity_correct), len(low_velocity_correct) - sum(low_velocity_correct)],
            [sum(high_velocity_correct), len(high_velocity_correct) - sum(high_velocity_correct)]
        ]
        chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
        print(f"Chi-square test: chi2={chi2:.3f}, p={chi_p:.4f}")

        if chi_p < 0.05:
            if low_success_rate > high_success_rate:
                print("\n** FINDING: Low velocity predicts SUCCESS (p < 0.05) **")
            else:
                print("\n** FINDING: High velocity predicts SUCCESS (p < 0.05) **")
        else:
            print("\n** FINDING: Velocity does NOT predict success (p >= 0.05) **")

    # Per-pattern analysis
    print(f"\n=== BY PIVOT PATTERN ===")
    pattern_correct = defaultdict(list)
    pattern_incorrect = defaultdict(list)

    for r in results:
        for pv in r['pivot_velocities']:
            if r['is_correct']:
                pattern_correct[pv['pattern']].append(pv['velocity'])
            else:
                pattern_incorrect[pv['pattern']].append(pv['velocity'])

    for pattern in set(pattern_correct.keys()) | set(pattern_incorrect.keys()):
        c_vels = pattern_correct.get(pattern, [])
        i_vels = pattern_incorrect.get(pattern, [])

        if c_vels and i_vels:
            print(f"\n{pattern}:")
            print(f"  Correct: n={len(c_vels)}, mean={np.mean(c_vels):.3f}")
            print(f"  Incorrect: n={len(i_vels)}, mean={np.mean(i_vels):.3f}")

            if len(c_vels) >= 3 and len(i_vels) >= 3:
                t, p = stats.ttest_ind(c_vels, i_vels)
                print(f"  t={t:.2f}, p={p:.4f}")

    # Save results
    summary = {
        'n_samples': len(results),
        'n_correct': len(correct_samples),
        'n_incorrect': len(incorrect_samples),
        'correct_rate': len(correct_samples) / len(results),
        'velocity_analysis': {
            'correct_mean': float(np.mean(correct_pivot_velocities)),
            'correct_std': float(np.std(correct_pivot_velocities)),
            'incorrect_mean': float(np.mean(incorrect_pivot_velocities)),
            'incorrect_std': float(np.std(incorrect_pivot_velocities)),
            't_statistic': float(t_stat) if t_stat else None,
            'p_value': float(p_value) if p_value else None,
            'cohens_d': float(cohens_d) if cohens_d else None,
        },
        'prediction_analysis': {
            'median_velocity': float(median_velocity),
            'low_velocity_success_rate': float(low_success_rate),
            'high_velocity_success_rate': float(high_success_rate),
        },
        'per_pattern': {
            pattern: {
                'correct_count': len(pattern_correct.get(pattern, [])),
                'incorrect_count': len(pattern_incorrect.get(pattern, [])),
                'correct_mean': float(np.mean(pattern_correct[pattern])) if pattern in pattern_correct else None,
                'incorrect_mean': float(np.mean(pattern_incorrect[pattern])) if pattern in pattern_incorrect else None,
            }
            for pattern in set(pattern_correct.keys()) | set(pattern_incorrect.keys())
        }
    }

    output_file = output_dir / 'pivot_outcome_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_file}")

    # Save detailed results
    detailed_file = output_dir / 'pivot_outcome_detailed.json'
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to {detailed_file}")


if __name__ == '__main__':
    main()
