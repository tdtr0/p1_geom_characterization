#!/usr/bin/env python3
"""
Analyze actual self-corrections in reasoning traces.

Detects when the model:
1. Computes one answer
2. Says "Wait", "Actually", etc.
3. Computes a DIFFERENT answer

Then tests if velocity at true corrections predicts success.
"""

import re
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy import stats


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numerical answer from GSM8K format."""
    # Try #### format first
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return float(match.group(1).replace(',', ''))

    # Try \boxed{} format
    match = re.search(r'\\boxed\{(-?[\d,]+\.?\d*)\}', text)
    if match:
        return float(match.group(1).replace(',', ''))

    return None


def extract_ground_truth(gt_str: str) -> Optional[float]:
    """Extract numerical answer from ground truth string."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', gt_str)
    if match:
        return float(match.group(1).replace(',', ''))
    try:
        return float(gt_str.strip().replace(',', ''))
    except:
        return None


def find_self_corrections(text: str) -> List[Dict]:
    """Find actual self-corrections where the model changes a number.

    Looks for patterns like:
    - "= X ... Wait ... = Y" where X != Y
    - "answer is X ... Actually ... answer is Y" where X != Y
    """
    corrections = []

    # Pattern: pivot word followed by a number change
    pivot_pattern = r'(Wait|But wait|Actually|no,|Hmm|however)'

    # Find all numbers in text (must have at least one digit)
    number_pattern = r'(?:=|is|:)\s*\$?(-?\d[\d,]*\.?\d*)'
    numbers_with_pos = []
    for m in re.finditer(number_pattern, text):
        try:
            num = float(m.group(1).replace(',', ''))
            numbers_with_pos.append((m.start(), num))
        except ValueError:
            continue

    # Find pivot positions
    pivots = [(m.start(), m.group(1)) for m in re.finditer(pivot_pattern, text, re.I)]

    # For each pivot, check if numbers before and after differ
    for pivot_pos, pivot_word in pivots:
        # Find numbers before pivot (within 200 chars)
        before_nums = [(pos, num) for pos, num in numbers_with_pos
                       if pivot_pos - 200 < pos < pivot_pos]

        # Find numbers after pivot (within 200 chars)
        after_nums = [(pos, num) for pos, num in numbers_with_pos
                      if pivot_pos < pos < pivot_pos + 200]

        if before_nums and after_nums:
            last_before = before_nums[-1][1]
            first_after = after_nums[0][1]

            # Check if number changed
            if abs(last_before - first_after) > 0.01:
                corrections.append({
                    'pivot_word': pivot_word,
                    'pivot_pos': pivot_pos,
                    'before_value': last_before,
                    'after_value': first_after,
                    'context': text[max(0, pivot_pos-50):pivot_pos+100]
                })

    return corrections


def find_explicit_corrections(text: str) -> List[Dict]:
    """Find explicit correction phrases with numerical changes.

    More strict: looks for phrases like:
    - "Wait, X should be Y"
    - "Actually, that's wrong. It's Y"
    - "Let me recalculate... = Y"
    """
    corrections = []

    # Explicit correction patterns
    patterns = [
        r'(Wait[,.]?\s*(?:that|I|let me|actually)[^.]*?)\s*=\s*(-?[\d,]+\.?\d*)',
        r'((?:Actually|However)[,.]?\s*[^.]*?(?:should be|is actually|is really|correct (?:answer|value)))\s*[:=]?\s*\$?(-?[\d,]+\.?\d*)',
        r'((?:no|wait)[,.]?\s*[^.]*?(?:wrong|mistake|error|incorrect)[^.]*?)\s*[:=]?\s*\$?(-?[\d,]+\.?\d*)',
        r'(Let me (?:recalculate|check|verify)[^.]*?)\s*=\s*(-?[\d,]+\.?\d*)',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            corrections.append({
                'pattern': pattern[:30],
                'context': match.group(1)[:100],
                'corrected_to': float(match.group(2).replace(',', '')),
                'char_pos': match.start()
            })

    return corrections


def compute_velocity(trajectory: np.ndarray) -> np.ndarray:
    """Compute velocity (L2 norm of difference) along trajectory."""
    avg_traj = trajectory.mean(axis=1)  # (seq_len, hidden_dim)
    diffs = np.diff(avg_traj, axis=0)  # (seq_len-1, hidden_dim)
    velocities = np.linalg.norm(diffs, axis=1)
    return velocities


def analyze_sample(
    sample_idx: int,
    generated_text: str,
    ground_truth: str,
    trajectory: np.ndarray,
    seq_length: int,
) -> Dict:
    """Analyze a single sample for self-corrections."""

    # Extract final answer
    predicted = extract_gsm8k_answer(generated_text)
    gt = extract_ground_truth(ground_truth)

    is_correct = False
    if predicted is not None and gt is not None:
        is_correct = abs(predicted - gt) < 0.01

    # Find self-corrections
    corrections = find_self_corrections(generated_text)
    explicit_corrections = find_explicit_corrections(generated_text)

    # Compute velocities
    traj = trajectory[:seq_length]
    velocities = compute_velocity(traj)

    # Get velocity at correction points
    correction_velocities = []
    text_len = len(generated_text)

    for corr in corrections:
        token_pos = int(corr['pivot_pos'] / text_len * seq_length)
        if 2 < token_pos < len(velocities) - 2:
            window = velocities[max(0, token_pos-2):min(len(velocities), token_pos+3)]
            correction_velocities.append({
                'velocity': float(np.mean(window)),
                'before_value': corr['before_value'],
                'after_value': corr['after_value'],
                'pivot_word': corr['pivot_word']
            })

    for corr in explicit_corrections:
        token_pos = int(corr['char_pos'] / text_len * seq_length)
        if 2 < token_pos < len(velocities) - 2:
            window = velocities[max(0, token_pos-2):min(len(velocities), token_pos+3)]
            correction_velocities.append({
                'velocity': float(np.mean(window)),
                'corrected_to': corr['corrected_to'],
                'context': corr['context'][:50]
            })

    return {
        'sample_idx': sample_idx,
        'is_correct': is_correct,
        'predicted': float(predicted) if predicted else None,
        'ground_truth': float(gt) if gt else None,
        'n_self_corrections': len(corrections),
        'n_explicit_corrections': len(explicit_corrections),
        'correction_velocities': correction_velocities,
        'has_correction': len(corrections) > 0 or len(explicit_corrections) > 0,
        'mean_velocity': float(np.mean(velocities)) if len(velocities) > 0 else None,
        'text_preview': generated_text[:500]
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='experiments/aha_moment/data/pivot_collection/pivot_trajectories.h5')
    parser.add_argument('--output', type=str,
                        default='experiments/aha_moment/results/self_correction/')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}...")

    with h5py.File(input_path, 'r') as f:
        generated_texts = [t.decode() if isinstance(t, bytes) else t for t in f['generated_texts'][:]]
        ground_truths = [t.decode() if isinstance(t, bytes) else t for t in f['ground_truth'][:]]
        trajectories = f['trajectories'][:]
        seq_lengths = f['sequence_lengths'][:]

    print(f"Loaded {len(generated_texts)} samples")

    # Analyze all samples
    all_results = []
    for i in range(len(generated_texts)):
        result = analyze_sample(
            i, generated_texts[i], ground_truths[i],
            trajectories[i], seq_lengths[i]
        )
        all_results.append(result)

    # Split by correction behavior
    with_corrections = [r for r in all_results if r['has_correction']]
    without_corrections = [r for r in all_results if not r['has_correction']]

    print(f"\n=== SELF-CORRECTION DETECTION ===")
    print(f"Samples WITH self-corrections: {len(with_corrections)} ({100*len(with_corrections)/len(all_results):.1f}%)")
    print(f"Samples WITHOUT self-corrections: {len(without_corrections)} ({100*len(without_corrections)/len(all_results):.1f}%)")

    # Correctness by correction behavior
    correct_with_corr = sum(1 for r in with_corrections if r['is_correct'])
    correct_without_corr = sum(1 for r in without_corrections if r['is_correct'])

    print(f"\n=== CORRECTION → CORRECTNESS ===")
    if with_corrections:
        print(f"WITH corrections: {100*correct_with_corr/len(with_corrections):.1f}% correct ({correct_with_corr}/{len(with_corrections)})")
    if without_corrections:
        print(f"WITHOUT corrections: {100*correct_without_corr/len(without_corrections):.1f}% correct ({correct_without_corr}/{len(without_corrections)})")

    # Chi-square test
    if len(with_corrections) >= 5 and len(without_corrections) >= 5:
        contingency = [
            [correct_with_corr, len(with_corrections) - correct_with_corr],
            [correct_without_corr, len(without_corrections) - correct_without_corr]
        ]
        chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
        print(f"Chi-square: chi2={chi2:.3f}, p={chi_p:.4f}")

    # Velocity analysis for samples with corrections
    print(f"\n=== VELOCITY AT CORRECTION POINTS ===")

    # Collect velocities from samples with corrections
    correct_correction_vels = []
    incorrect_correction_vels = []

    for r in with_corrections:
        if r['correction_velocities']:
            mean_vel = np.mean([cv['velocity'] for cv in r['correction_velocities']])
            if r['is_correct']:
                correct_correction_vels.append(mean_vel)
            else:
                incorrect_correction_vels.append(mean_vel)

    if correct_correction_vels and incorrect_correction_vels:
        print(f"Correct samples (with corrections): n={len(correct_correction_vels)}, mean vel={np.mean(correct_correction_vels):.3f}")
        print(f"Incorrect samples (with corrections): n={len(incorrect_correction_vels)}, mean vel={np.mean(incorrect_correction_vels):.3f}")

        if len(correct_correction_vels) >= 3 and len(incorrect_correction_vels) >= 3:
            t_stat, p_value = stats.ttest_ind(correct_correction_vels, incorrect_correction_vels)
            pooled_std = np.sqrt(
                ((len(correct_correction_vels)-1) * np.var(correct_correction_vels) +
                 (len(incorrect_correction_vels)-1) * np.var(incorrect_correction_vels)) /
                (len(correct_correction_vels) + len(incorrect_correction_vels) - 2)
            )
            cohens_d = (np.mean(correct_correction_vels) - np.mean(incorrect_correction_vels)) / pooled_std

            print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")
            print(f"Cohen's d: {cohens_d:.3f}")
    else:
        print(f"Correct samples with velocities: {len(correct_correction_vels)}")
        print(f"Incorrect samples with velocities: {len(incorrect_correction_vels)}")
        print("Insufficient samples for comparison")

    # Show examples
    print(f"\n=== EXAMPLE CORRECTIONS ===")
    examples_shown = 0
    for r in with_corrections[:10]:
        if r['n_self_corrections'] > 0 and examples_shown < 5:
            print(f"\nSample {r['sample_idx']} (Correct: {r['is_correct']}):")
            # Find the correction in the text
            for corr in find_self_corrections(r['text_preview']):
                print(f"  '{corr['pivot_word']}': {corr['before_value']} → {corr['after_value']}")
                print(f"  Context: ...{corr['context'][:80]}...")
            examples_shown += 1

    # Save results
    summary = {
        'n_samples': len(all_results),
        'n_with_corrections': len(with_corrections),
        'n_without_corrections': len(without_corrections),
        'correction_rate': len(with_corrections) / len(all_results),
        'correctness_with_corrections': correct_with_corr / len(with_corrections) if with_corrections else None,
        'correctness_without_corrections': correct_without_corr / len(without_corrections) if without_corrections else None,
        'velocity_analysis': {
            'n_correct_with_vel': len(correct_correction_vels),
            'n_incorrect_with_vel': len(incorrect_correction_vels),
            'correct_mean_vel': float(np.mean(correct_correction_vels)) if correct_correction_vels else None,
            'incorrect_mean_vel': float(np.mean(incorrect_correction_vels)) if incorrect_correction_vels else None,
        }
    }

    output_file = output_dir / 'self_correction_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_file}")


if __name__ == '__main__':
    main()
