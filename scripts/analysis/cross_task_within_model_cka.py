#!/usr/bin/env python3
"""
Cross-Task Within-Model CKA Analysis

Tests whether RL-Zero reduces the overlap between tasks (GSM8K vs HumanEval)
compared to base model.

Hypothesis: RL-Zero "sharpens" each task into its own corner, reducing cross-task CKA.
"""

import argparse
from pathlib import Path
import h5py
import numpy as np


def load_trajectories(filepath, max_samples=None):
    """Load trajectories and labels from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        traj = f['trajectories'][:]
        if 'is_correct' in f:
            labels = f['is_correct'][:]
        elif 'correct' in f:
            labels = f['correct'][:]
        else:
            raise KeyError(f"No correctness labels. Keys: {list(f.keys())}")

    if max_samples and len(traj) > max_samples:
        indices = np.random.RandomState(42).choice(len(traj), max_samples, replace=False)
        traj = traj[indices]
        labels = labels[indices]

    return traj.astype(np.float32), labels.astype(bool)


def get_mean_activations(trajectories, layer=-1):
    """Get mean activations across sequence for a specific layer."""
    return trajectories[:, :, layer, :].mean(axis=1)


def linear_CKA(X, Y):
    """Compute linear Centered Kernel Alignment (CKA)."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    K_X = X @ X.T
    K_Y = Y @ Y.T

    hsic_xy = np.trace(K_X @ K_Y)
    hsic_xx = np.trace(K_X @ K_X)
    hsic_yy = np.trace(K_Y @ K_Y)

    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)
    return cka


def compute_error_direction(activations, labels):
    """Compute error direction as difference in means."""
    correct_mean = activations[labels].mean(axis=0)
    incorrect_mean = activations[~labels].mean(axis=0)
    error_dir = incorrect_mean - correct_mean
    error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)
    return error_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--max-samples', type=int, default=200)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']
    tasks = ['gsm8k', 'humaneval']

    print("="*80)
    print("CROSS-TASK WITHIN-MODEL CKA ANALYSIS")
    print("="*80)
    print("\nHypothesis: RL-Zero reduces cross-task overlap compared to base")
    print()

    results = []

    for model in models:
        # Load both tasks for this model
        data = {}
        for task in tasks:
            path = data_dir / model / f"{task}_trajectories.h5"
            if not path.exists():
                print(f"  SKIP: {model}/{task} not found")
                continue

            traj, labels = load_trajectories(path, args.max_samples)
            act = get_mean_activations(traj, layer=-1)
            err_dir = compute_error_direction(act, labels)
            data[task] = {
                'activations': act,
                'labels': labels,
                'error_dir': err_dir,
                'n_correct': labels.sum(),
                'n_incorrect': (~labels).sum()
            }

        if len(data) < 2:
            continue

        # Cross-task CKA (GSM8K vs HumanEval activations)
        act_gsm = data['gsm8k']['activations']
        act_he = data['humaneval']['activations']

        # Need same number of samples for CKA
        n_samples = min(len(act_gsm), len(act_he))
        act_gsm = act_gsm[:n_samples]
        act_he = act_he[:n_samples]

        cka = linear_CKA(act_gsm, act_he)

        # Error direction alignment
        err_dir_gsm = data['gsm8k']['error_dir']
        err_dir_he = data['humaneval']['error_dir']
        err_dir_cos = np.abs(np.dot(err_dir_gsm, err_dir_he))

        results.append({
            'model': model,
            'cross_task_cka': cka,
            'error_dir_cos': err_dir_cos
        })

        print(f"{model}:")
        print(f"  GSM8K: {data['gsm8k']['n_correct']} correct, {data['gsm8k']['n_incorrect']} incorrect")
        print(f"  HumanEval: {data['humaneval']['n_correct']} correct, {data['humaneval']['n_incorrect']} incorrect")
        print(f"  Cross-task CKA (GSM8K vs HumanEval): {cka:.4f}")
        print(f"  Error direction cos similarity: {err_dir_cos:.4f}")
        print()

    # Summary comparison
    print("="*80)
    print("SUMMARY: Cross-Task Similarity by Model")
    print("="*80)
    print()
    print(f"{'Model':<20} {'Cross-Task CKA':<20} {'Error Dir Cos':<20}")
    print("-"*60)

    for r in results:
        print(f"{r['model']:<20} {r['cross_task_cka']:<20.4f} {r['error_dir_cos']:<20.4f}")

    print()
    print("="*80)
    print("INTERPRETATION")
    print("="*80)

    # Find base and rl_zero results
    base_result = next((r for r in results if r['model'] == 'olmo3_base'), None)
    rl_result = next((r for r in results if r['model'] == 'olmo3_rl_zero'), None)
    sft_result = next((r for r in results if r['model'] == 'olmo3_sft'), None)

    if base_result and rl_result:
        cka_diff = base_result['cross_task_cka'] - rl_result['cross_task_cka']
        print(f"\nBase vs RL-Zero cross-task CKA difference: {cka_diff:.4f}")

        if cka_diff > 0.01:
            print("  → RL-Zero REDUCES cross-task overlap (tasks become more orthogonal)")
            print("  → This explains why transfer accuracy drops!")
        elif cka_diff < -0.01:
            print("  → RL-Zero INCREASES cross-task overlap (unexpected)")
        else:
            print("  → Cross-task overlap is similar (CKA not the explanation)")

    if sft_result and base_result:
        cka_diff = sft_result['cross_task_cka'] - base_result['cross_task_cka']
        print(f"\nSFT vs Base cross-task CKA difference: {cka_diff:.4f}")

        if cka_diff > 0.01:
            print("  → SFT INCREASES cross-task overlap (tasks become more aligned)")
            print("  → This explains why SFT has better bidirectional transfer!")
        elif cka_diff < -0.01:
            print("  → SFT DECREASES cross-task overlap (unexpected)")
        else:
            print("  → Cross-task overlap is similar")


if __name__ == '__main__':
    main()
