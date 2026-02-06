#!/usr/bin/env python3
"""
Cross-Task Cross-Model Error Direction Test

Tests whether SFT creates a "universal correctness direction" that works
across both tasks and models.

Key question: Does SFT's GSM8K error direction work on:
1. SFT's LogiQA (cross-task, same model) - we know this works
2. Base's GSM8K (cross-model, same task) - we just tested this
3. Base's LogiQA (cross-task AND cross-model) - the key test!

If (3) works better for SFT than for base, then SFT has created a universal direction.
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
from sklearn.metrics import roc_auc_score


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


def compute_error_direction(activations, labels):
    """Compute error direction as difference in means."""
    correct_mean = activations[labels].mean(axis=0)
    incorrect_mean = activations[~labels].mean(axis=0)
    error_dir = incorrect_mean - correct_mean
    error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)
    return error_dir


def test_direction(activations, labels, direction):
    """Test how well a direction separates correct/incorrect."""
    projections = activations @ direction
    try:
        auc = roc_auc_score(labels, projections)
        # Flip if needed (direction might be reversed)
        if auc < 0.5:
            auc = 1 - auc
    except:
        auc = 0.5
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--max-samples', type=int, default=200)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero']
    tasks = ['gsm8k', 'humaneval']  # Using humaneval instead of logiqa (truncated)

    # Load all data
    data = {}
    for model in models:
        for task in tasks:
            path = data_dir / model / f"{task}_trajectories.h5"
            if path.exists():
                traj, labels = load_trajectories(path, args.max_samples)
                act = get_mean_activations(traj, layer=-1)
                err_dir = compute_error_direction(act, labels)
                data[(model, task)] = {
                    'activations': act,
                    'labels': labels,
                    'error_dir': err_dir
                }
                print(f"Loaded {model}/{task}: {len(labels)} samples, {labels.sum()} correct")

    print("\n" + "="*80)
    print("CROSS-TASK CROSS-MODEL ERROR DIRECTION TEST")
    print("="*80)

    # Test matrix: for each source model/task, test on all target model/tasks
    print("\nAUC Matrix: rows=source (where error direction computed), cols=target (where applied)")
    print("-" * 80)

    header = "Source Model/Task".ljust(25) + " | "
    for target_model in models:
        for target_task in tasks:
            header += f"{target_model[-8:]}/{target_task[:4]}".center(12) + " "
    print(header)
    print("-" * 80)

    results = []
    for source_model in models:
        for source_task in tasks:
            if (source_model, source_task) not in data:
                continue

            row = f"{source_model}/{source_task}".ljust(25) + " | "
            source_dir = data[(source_model, source_task)]['error_dir']

            for target_model in models:
                for target_task in tasks:
                    if (target_model, target_task) not in data:
                        row += "   N/A   "
                        continue

                    target_act = data[(target_model, target_task)]['activations']
                    target_labels = data[(target_model, target_task)]['labels']

                    auc = test_direction(target_act, target_labels, source_dir)

                    # Highlight cross-task same-model (within model transfer)
                    # and cross-model same-task (within task transfer)
                    # and cross-both (the key test)
                    if source_model == target_model and source_task == target_task:
                        marker = "**"  # Same model same task (baseline)
                    elif source_model == target_model:
                        marker = "CT"  # Cross-task
                    elif source_task == target_task:
                        marker = "CM"  # Cross-model
                    else:
                        marker = "!!"  # Cross-both (key test)

                    row += f"{auc:.3f}{marker}".center(12) + " "

                    results.append({
                        'source_model': source_model,
                        'source_task': source_task,
                        'target_model': target_model,
                        'target_task': target_task,
                        'auc': auc,
                        'type': 'same' if source_model == target_model and source_task == target_task else
                               'cross_task' if source_model == target_model else
                               'cross_model' if source_task == target_task else
                               'cross_both'
                    })

            print(row)

    print("-" * 80)
    print("Legend: ** = same model/task, CT = cross-task, CM = cross-model, !! = cross-both")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY BY TRANSFER TYPE")
    print("="*80)

    for source_model in models:
        cross_task_aucs = [r['auc'] for r in results
                          if r['source_model'] == source_model
                          and r['type'] == 'cross_task']
        cross_model_aucs = [r['auc'] for r in results
                           if r['source_model'] == source_model
                           and r['type'] == 'cross_model']
        cross_both_aucs = [r['auc'] for r in results
                          if r['source_model'] == source_model
                          and r['type'] == 'cross_both']

        print(f"\n{source_model}:")
        if cross_task_aucs:
            print(f"  Cross-task (same model): {np.mean(cross_task_aucs):.3f}")
        if cross_model_aucs:
            print(f"  Cross-model (same task): {np.mean(cross_model_aucs):.3f}")
        if cross_both_aucs:
            print(f"  Cross-both (KEY TEST):   {np.mean(cross_both_aucs):.3f}")

    # The key comparison
    print("\n" + "="*80)
    print("KEY COMPARISON: Cross-both transfer (error dir from model A task X → model B task Y)")
    print("="*80)

    for source_model in models:
        cross_both = [r for r in results
                     if r['source_model'] == source_model
                     and r['type'] == 'cross_both']
        if cross_both:
            mean_auc = np.mean([r['auc'] for r in cross_both])
            print(f"{source_model}: mean AUC = {mean_auc:.3f}")
            for r in cross_both:
                print(f"  {r['source_task']} → {r['target_model']}/{r['target_task']}: {r['auc']:.3f}")


if __name__ == '__main__':
    main()
