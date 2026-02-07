#!/usr/bin/env python3
"""
Layer-Localized Path Dynamics Analysis

Characterize "working harder" geometrically via layer-by-layer path properties:
1. Per-layer step magnitude (where does the path move most?)
2. Jump detection (discontinuities in path properties)
3. Acceleration/deceleration profiles
4. Dimensionality trajectory

Cross-validates with prior findings:
- CKA peak at L7
- Error direction at L24-28
- Wynroe patching L16-18

Usage:
    python layer_path_dynamics.py --data-dir data/trajectories_0shot --model olmo3_rl_zero --task gsm8k
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_trajectories(filepath, max_samples=None):
    """Load trajectories and labels from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        trajectories = f['trajectories'][:]
        if 'is_correct' in f:
            labels = f['is_correct'][:]
        else:
            labels = f['correct'][:]

        if max_samples and max_samples < len(trajectories):
            np.random.seed(42)
            indices = np.random.choice(len(trajectories), max_samples, replace=False)
            trajectories = trajectories[indices]
            labels = labels[indices]

    return trajectories.astype(np.float32), labels.astype(bool)


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_per_layer_step_profile(trajectories, labels):
    """Compare step magnitude at each layer: correct vs incorrect."""
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    print("\n" + "="*70)
    print("ANALYSIS 1: Per-Layer Step Magnitude Profile")
    print("="*70)
    print("\nHypothesis: Correct vs incorrect differ in WHERE they take big steps")
    print("Positive d = correct takes bigger steps, Negative d = incorrect takes bigger\n")

    print(f"{'Layer':<8} {'Correct':<12} {'Incorrect':<12} {'Diff':<12} {'Cohen d':<10} {'p-value':<10} {'Sig':<5}")
    print("-"*70)

    results_by_layer = []
    for l in range(n_layers - 1):
        # Step magnitude for each sample (mean over tokens)
        steps = np.linalg.norm(
            trajectories[:, :, l+1, :] - trajectories[:, :, l, :],
            axis=2
        ).mean(axis=1)  # (n_samples,)

        correct_steps = steps[labels]
        incorrect_steps = steps[~labels]

        d = cohens_d(correct_steps, incorrect_steps)
        _, p = stats.ttest_ind(correct_steps, incorrect_steps)

        diff = np.mean(correct_steps) - np.mean(incorrect_steps)
        sig = "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""

        print(f"L{l}→L{l+1:<4} {np.mean(correct_steps):<12.2f} {np.mean(incorrect_steps):<12.2f} "
              f"{diff:<12.2f} {d:<10.3f} {p:<10.4f} {sig:<5}")

        results_by_layer.append({
            'layer': l,
            'd': float(d),
            'p': float(p),
            'correct_mean': float(np.mean(correct_steps)),
            'incorrect_mean': float(np.mean(incorrect_steps)),
            'diff': float(diff)
        })

    # Find most discriminative layer
    best_layer = max(results_by_layer, key=lambda x: abs(x['d']))
    print("-"*70)
    print(f"\nMost discriminative layer: L{best_layer['layer']}→L{best_layer['layer']+1} "
          f"(d={best_layer['d']:.3f}, p={best_layer['p']:.4f})")

    return results_by_layer


def detect_jumps(step_profile):
    """Detect discontinuities in step profile using derivatives."""
    step_profile = np.array(step_profile)

    # First derivative (change in step magnitude)
    d_step = np.diff(step_profile)

    # Statistical threshold
    if len(d_step) > 1 and np.std(np.abs(d_step)) > 0:
        threshold = np.mean(np.abs(d_step)) + 2 * np.std(np.abs(d_step))
        jumps = np.where(np.abs(d_step) > threshold)[0]
    else:
        jumps = np.array([])

    # Second derivative (curvature of step profile)
    dd_step = np.diff(d_step) if len(d_step) > 1 else np.array([0])

    return {
        'jump_layers': jumps.tolist(),
        'jump_magnitudes': d_step[jumps].tolist() if len(jumps) > 0 else [],
        'max_curvature_layer': int(np.argmax(np.abs(dd_step))) if len(dd_step) > 0 else 0,
        'n_jumps': len(jumps)
    }


def analyze_jumps(trajectories, labels):
    """Compare jump patterns between correct and incorrect."""
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    print("\n" + "="*70)
    print("ANALYSIS 2: Jump Detection (Discontinuities)")
    print("="*70)
    print("\nHypothesis: Correct solutions have different jump patterns")

    # Compute step profile for each sample
    correct_jumps = []
    incorrect_jumps = []

    for i in range(n_samples):
        # Step magnitudes for this sample
        step_profile = []
        for l in range(n_layers - 1):
            step = np.linalg.norm(
                trajectories[i, :, l+1, :] - trajectories[i, :, l, :],
                axis=1
            ).mean()
            step_profile.append(step)

        jump_info = detect_jumps(step_profile)

        if labels[i]:
            correct_jumps.append(jump_info)
        else:
            incorrect_jumps.append(jump_info)

    # Compare jump counts
    correct_n_jumps = [j['n_jumps'] for j in correct_jumps]
    incorrect_n_jumps = [j['n_jumps'] for j in incorrect_jumps]

    d_jumps = cohens_d(correct_n_jumps, incorrect_n_jumps)
    _, p_jumps = stats.ttest_ind(correct_n_jumps, incorrect_n_jumps)

    print(f"\nJump count comparison:")
    print(f"  Correct mean: {np.mean(correct_n_jumps):.2f} jumps")
    print(f"  Incorrect mean: {np.mean(incorrect_n_jumps):.2f} jumps")
    print(f"  Cohen's d: {d_jumps:.3f}, p-value: {p_jumps:.4f}")

    # Analyze jump locations
    correct_jump_layers = []
    incorrect_jump_layers = []
    for j in correct_jumps:
        correct_jump_layers.extend(j['jump_layers'])
    for j in incorrect_jumps:
        incorrect_jump_layers.extend(j['jump_layers'])

    print(f"\nJump location distribution:")
    if len(correct_jump_layers) > 0:
        print(f"  Correct: mode at L{stats.mode(correct_jump_layers, keepdims=False).mode if len(correct_jump_layers) > 0 else 'N/A'}")
    if len(incorrect_jump_layers) > 0:
        print(f"  Incorrect: mode at L{stats.mode(incorrect_jump_layers, keepdims=False).mode if len(incorrect_jump_layers) > 0 else 'N/A'}")

    return {
        'correct_n_jumps_mean': float(np.mean(correct_n_jumps)),
        'incorrect_n_jumps_mean': float(np.mean(incorrect_n_jumps)),
        'd_jumps': float(d_jumps),
        'p_jumps': float(p_jumps),
        'correct_jump_layers': correct_jump_layers,
        'incorrect_jump_layers': incorrect_jump_layers
    }


def analyze_acceleration(trajectories, labels):
    """Analyze velocity, acceleration, jerk profiles."""
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    print("\n" + "="*70)
    print("ANALYSIS 3: Acceleration Profile")
    print("="*70)
    print("\nHypothesis: Correct solutions may accelerate/decelerate at specific layers")

    # Compute velocity at each layer (per sample)
    velocities = np.zeros((n_samples, n_layers - 1))
    for l in range(n_layers - 1):
        velocities[:, l] = np.linalg.norm(
            trajectories[:, :, l+1, :] - trajectories[:, :, l, :],
            axis=2
        ).mean(axis=1)

    # Acceleration (change in velocity)
    accelerations = np.diff(velocities, axis=1)

    # Jerk (change in acceleration)
    jerks = np.diff(accelerations, axis=1)

    # Per-layer acceleration comparison
    print(f"\n{'Transition':<12} {'Correct Acc':<12} {'Incorrect Acc':<12} {'Cohen d':<10} {'p-value':<10}")
    print("-"*55)

    acc_results = []
    for l in range(accelerations.shape[1]):
        correct_acc = accelerations[labels, l]
        incorrect_acc = accelerations[~labels, l]

        d = cohens_d(correct_acc, incorrect_acc)
        _, p = stats.ttest_ind(correct_acc, incorrect_acc)

        print(f"L{l}→L{l+2:<6} {np.mean(correct_acc):<12.2f} {np.mean(incorrect_acc):<12.2f} "
              f"{d:<10.3f} {p:<10.4f}")

        acc_results.append({
            'layer': l,
            'd': float(d),
            'p': float(p)
        })

    # Overall acceleration metrics
    mean_acc_correct = accelerations[labels].mean(axis=1)
    mean_acc_incorrect = accelerations[~labels].mean(axis=1)

    d_mean_acc = cohens_d(mean_acc_correct, mean_acc_incorrect)
    _, p_mean_acc = stats.ttest_ind(mean_acc_correct, mean_acc_incorrect)

    print(f"\nMean acceleration (overall speedup/slowdown):")
    print(f"  Correct: {np.mean(mean_acc_correct):.2f}, Incorrect: {np.mean(mean_acc_incorrect):.2f}")
    print(f"  Cohen's d: {d_mean_acc:.3f}, p-value: {p_mean_acc:.4f}")

    # Max jerk analysis
    max_jerk_layer_correct = np.argmax(np.abs(jerks[labels]).mean(axis=0))
    max_jerk_layer_incorrect = np.argmax(np.abs(jerks[~labels]).mean(axis=0))

    print(f"\nMax jerk (sudden shift) location:")
    print(f"  Correct: L{max_jerk_layer_correct}")
    print(f"  Incorrect: L{max_jerk_layer_incorrect}")

    # Jerk magnitude comparison
    max_jerk_correct = np.abs(jerks[labels]).max(axis=1)
    max_jerk_incorrect = np.abs(jerks[~labels]).max(axis=1)

    d_jerk = cohens_d(max_jerk_correct, max_jerk_incorrect)
    _, p_jerk = stats.ttest_ind(max_jerk_correct, max_jerk_incorrect)

    print(f"\nMax jerk magnitude:")
    print(f"  Correct: {np.mean(max_jerk_correct):.2f}, Incorrect: {np.mean(max_jerk_incorrect):.2f}")
    print(f"  Cohen's d: {d_jerk:.3f}, p-value: {p_jerk:.4f}")

    return {
        'per_layer_acc': acc_results,
        'd_mean_acc': float(d_mean_acc),
        'p_mean_acc': float(p_mean_acc),
        'max_jerk_layer_correct': int(max_jerk_layer_correct),
        'max_jerk_layer_incorrect': int(max_jerk_layer_incorrect),
        'd_max_jerk': float(d_jerk),
        'p_max_jerk': float(p_jerk)
    }


def analyze_dimensionality(trajectories, labels, use_randomized_svd=True):
    """Analyze effective dimensionality trajectory."""
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    print("\n" + "="*70)
    print("ANALYSIS 4: Dimensionality Trajectory")
    print("="*70)
    print("\nHypothesis: Correct solutions may compress to lower-dimensional subspaces")

    # Compute effective dimensionality at each layer for each sample
    # Use randomized SVD for speed
    d_eff = np.zeros((n_samples, n_layers))

    print("\nComputing effective dimensionality (this may take a minute)...")

    for i in range(n_samples):
        for l in range(n_layers):
            X = trajectories[i, :, l, :]  # (seq_len, d_model)

            if use_randomized_svd:
                # Randomized SVD for speed (k=100 components)
                from sklearn.utils.extmath import randomized_svd
                try:
                    _, S, _ = randomized_svd(X, n_components=min(100, seq_len, d_model), random_state=42)
                except:
                    _, S, _ = np.linalg.svd(X, full_matrices=False)
            else:
                _, S, _ = np.linalg.svd(X, full_matrices=False)

            # Participation ratio
            d_eff[i, l] = (S.sum())**2 / ((S**2).sum() + 1e-10)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_samples} samples")

    # Per-layer dimensionality comparison
    print(f"\n{'Layer':<8} {'Correct d_eff':<14} {'Incorrect d_eff':<16} {'Cohen d':<10} {'p-value':<10}")
    print("-"*60)

    dim_results = []
    for l in range(n_layers):
        correct_dim = d_eff[labels, l]
        incorrect_dim = d_eff[~labels, l]

        d = cohens_d(correct_dim, incorrect_dim)
        _, p = stats.ttest_ind(correct_dim, incorrect_dim)

        print(f"L{l:<7} {np.mean(correct_dim):<14.2f} {np.mean(incorrect_dim):<16.2f} "
              f"{d:<10.3f} {p:<10.4f}")

        dim_results.append({
            'layer': l,
            'd': float(d),
            'p': float(p),
            'correct_mean': float(np.mean(correct_dim)),
            'incorrect_mean': float(np.mean(incorrect_dim))
        })

    # Dimensionality trajectory features
    # Compression rate (first layer to last layer ratio)
    compression_correct = d_eff[labels, -1] / (d_eff[labels, 0] + 1e-10)
    compression_incorrect = d_eff[~labels, -1] / (d_eff[~labels, 0] + 1e-10)

    d_compression = cohens_d(compression_correct, compression_incorrect)
    _, p_compression = stats.ttest_ind(compression_correct, compression_incorrect)

    print(f"\nCompression rate (final / initial dimensionality):")
    print(f"  Correct: {np.mean(compression_correct):.3f}, Incorrect: {np.mean(compression_incorrect):.3f}")
    print(f"  Cohen's d: {d_compression:.3f}, p-value: {p_compression:.4f}")

    # Dimensionality jump detection
    dim_jumps_correct = np.abs(np.diff(d_eff[labels], axis=1)).max(axis=1)
    dim_jumps_incorrect = np.abs(np.diff(d_eff[~labels], axis=1)).max(axis=1)

    d_dim_jump = cohens_d(dim_jumps_correct, dim_jumps_incorrect)
    _, p_dim_jump = stats.ttest_ind(dim_jumps_correct, dim_jumps_incorrect)

    print(f"\nMax dimensionality jump magnitude:")
    print(f"  Correct: {np.mean(dim_jumps_correct):.2f}, Incorrect: {np.mean(dim_jumps_incorrect):.2f}")
    print(f"  Cohen's d: {d_dim_jump:.3f}, p-value: {p_dim_jump:.4f}")

    return {
        'per_layer_dim': dim_results,
        'd_compression': float(d_compression),
        'p_compression': float(p_compression),
        'd_dim_jump': float(d_dim_jump),
        'p_dim_jump': float(p_dim_jump),
        'correct_compression_mean': float(np.mean(compression_correct)),
        'incorrect_compression_mean': float(np.mean(compression_incorrect))
    }


def summarize_findings(step_results, jump_results, acc_results, dim_results):
    """Summarize key findings and cross-validate with prior results."""
    print("\n" + "="*70)
    print("SUMMARY: Layer-Localized Path Dynamics")
    print("="*70)

    # Find most discriminative layer for step magnitude
    best_step_layer = max(step_results, key=lambda x: abs(x['d']))

    print(f"""
KEY FINDINGS:

1. STEP MAGNITUDE:
   - Most discriminative layer: L{best_step_layer['layer']} (d={best_step_layer['d']:.3f}, p={best_step_layer['p']:.4f})
   - Direction: {'Correct takes bigger steps' if best_step_layer['d'] > 0 else 'Incorrect takes bigger steps'}

2. JUMP PATTERNS:
   - Correct avg jumps: {jump_results['correct_n_jumps_mean']:.2f}
   - Incorrect avg jumps: {jump_results['incorrect_n_jumps_mean']:.2f}
   - Cohen's d: {jump_results['d_jumps']:.3f}

3. ACCELERATION:
   - Mean acceleration d: {acc_results['d_mean_acc']:.3f}
   - Max jerk layer (correct): L{acc_results['max_jerk_layer_correct']}
   - Max jerk layer (incorrect): L{acc_results['max_jerk_layer_incorrect']}
   - Jerk magnitude d: {acc_results['d_max_jerk']:.3f}

4. DIMENSIONALITY:
   - Compression rate d: {dim_results['d_compression']:.3f}
   - Correct compression: {dim_results['correct_compression_mean']:.3f}
   - Incorrect compression: {dim_results['incorrect_compression_mean']:.3f}
""")

    print("CROSS-VALIDATION WITH PRIOR FINDINGS:")
    print("-"*50)
    print(f"  CKA peak at L7:        Step profile L{best_step_layer['layer']} {'ALIGNS' if abs(best_step_layer['layer'] - 7) <= 2 else 'DIFFERS'}")
    print(f"  Error direction L24-28: Need to check late layers")
    print(f"  Wynroe L16-18 spike:    Max jerk at L{acc_results['max_jerk_layer_correct']} (correct)")


def main():
    parser = argparse.ArgumentParser(description='Layer-Localized Path Dynamics Analysis')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='olmo3_rl_zero')
    parser.add_argument('--task', type=str, default='gsm8k')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--max-samples', type=int, default=100)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = data_dir / args.model / f"{args.task}_trajectories.h5"
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found")
        return

    print(f"\n{'='*70}")
    print("Layer-Localized Path Dynamics Analysis")
    print(f"{'='*70}")
    print(f"Model: {args.model}, Task: {args.task}")

    # Load data
    trajectories, labels = load_trajectories(h5_path, args.max_samples)
    n_samples = len(labels)
    n_correct = labels.sum()
    print(f"Loaded: {n_samples} samples ({n_correct} correct, {n_samples - n_correct} incorrect)")
    print(f"Trajectory shape: {trajectories.shape}")

    # Run analyses
    step_results = compute_per_layer_step_profile(trajectories, labels)
    jump_results = analyze_jumps(trajectories, labels)
    acc_results = analyze_acceleration(trajectories, labels)
    dim_results = analyze_dimensionality(trajectories, labels)

    # Summarize
    summarize_findings(step_results, jump_results, acc_results, dim_results)

    # Save results
    all_results = {
        'model': args.model,
        'task': args.task,
        'n_samples': int(n_samples),
        'n_correct': int(n_correct),
        'step_profile': step_results,
        'jump_analysis': jump_results,
        'acceleration': acc_results,
        'dimensionality': dim_results
    }

    output_file = output_dir / f'layer_path_dynamics_{args.model}_{args.task}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
