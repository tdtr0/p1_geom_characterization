#!/usr/bin/env python3
"""
Paired Trajectory Divergence Analysis

Compare same question across different models to find WHERE paths diverge.
Key insight: seed=42 ensures sample index i is the same question across all models.

Analyses:
1. Cross-model layer-by-layer divergence (same question, different models)
2. Discordant pair analysis (model A correct, model B wrong on same question)
3. Neuron activation patterns (sparsity, top-K active)
4. Token-position divergence

Usage:
    python paired_trajectory_divergence.py \
        --data-dir data/trajectories_0shot \
        --model-a olmo3_base \
        --model-b olmo3_sft \
        --task gsm8k
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
            trajectories = trajectories[:max_samples]
            labels = labels[:max_samples]

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


def compute_cross_model_divergence(traj_A, traj_B):
    """Compute layer-by-layer divergence between two trajectories (same question)."""
    seq_len, n_layers, d_model = traj_A.shape

    cos_sim = np.zeros(n_layers)
    distance = np.zeros(n_layers)

    for l in range(n_layers):
        # Mean-pool over tokens
        act_A = traj_A[:, l, :].mean(axis=0)  # (d_model,)
        act_B = traj_B[:, l, :].mean(axis=0)

        # Cosine similarity
        norm_A = np.linalg.norm(act_A)
        norm_B = np.linalg.norm(act_B)
        if norm_A > 1e-10 and norm_B > 1e-10:
            cos_sim[l] = np.dot(act_A, act_B) / (norm_A * norm_B)
        else:
            cos_sim[l] = 0.0

        # Euclidean distance
        distance[l] = np.linalg.norm(act_A - act_B)

    return cos_sim, distance


def find_discordant_pairs(labels_A, labels_B):
    """Find questions where models disagree on correctness."""
    discordant = labels_A != labels_B
    A_correct = labels_A & ~labels_B  # A correct, B wrong
    B_correct = ~labels_A & labels_B  # B correct, A wrong

    return {
        'discordant_indices': np.where(discordant)[0],
        'A_correct_indices': np.where(A_correct)[0],
        'B_correct_indices': np.where(B_correct)[0],
        'n_discordant': int(discordant.sum()),
        'agreement_rate': float((~discordant).mean())
    }


def compute_divergence_onset(cos_sim_profile, threshold=0.95):
    """Find layer where similarity drops below threshold."""
    below_threshold = cos_sim_profile < threshold
    if below_threshold.any():
        return int(np.argmax(below_threshold))  # First layer below threshold
    return -1  # Never diverges


def compute_activation_sparsity(trajectory, percentile=90):
    """Compute how many neurons are 'active' at each layer."""
    seq_len, n_layers, d_model = trajectory.shape

    sparsity = np.zeros(n_layers)
    top_neurons = []

    for l in range(n_layers):
        activation = trajectory[:, l, :]  # (seq_len, d_model)
        magnitude = np.abs(activation).mean(axis=0)  # (d_model,)
        threshold = np.percentile(magnitude, percentile)
        active_mask = magnitude > threshold
        sparsity[l] = active_mask.sum() / d_model

        # Top 10 neurons by activation magnitude
        top_k = np.argsort(magnitude)[-10:][::-1]
        top_neurons.append(top_k.tolist())

    return sparsity, top_neurons


def analyze_layer_divergence(traj_A, traj_B, labels_A, labels_B):
    """Main analysis: layer-by-layer divergence across all samples."""
    n_samples = min(len(traj_A), len(traj_B))
    n_layers = traj_A.shape[2]

    print("\n" + "="*70)
    print("ANALYSIS 1: Cross-Model Layer-by-Layer Divergence")
    print("="*70)

    # Compute divergence for all samples
    all_cos_sim = np.zeros((n_samples, n_layers))
    all_distance = np.zeros((n_samples, n_layers))

    for i in range(n_samples):
        cos_sim, distance = compute_cross_model_divergence(traj_A[i], traj_B[i])
        all_cos_sim[i] = cos_sim
        all_distance[i] = distance

    # Summary statistics per layer
    print(f"\n{'Layer':<8} {'Mean Cos Sim':<14} {'Std':<10} {'Mean Dist':<12} {'Std':<10}")
    print("-"*55)

    layer_stats = []
    for l in range(n_layers):
        mean_cos = all_cos_sim[:, l].mean()
        std_cos = all_cos_sim[:, l].std()
        mean_dist = all_distance[:, l].mean()
        std_dist = all_distance[:, l].std()

        print(f"L{l:<7} {mean_cos:<14.4f} {std_cos:<10.4f} {mean_dist:<12.2f} {std_dist:<10.2f}")

        layer_stats.append({
            'layer': l,
            'mean_cos_sim': float(mean_cos),
            'std_cos_sim': float(std_cos),
            'mean_distance': float(mean_dist),
            'std_distance': float(std_dist)
        })

    # Find divergence onset for each sample
    divergence_onsets = []
    for i in range(n_samples):
        onset = compute_divergence_onset(all_cos_sim[i], threshold=0.95)
        divergence_onsets.append(onset)

    print(f"\nDivergence onset layer (cos_sim < 0.95):")
    print(f"  Mean: {np.mean([x for x in divergence_onsets if x >= 0]):.1f}")
    print(f"  Never diverges: {(np.array(divergence_onsets) == -1).sum()} samples")

    return {
        'layer_stats': layer_stats,
        'all_cos_sim': all_cos_sim,
        'all_distance': all_distance,
        'divergence_onsets': divergence_onsets
    }


def analyze_discordant_pairs(traj_A, traj_B, labels_A, labels_B):
    """Analyze samples where models disagree on correctness."""
    n_samples = min(len(traj_A), len(traj_B))
    n_layers = traj_A.shape[2]

    print("\n" + "="*70)
    print("ANALYSIS 2: Discordant Pair Analysis")
    print("="*70)

    # Find discordant pairs
    discordant_info = find_discordant_pairs(labels_A[:n_samples], labels_B[:n_samples])

    print(f"\nTotal samples: {n_samples}")
    print(f"Model A correct: {labels_A[:n_samples].sum()}")
    print(f"Model B correct: {labels_B[:n_samples].sum()}")
    print(f"Discordant pairs: {discordant_info['n_discordant']}")
    print(f"Agreement rate: {discordant_info['agreement_rate']:.1%}")
    print(f"  - A correct, B wrong: {len(discordant_info['A_correct_indices'])}")
    print(f"  - B correct, A wrong: {len(discordant_info['B_correct_indices'])}")

    if discordant_info['n_discordant'] == 0:
        print("\nNo discordant pairs found!")
        return discordant_info

    # Analyze divergence for discordant pairs
    discordant_idx = discordant_info['discordant_indices']

    # Compute layer-wise divergence for discordant vs concordant
    discordant_cos_sim = np.zeros((len(discordant_idx), n_layers))
    for j, i in enumerate(discordant_idx):
        cos_sim, _ = compute_cross_model_divergence(traj_A[i], traj_B[i])
        discordant_cos_sim[j] = cos_sim

    concordant_idx = [i for i in range(n_samples) if i not in discordant_idx]
    concordant_cos_sim = np.zeros((len(concordant_idx), n_layers))
    for j, i in enumerate(concordant_idx):
        cos_sim, _ = compute_cross_model_divergence(traj_A[i], traj_B[i])
        concordant_cos_sim[j] = cos_sim

    # Compare discordant vs concordant divergence per layer
    print(f"\n{'Layer':<8} {'Discordant Cos':<16} {'Concordant Cos':<16} {'Cohen d':<10} {'p-value':<10}")
    print("-"*65)

    comparison_stats = []
    for l in range(n_layers):
        d = cohens_d(discordant_cos_sim[:, l], concordant_cos_sim[:, l])
        if len(discordant_cos_sim) > 1 and len(concordant_cos_sim) > 1:
            _, p = stats.ttest_ind(discordant_cos_sim[:, l], concordant_cos_sim[:, l])
        else:
            p = 1.0

        sig = "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""

        print(f"L{l:<7} {discordant_cos_sim[:, l].mean():<16.4f} "
              f"{concordant_cos_sim[:, l].mean():<16.4f} "
              f"{d:<10.3f} {p:<10.4f} {sig}")

        comparison_stats.append({
            'layer': l,
            'discordant_mean': float(discordant_cos_sim[:, l].mean()),
            'concordant_mean': float(concordant_cos_sim[:, l].mean()),
            'd': float(d),
            'p': float(p)
        })

    discordant_info['comparison_stats'] = comparison_stats
    return discordant_info


def analyze_neuron_patterns(traj_A, traj_B, labels_A, labels_B):
    """Analyze neuron activation patterns."""
    n_samples = min(len(traj_A), len(traj_B))
    n_layers = traj_A.shape[2]

    print("\n" + "="*70)
    print("ANALYSIS 3: Neuron Activation Patterns")
    print("="*70)

    # Compute sparsity for correct vs incorrect samples in each model
    print("\nActivation sparsity (top 10% active neurons):")

    # Model A
    correct_A_sparsity = []
    incorrect_A_sparsity = []
    for i in range(n_samples):
        sparsity, _ = compute_activation_sparsity(traj_A[i])
        if labels_A[i]:
            correct_A_sparsity.append(sparsity)
        else:
            incorrect_A_sparsity.append(sparsity)

    if len(correct_A_sparsity) > 0 and len(incorrect_A_sparsity) > 0:
        correct_A_sparsity = np.array(correct_A_sparsity)
        incorrect_A_sparsity = np.array(incorrect_A_sparsity)

        print(f"\nModel A - Sparsity by correctness:")
        print(f"{'Layer':<8} {'Correct':<12} {'Incorrect':<12} {'Cohen d':<10}")
        print("-"*45)

        sparsity_stats_A = []
        for l in range(n_layers):
            d = cohens_d(correct_A_sparsity[:, l], incorrect_A_sparsity[:, l])
            print(f"L{l:<7} {correct_A_sparsity[:, l].mean():<12.4f} "
                  f"{incorrect_A_sparsity[:, l].mean():<12.4f} {d:<10.3f}")
            sparsity_stats_A.append({
                'layer': l,
                'correct_mean': float(correct_A_sparsity[:, l].mean()),
                'incorrect_mean': float(incorrect_A_sparsity[:, l].mean()),
                'd': float(d)
            })
    else:
        sparsity_stats_A = []

    # Model B
    correct_B_sparsity = []
    incorrect_B_sparsity = []
    for i in range(n_samples):
        sparsity, _ = compute_activation_sparsity(traj_B[i])
        if labels_B[i]:
            correct_B_sparsity.append(sparsity)
        else:
            incorrect_B_sparsity.append(sparsity)

    if len(correct_B_sparsity) > 0 and len(incorrect_B_sparsity) > 0:
        correct_B_sparsity = np.array(correct_B_sparsity)
        incorrect_B_sparsity = np.array(incorrect_B_sparsity)

        print(f"\nModel B - Sparsity by correctness:")
        print(f"{'Layer':<8} {'Correct':<12} {'Incorrect':<12} {'Cohen d':<10}")
        print("-"*45)

        sparsity_stats_B = []
        for l in range(n_layers):
            d = cohens_d(correct_B_sparsity[:, l], incorrect_B_sparsity[:, l])
            print(f"L{l:<7} {correct_B_sparsity[:, l].mean():<12.4f} "
                  f"{incorrect_B_sparsity[:, l].mean():<12.4f} {d:<10.3f}")
            sparsity_stats_B.append({
                'layer': l,
                'correct_mean': float(correct_B_sparsity[:, l].mean()),
                'incorrect_mean': float(incorrect_B_sparsity[:, l].mean()),
                'd': float(d)
            })
    else:
        sparsity_stats_B = []

    return {
        'sparsity_stats_A': sparsity_stats_A,
        'sparsity_stats_B': sparsity_stats_B
    }


def analyze_token_divergence(traj_A, labels_A, target_layer=7):
    """Find which token positions show correctness signal."""
    n_samples, seq_len, n_layers, d_model = traj_A.shape

    print("\n" + "="*70)
    print(f"ANALYSIS 4: Token-Position Divergence (Layer {target_layer})")
    print("="*70)

    # Compute signal at each token position
    d_by_token = np.zeros(seq_len)
    for token in range(seq_len):
        act = traj_A[:, token, target_layer, :]  # (n_samples, d_model)
        act_norm = np.linalg.norm(act, axis=1)  # (n_samples,)

        correct_norm = act_norm[labels_A]
        incorrect_norm = act_norm[~labels_A]

        if len(correct_norm) > 1 and len(incorrect_norm) > 1:
            d_by_token[token] = cohens_d(correct_norm, incorrect_norm)

    # Find peak signal positions
    peak_positions = np.argsort(np.abs(d_by_token))[-10:][::-1]

    print(f"\nTop 10 token positions by |Cohen's d|:")
    print(f"{'Token':<8} {'Cohen d':<12} {'Position %':<12}")
    print("-"*35)

    for pos in peak_positions:
        print(f"T{pos:<7} {d_by_token[pos]:<12.3f} {100*pos/seq_len:<12.1f}%")

    # Aggregate by position range
    early = d_by_token[:seq_len//3].mean()
    middle = d_by_token[seq_len//3:2*seq_len//3].mean()
    late = d_by_token[2*seq_len//3:].mean()

    print(f"\nSignal by position range:")
    print(f"  Early (0-33%): d = {early:.3f}")
    print(f"  Middle (33-66%): d = {middle:.3f}")
    print(f"  Late (66-100%): d = {late:.3f}")

    return {
        'd_by_token': d_by_token.tolist(),
        'peak_positions': peak_positions.tolist(),
        'early_mean_d': float(early),
        'middle_mean_d': float(middle),
        'late_mean_d': float(late)
    }


def main():
    parser = argparse.ArgumentParser(description='Paired Trajectory Divergence Analysis')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--model-a', type=str, default='olmo3_base')
    parser.add_argument('--model-b', type=str, default='olmo3_sft')
    parser.add_argument('--task', type=str, default='gsm8k')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--max-samples', type=int, default=100)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load both models' trajectories
    path_A = data_dir / args.model_a / f"{args.task}_trajectories.h5"
    path_B = data_dir / args.model_b / f"{args.task}_trajectories.h5"

    if not path_A.exists():
        print(f"ERROR: {path_A} not found")
        return
    if not path_B.exists():
        print(f"ERROR: {path_B} not found")
        return

    print(f"\n{'='*70}")
    print("Paired Trajectory Divergence Analysis")
    print(f"{'='*70}")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print(f"Task: {args.task}")

    # Load data
    traj_A, labels_A = load_trajectories(path_A, args.max_samples)
    traj_B, labels_B = load_trajectories(path_B, args.max_samples)

    n_samples = min(len(traj_A), len(traj_B))
    print(f"\nLoaded: {n_samples} matched samples (same questions)")
    print(f"Model A ({args.model_a}): {labels_A[:n_samples].sum()} correct")
    print(f"Model B ({args.model_b}): {labels_B[:n_samples].sum()} correct")
    print(f"Trajectory shape: {traj_A.shape}")

    # Run analyses
    all_results = {
        'model_a': args.model_a,
        'model_b': args.model_b,
        'task': args.task,
        'n_samples': int(n_samples),
        'n_correct_A': int(labels_A[:n_samples].sum()),
        'n_correct_B': int(labels_B[:n_samples].sum())
    }

    # Analysis 1: Layer-by-layer divergence
    divergence_results = analyze_layer_divergence(traj_A, traj_B, labels_A, labels_B)
    all_results['layer_divergence'] = divergence_results['layer_stats']
    all_results['divergence_onsets'] = divergence_results['divergence_onsets']

    # Analysis 2: Discordant pairs
    discordant_results = analyze_discordant_pairs(traj_A, traj_B, labels_A, labels_B)
    all_results['discordant_analysis'] = discordant_results

    # Analysis 3: Neuron patterns
    neuron_results = analyze_neuron_patterns(traj_A, traj_B, labels_A, labels_B)
    all_results['neuron_patterns'] = neuron_results

    # Analysis 4: Token divergence
    token_results = analyze_token_divergence(traj_A, labels_A, target_layer=7)
    all_results['token_divergence'] = token_results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"""
Key Findings:

1. LAYER DIVERGENCE:
   - Models start similar (L0 cos_sim ≈ {divergence_results['layer_stats'][0]['mean_cos_sim']:.3f})
   - Final layer cos_sim ≈ {divergence_results['layer_stats'][-1]['mean_cos_sim']:.3f}

2. DISCORDANT PAIRS:
   - {discordant_results['n_discordant']} questions where models disagree
   - Agreement rate: {discordant_results['agreement_rate']:.1%}

3. TOKEN POSITION:
   - Early tokens: d = {token_results['early_mean_d']:.3f}
   - Middle tokens: d = {token_results['middle_mean_d']:.3f}
   - Late tokens: d = {token_results['late_mean_d']:.3f}
""")

    # Save results
    output_file = output_dir / f'paired_divergence_{args.model_a}_{args.model_b}_{args.task}.json'

    # Convert numpy arrays to lists for JSON serialization
    save_results = {}
    for k, v in all_results.items():
        if isinstance(v, np.ndarray):
            save_results[k] = v.tolist()
        elif isinstance(v, dict):
            save_results[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, np.ndarray):
                    save_results[k][k2] = v2.tolist()
                else:
                    save_results[k][k2] = v2
        else:
            save_results[k] = v

    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
