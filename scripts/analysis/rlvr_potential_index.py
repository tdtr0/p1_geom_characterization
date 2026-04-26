#!/usr/bin/env python3
"""
RLVR Potential Index - Predicting Model Readiness for RL Training

Computes metrics to predict how ready a base model is for RLVR training:
- LCS: Latent Capability Score (probe AUC on base model)
- GAS: Geometric Alignment Score (error direction alignment)
- ID: Intrinsic Dimensionality (effective rank)
- TER: Training Efficiency Ratio (Δacc / (1 - CKA))
- RRS: RLVR Readiness Score (composite)

Usage:
    python rlvr_potential_index.py \
        --data-dir /data/thanhdo/trajectories_0shot \
        --output-dir results
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


# =============================================================================
# DATA LOADING
# =============================================================================

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


# =============================================================================
# CORE METRICS
# =============================================================================

def compute_effective_rank(activations, eps=1e-10):
    """
    Effective rank: exponential of entropy of normalized singular values.

    Measures how many dimensions are "actively used" in the representation.
    """
    _, s, _ = svd(activations, full_matrices=False)
    s_normalized = s / (s.sum() + eps)
    s_normalized = s_normalized[s_normalized > eps]

    if len(s_normalized) == 0:
        return 1.0

    return np.exp(entropy(s_normalized))


def compute_cka(X, Y):
    """
    Compute Centered Kernel Alignment (CKA) between two activation matrices.

    CKA is invariant to orthogonal transformations and isotropic scaling.
    """
    def center_gram(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K_X = X @ X.T
    K_Y = Y @ Y.T

    K_X_centered = center_gram(K_X)
    K_Y_centered = center_gram(K_Y)

    hsic = np.sum(K_X_centered * K_Y_centered)
    var_x = np.sqrt(np.sum(K_X_centered * K_X_centered))
    var_y = np.sqrt(np.sum(K_Y_centered * K_Y_centered))

    if var_x * var_y == 0:
        return 0.0

    return hsic / (var_x * var_y)


def compute_error_direction(activations, labels):
    """Compute error direction as difference in means (incorrect - correct)."""
    if labels.sum() == 0 or (~labels).sum() == 0:
        return np.zeros(activations.shape[1])

    correct_mean = activations[labels].mean(axis=0)
    incorrect_mean = activations[~labels].mean(axis=0)
    error_dir = incorrect_mean - correct_mean
    norm = np.linalg.norm(error_dir)
    if norm > 1e-10:
        error_dir = error_dir / norm
    return error_dir


# =============================================================================
# RLVR POTENTIAL INDEX COMPONENTS
# =============================================================================

def compute_latent_capability_score(activations, labels, n_cv=5):
    """
    Latent Capability Score (LCS): Probe AUC on base model.

    Measures how much the base model "already knows" about the task.
    Range: [0.5, 1.0] where 0.5 = random, 1.0 = perfect separability.
    """
    if labels.sum() == 0 or (~labels).sum() == 0:
        return 0.5  # No signal if all same class

    try:
        clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
        scores = cross_val_score(clf, activations, labels, cv=n_cv, scoring='roc_auc')
        return float(np.mean(scores))
    except Exception as e:
        print(f"    Warning: LCS computation failed: {e}")
        return 0.5


def compute_geometric_alignment_score(error_dir_base, error_dir_trained):
    """
    Geometric Alignment Score (GAS): Cosine similarity between error directions.

    Measures how well the base geometry predicts where training will go.
    Range: [0, 1] where 1 = perfect alignment.
    """
    cos_sim = np.abs(np.dot(error_dir_base, error_dir_trained))
    return float(cos_sim)


def compute_intrinsic_dimensionality(activations):
    """
    Intrinsic Dimensionality (ID): Effective rank of activation subspace.

    Lower = more compressed/structured representation.
    """
    return float(compute_effective_rank(activations))


def compute_training_efficiency_ratio(acc_base, acc_trained, cka):
    """
    Training Efficiency Ratio (TER): Accuracy gain per unit of representation change.

    TER = Δaccuracy / (1 - CKA)
    Higher = more efficient training (big acc gain with small geometry change).
    """
    delta_acc = acc_trained - acc_base
    geometry_change = 1 - cka

    if geometry_change < 0.01:  # Very small change
        return float('inf') if delta_acc > 0 else 0.0

    return float(delta_acc / geometry_change)


def compute_rlvr_readiness_score(lcs, gas, id_score):
    """
    RLVR Readiness Score (RRS): Composite metric.

    RRS = LCS × GAS × (1 / log(ID))

    Higher = model is more "ready" for RLVR training.
    """
    # Normalize ID contribution: lower ID is better, so use 1/log(ID)
    id_factor = 1.0 / np.log(max(id_score, 2.0))  # Avoid log(1) = 0

    rrs = lcs * gas * id_factor
    return float(rrs)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_task(data_dir, task, models, max_samples=200):
    """Run RLVR Potential Index analysis for a single task."""
    data_dir = Path(data_dir)
    results = {'task': task, 'models': {}}

    # First, load base model data
    base_path = data_dir / 'olmo3_base' / f'{task}_trajectories.h5'
    if not base_path.exists():
        print(f"  SKIP: base model data not found at {base_path}")
        return None

    print(f"\n  Loading base model ({task})...")
    try:
        traj_base, labels_base = load_trajectories(base_path, max_samples)
    except Exception as e:
        print(f"  ERROR loading base: {e}")
        return None

    act_base = get_mean_activations(traj_base, layer=-1)
    n_correct_base = int(labels_base.sum())
    n_total_base = len(labels_base)
    acc_base = n_correct_base / n_total_base

    print(f"    Base: {n_correct_base}/{n_total_base} correct ({acc_base:.1%})")

    # Compute base model metrics
    print(f"  Computing base model metrics...")
    lcs = compute_latent_capability_score(act_base, labels_base)
    id_base = compute_intrinsic_dimensionality(act_base)
    error_dir_base = compute_error_direction(act_base, labels_base)

    print(f"    LCS (Latent Capability): {lcs:.4f}")
    print(f"    ID (Intrinsic Dimensionality): {id_base:.2f}")

    results['base'] = {
        'accuracy': acc_base,
        'n_correct': n_correct_base,
        'n_total': n_total_base,
        'lcs': lcs,
        'intrinsic_dimensionality': id_base
    }

    # Now analyze each trained model
    trained_models = ['olmo3_rl_zero', 'olmo3_sft', 'olmo3_think']

    for model in trained_models:
        model_path = data_dir / model / f'{task}_trajectories.h5'
        if not model_path.exists():
            print(f"\n  SKIP: {model} not found")
            continue

        print(f"\n  Analyzing {model}...")
        try:
            traj_trained, labels_trained = load_trajectories(model_path, max_samples)
        except Exception as e:
            print(f"    ERROR loading: {e}")
            continue

        # Match sample counts
        n_samples = min(len(act_base), len(traj_trained))
        act_trained = get_mean_activations(traj_trained[:n_samples], layer=-1)
        labels_trained_matched = labels_trained[:n_samples]
        act_base_matched = act_base[:n_samples]
        labels_base_matched = labels_base[:n_samples]

        n_correct_trained = int(labels_trained_matched.sum())
        n_total_trained = len(labels_trained_matched)
        acc_trained = n_correct_trained / n_total_trained

        print(f"    Accuracy: {n_correct_trained}/{n_total_trained} ({acc_trained:.1%})")

        # Compute trained model error direction
        error_dir_trained = compute_error_direction(act_trained, labels_trained_matched)

        # Compute GAS (error_dir is already (d_model,) vector, no slicing needed)
        gas = compute_geometric_alignment_score(error_dir_base, error_dir_trained)
        print(f"    GAS (Geometric Alignment): {gas:.4f}")

        # Compute CKA
        cka = compute_cka(act_base_matched, act_trained)
        print(f"    CKA (Representation Similarity): {cka:.4f}")

        # Compute TER
        ter = compute_training_efficiency_ratio(acc_base, acc_trained, cka)
        print(f"    TER (Training Efficiency): {ter:.4f}")

        # Compute RRS
        rrs = compute_rlvr_readiness_score(lcs, gas, id_base)
        print(f"    RRS (RLVR Readiness Score): {rrs:.4f}")

        # Store results
        results['models'][model] = {
            'accuracy': acc_trained,
            'n_correct': n_correct_trained,
            'n_total': n_total_trained,
            'accuracy_delta': acc_trained - acc_base,
            'gas': gas,
            'cka': cka,
            'ter': ter,
            'rrs': rrs
        }

    return results


def generate_report(all_results, output_dir):
    """Generate markdown report from results."""
    output_dir = Path(output_dir)

    report = []
    report.append("# RLVR Potential Index Report")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Metrics Explained")
    report.append("")
    report.append("| Metric | Description | Range | Interpretation |")
    report.append("|--------|-------------|-------|----------------|")
    report.append("| **LCS** | Latent Capability Score | [0.5, 1.0] | How much base model 'knows' about task |")
    report.append("| **GAS** | Geometric Alignment Score | [0, 1] | Does base geometry predict training direction? |")
    report.append("| **ID** | Intrinsic Dimensionality | [1, d_model] | Complexity of representation (lower = better) |")
    report.append("| **CKA** | Centered Kernel Alignment | [0, 1] | Representation similarity base → trained |")
    report.append("| **TER** | Training Efficiency Ratio | [0, ∞) | Accuracy gain per geometry change |")
    report.append("| **RRS** | RLVR Readiness Score | [0, ∞) | Composite: LCS × GAS × (1/log(ID)) |")
    report.append("")

    # Summary table
    report.append("## Summary by Task")
    report.append("")

    for task_results in all_results:
        if task_results is None:
            continue

        task = task_results['task']
        base = task_results['base']

        report.append(f"### {task.upper()}")
        report.append("")
        report.append(f"**Base Model (OLMo-3-Base)**:")
        report.append(f"- Accuracy: {base['accuracy']:.1%} ({base['n_correct']}/{base['n_total']})")
        report.append(f"- Latent Capability Score (LCS): **{base['lcs']:.4f}**")
        report.append(f"- Intrinsic Dimensionality (ID): **{base['intrinsic_dimensionality']:.2f}**")
        report.append("")

        if task_results['models']:
            report.append("| Trained Model | Accuracy | Δ Acc | GAS | CKA | TER | RRS |")
            report.append("|---------------|----------|-------|-----|-----|-----|-----|")

            for model, metrics in task_results['models'].items():
                model_short = model.replace('olmo3_', '')
                report.append(
                    f"| {model_short} | {metrics['accuracy']:.1%} | "
                    f"{metrics['accuracy_delta']:+.1%} | "
                    f"{metrics['gas']:.3f} | {metrics['cka']:.3f} | "
                    f"{metrics['ter']:.3f} | **{metrics['rrs']:.4f}** |"
                )
            report.append("")

    # Interpretation
    report.append("## Interpretation")
    report.append("")

    # Find best RRS across all tasks/models
    best_rrs = 0
    best_combo = None
    for task_results in all_results:
        if task_results is None:
            continue
        task = task_results['task']
        for model, metrics in task_results['models'].items():
            if metrics['rrs'] > best_rrs:
                best_rrs = metrics['rrs']
                best_combo = (task, model)

    if best_combo:
        report.append(f"**Highest RLVR Readiness**: {best_combo[1]} on {best_combo[0]} (RRS = {best_rrs:.4f})")
        report.append("")

    # Key findings
    report.append("### Key Findings")
    report.append("")

    for task_results in all_results:
        if task_results is None:
            continue
        task = task_results['task']
        base = task_results['base']

        report.append(f"**{task.upper()}**:")
        report.append(f"- Base model LCS = {base['lcs']:.3f} → {'Strong' if base['lcs'] > 0.65 else 'Moderate' if base['lcs'] > 0.55 else 'Weak'} latent capability")

        # Compare RL-Zero vs SFT
        rl_zero = task_results['models'].get('olmo3_rl_zero', {})
        sft = task_results['models'].get('olmo3_sft', {})

        if rl_zero and sft:
            if rl_zero['gas'] > sft['gas']:
                report.append(f"- RL-Zero has HIGHER GAS ({rl_zero['gas']:.3f}) than SFT ({sft['gas']:.3f}) → RL preserves base geometry better")
            else:
                report.append(f"- SFT has HIGHER GAS ({sft['gas']:.3f}) than RL-Zero ({rl_zero['gas']:.3f}) → SFT preserves base geometry better")

            if rl_zero['ter'] > sft['ter']:
                report.append(f"- RL-Zero is MORE EFFICIENT (TER={rl_zero['ter']:.3f}) than SFT (TER={sft['ter']:.3f})")
            else:
                report.append(f"- SFT is MORE EFFICIENT (TER={sft['ter']:.3f}) than RL-Zero (TER={rl_zero['ter']:.3f})")

        report.append("")

    # Write report
    report_path = output_dir / 'rlvr_potential_index.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nSaved report to {report_path}")
    return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description='RLVR Potential Index Analysis')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing trajectory HDF5 files')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--tasks', type=str, default='gsm8k,humaneval,logiqa',
                        help='Comma-separated list of tasks')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Maximum samples per model/task')

    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RLVR POTENTIAL INDEX ANALYSIS")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Tasks: {tasks}")
    print(f"Max samples: {args.max_samples}")
    print("=" * 70)

    all_results = []

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        result = analyze_task(args.data_dir, task, [], args.max_samples)
        if result:
            all_results.append(result)

    # Save JSON results
    json_path = output_dir / 'rlvr_potential_index.json'
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data_dir': args.data_dir,
                'tasks': tasks,
                'max_samples': args.max_samples
            },
            'results': all_results
        }, f, indent=2)
    print(f"\nSaved JSON to {json_path}")

    # Generate report
    report = generate_report(all_results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(report)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
