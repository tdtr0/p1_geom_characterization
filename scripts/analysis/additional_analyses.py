#!/usr/bin/env python3
"""
Additional analyses for surface structure vs reasoning investigation.

1. Path Signatures - Higher-order trajectory structure via iterated integrals
2. Token-Position Specificity - Where does the error signal peak in the sequence?
3. Cross-Domain Subspace Alignment - Do error directions align across tasks?

These analyses address whether H_jac2 detects surface format or actual reasoning.
"""

import argparse
import json
import numpy as np
import h5py
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Try to import signatory for path signatures
try:
    import signatory
    import torch
    HAS_SIGNATORY = True
except ImportError:
    HAS_SIGNATORY = False
    print("Warning: signatory not installed. Path signature analysis will be skipped.")
    print("To install: pip install signatory==1.2.6.1.9.0")


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def load_trajectories(filepath, max_samples=None):
    """Load trajectories and labels from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        traj = f['trajectories'][:]
        labels = f['is_correct'][:]

    if max_samples and len(traj) > max_samples:
        indices = np.random.RandomState(42).choice(len(traj), max_samples, replace=False)
        traj = traj[indices]
        labels = labels[indices]

    return traj.astype(np.float32), labels.astype(bool)


# ============================================================================
# ANALYSIS 1: PATH SIGNATURES
# ============================================================================

def compute_path_signature(trajectory, depth=3):
    """
    Compute path signature of a trajectory using signatory library.

    Path signatures are reparameterization-invariant features that capture
    the "shape" of a path, including higher-order interactions.

    Args:
        trajectory: (n_layers, d_model) - one trajectory through layers
        depth: signature depth (higher = more features but slower)

    Returns:
        signature: 1D array of signature features
    """
    if not HAS_SIGNATORY:
        return None

    # signatory expects (batch, length, channels)
    # We'll use PCA to reduce d_model first (signatory is slow for high dim)
    pca = PCA(n_components=min(32, trajectory.shape[1]))
    traj_reduced = pca.fit_transform(trajectory)  # (n_layers, 32)

    # Convert to torch tensor with batch dim
    traj_tensor = torch.tensor(traj_reduced, dtype=torch.float32).unsqueeze(0)

    # Compute signature
    sig = signatory.signature(traj_tensor, depth=depth)

    return sig.squeeze().numpy()


def path_signature_analysis(trajectories, labels, depth=3):
    """
    Analyze path signatures for correct vs incorrect trajectories.

    Tests whether higher-order trajectory structure differs.
    """
    if not HAS_SIGNATORY:
        return {'error': 'signatory not installed'}

    n_samples, seq_len, n_layers, d_model = trajectories.shape

    print(f"      Computing path signatures (depth={depth})...")

    # Average over sequence tokens first
    traj_mean = trajectories.mean(axis=1)  # (n_samples, n_layers, d_model)

    signatures = []
    for i in range(n_samples):
        sig = compute_path_signature(traj_mean[i], depth=depth)
        if sig is not None:
            signatures.append(sig)
        else:
            return {'error': 'signature computation failed'}

    signatures = np.array(signatures)
    print(f"      Signature shape: {signatures.shape}")

    # Compare correct vs incorrect using signature norm
    sig_norms = np.linalg.norm(signatures, axis=1)
    correct_norms = sig_norms[labels]
    incorrect_norms = sig_norms[~labels]

    d_norm = cohens_d(incorrect_norms, correct_norms)
    _, p_norm = stats.mannwhitneyu(incorrect_norms, correct_norms, alternative='two-sided')

    # Train classifier on signatures
    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, signatures, labels, cv=5, scoring='roc_auc')

    results = {
        'signature_depth': depth,
        'signature_dim': signatures.shape[1],
        'norm_effect_size': {
            'd': float(d_norm),
            'p': float(p_norm),
            'correct_mean': float(correct_norms.mean()),
            'incorrect_mean': float(incorrect_norms.mean())
        },
        'classifier_auc': {
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std()),
            'scores': cv_scores.tolist()
        },
        'n_correct': int(labels.sum()),
        'n_incorrect': int((~labels).sum())
    }

    return results


# ============================================================================
# ANALYSIS 2: TOKEN-POSITION SPECIFICITY
# ============================================================================

def compute_token_error_signal(trajectories, labels, n_folds=5):
    """
    Compute where the error signal peaks across token positions.

    For each token position, compute the discriminability (d') between
    correct and incorrect trajectories. This shows whether the signal
    is in reasoning tokens vs answer tokens.

    Uses cross-validation to avoid circularity.
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    print(f"      Computing per-token error signal (CV)...")

    # For each token position, compute the projection onto error direction
    # and measure discriminability

    token_d_values = np.zeros(seq_len)
    token_p_values = np.zeros(seq_len)

    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n_samples // n_folds

    # Accumulate CV predictions for each token
    cv_projections = np.zeros((n_samples, seq_len))

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        train_labels = labels[train_idx]
        train_traj = trajectories[train_idx]

        # Skip if insufficient samples
        if train_labels.sum() < 3 or (~train_labels).sum() < 3:
            continue

        # Compute error direction from train set (using last layer, all tokens)
        train_correct = train_traj[train_labels]
        train_incorrect = train_traj[~train_labels]

        # Error direction per token position
        for t in range(seq_len):
            # Use last layer for direction
            correct_act = train_correct[:, t, -1, :]  # (n_correct, d_model)
            incorrect_act = train_incorrect[:, t, -1, :]  # (n_incorrect, d_model)

            error_dir = incorrect_act.mean(axis=0) - correct_act.mean(axis=0)
            error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)

            # Project test samples
            for idx in test_idx:
                test_act = trajectories[idx, t, -1, :]
                cv_projections[idx, t] = test_act @ error_dir

    # Now compute d' for each token position
    for t in range(seq_len):
        correct_proj = cv_projections[labels, t]
        incorrect_proj = cv_projections[~labels, t]

        token_d_values[t] = cohens_d(incorrect_proj, correct_proj)
        _, token_p_values[t] = stats.mannwhitneyu(
            incorrect_proj, correct_proj, alternative='two-sided'
        )

    # Find peak positions
    peak_idx = np.argmax(np.abs(token_d_values))

    # Categorize: early (first 25%), middle (25-75%), late (last 25%)
    early_d = np.abs(token_d_values[:seq_len//4]).mean()
    middle_d = np.abs(token_d_values[seq_len//4:3*seq_len//4]).mean()
    late_d = np.abs(token_d_values[3*seq_len//4:]).mean()

    results = {
        'peak_token_position': int(peak_idx),
        'peak_token_fraction': float(peak_idx / seq_len),
        'peak_d': float(token_d_values[peak_idx]),
        'peak_p': float(token_p_values[peak_idx]),
        'early_mean_d': float(early_d),
        'middle_mean_d': float(middle_d),
        'late_mean_d': float(late_d),
        'signal_location': 'early' if early_d > max(middle_d, late_d) else
                          ('late' if late_d > middle_d else 'middle'),
        'd_by_position': token_d_values.tolist(),
        'n_correct': int(labels.sum()),
        'n_incorrect': int((~labels).sum())
    }

    return results


def jacobian_norm_by_token(trajectories, labels):
    """
    Compute Jacobian norm per token position to find "decision tokens".

    Higher Jacobian norm = more transformation happening at that position.
    """
    n_samples, seq_len, n_layers, d_model = trajectories.shape

    print(f"      Computing per-token Jacobian norms...")

    # Compute ||x_{l+1} - x_l|| for each token and layer
    velocity_norms = np.zeros((n_samples, seq_len, n_layers - 1))

    for i in range(n_samples):
        for l in range(n_layers - 1):
            for t in range(seq_len):
                v = trajectories[i, t, l+1, :] - trajectories[i, t, l, :]
                velocity_norms[i, t, l] = np.linalg.norm(v)

    # Average over layers
    mean_velocity = velocity_norms.mean(axis=2)  # (n_samples, seq_len)

    # Compare correct vs incorrect per token
    token_d_values = []
    for t in range(seq_len):
        correct_v = mean_velocity[labels, t]
        incorrect_v = mean_velocity[~labels, t]
        d = cohens_d(incorrect_v, correct_v)
        token_d_values.append(d)

    token_d_values = np.array(token_d_values)

    # Find where the difference is largest
    peak_idx = np.argmax(np.abs(token_d_values))

    results = {
        'jacobian_peak_position': int(peak_idx),
        'jacobian_peak_fraction': float(peak_idx / seq_len),
        'jacobian_peak_d': float(token_d_values[peak_idx]),
        'jacobian_d_by_position': token_d_values.tolist()
    }

    return results


# ============================================================================
# ANALYSIS 3: CROSS-DOMAIN SUBSPACE ALIGNMENT
# ============================================================================

def compute_error_subspace(trajectories, labels, k=10):
    """
    Compute the k-dimensional subspace that best separates correct/incorrect.

    Returns the top-k principal components of (incorrect_mean - correct_mean).
    """
    # Use last layer, averaged over tokens
    activations = trajectories[:, :, -1, :].mean(axis=1)  # (n_samples, d_model)

    correct_mean = activations[labels].mean(axis=0)
    incorrect_mean = activations[~labels].mean(axis=0)

    # Get the direction
    error_dir = incorrect_mean - correct_mean
    error_dir = error_dir / (np.linalg.norm(error_dir) + 1e-10)

    # Also get PCA components of the difference
    diff = activations[~labels] - correct_mean  # Deviation from correct mean
    pca = PCA(n_components=min(k, diff.shape[0]-1, diff.shape[1]))
    pca.fit(diff)

    return error_dir, pca.components_  # (d_model,), (k, d_model)


def cross_domain_subspace_alignment(task1_traj, task1_labels, task2_traj, task2_labels,
                                    task1_name, task2_name):
    """
    Measure alignment between error subspaces from two different tasks.

    High alignment suggests a shared "error signature".
    Low alignment suggests task-specific format detection.
    """
    print(f"      Computing subspace alignment: {task1_name} vs {task2_name}...")

    # Compute error directions
    dir1, subspace1 = compute_error_subspace(task1_traj, task1_labels, k=10)
    dir2, subspace2 = compute_error_subspace(task2_traj, task2_labels, k=10)

    # 1. Cosine similarity of primary error directions
    cos_sim = np.abs(np.dot(dir1, dir2))

    # 2. Subspace alignment (Grassmann distance approximation)
    # Use principal angles between subspaces
    S = subspace1 @ subspace2.T  # (k, k)
    _, singular_values, _ = np.linalg.svd(S)
    principal_angles = np.arccos(np.clip(singular_values, -1, 1))
    mean_angle = np.mean(principal_angles)

    # 3. Cross-transfer test: direction from task1 predicts on task2?
    # Project task2 onto task1's error direction
    task2_act = task2_traj[:, :, -1, :].mean(axis=1)
    projections = task2_act @ dir1

    correct_proj = projections[task2_labels]
    incorrect_proj = projections[~task2_labels]

    transfer_d = cohens_d(incorrect_proj, correct_proj)
    _, transfer_p = stats.mannwhitneyu(incorrect_proj, correct_proj, alternative='two-sided')

    results = {
        'task1': task1_name,
        'task2': task2_name,
        'direction_cosine_similarity': float(cos_sim),
        'mean_principal_angle_rad': float(mean_angle),
        'mean_principal_angle_deg': float(np.degrees(mean_angle)),
        'transfer_effect_size': {
            'd': float(transfer_d),
            'p': float(transfer_p)
        },
        'interpretation': 'aligned' if cos_sim > 0.5 else 'orthogonal'
    }

    return results


# ============================================================================
# MAIN
# ============================================================================

def analyze_single_file(filepath, task_name, max_samples=200):
    """Run all additional analyses on a single trajectory file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {task_name}")
    print(f"File: {filepath}")
    print(f"{'='*60}")

    traj, labels = load_trajectories(filepath, max_samples)
    n_correct = labels.sum()
    n_incorrect = (~labels).sum()

    print(f"Shape: {traj.shape}")
    print(f"Correct: {n_correct}/{len(labels)} ({100*n_correct/len(labels):.1f}%)")

    if n_correct < 5 or n_incorrect < 5:
        print("SKIPPED: Insufficient samples for analysis")
        return {'skipped': True, 'reason': 'insufficient samples'}

    results = {
        'task': task_name,
        'n_samples': len(labels),
        'n_correct': int(n_correct)
    }

    # Analysis 1: Path Signatures
    print("\n1. Path Signature Analysis...")
    if HAS_SIGNATORY:
        try:
            path_sig_results = path_signature_analysis(traj, labels, depth=3)
            results['path_signatures'] = path_sig_results
            if 'classifier_auc' in path_sig_results:
                print(f"   Classifier AUC: {path_sig_results['classifier_auc']['mean']:.3f} +/- {path_sig_results['classifier_auc']['std']:.3f}")
                print(f"   Norm d: {path_sig_results['norm_effect_size']['d']:.3f}")
        except Exception as e:
            print(f"   ERROR: {e}")
            results['path_signatures'] = {'error': str(e)}
    else:
        print("   SKIPPED: signatory not installed")
        results['path_signatures'] = {'skipped': True, 'reason': 'signatory not installed'}

    # Analysis 2: Token-Position Specificity
    print("\n2. Token-Position Specificity...")
    try:
        token_results = compute_token_error_signal(traj, labels)
        results['token_position'] = token_results
        print(f"   Peak at position {token_results['peak_token_position']}/{traj.shape[1]} ({token_results['peak_token_fraction']:.2f})")
        print(f"   Signal location: {token_results['signal_location']}")
        print(f"   Early/Middle/Late d: {token_results['early_mean_d']:.3f} / {token_results['middle_mean_d']:.3f} / {token_results['late_mean_d']:.3f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['token_position'] = {'error': str(e)}

    # Analysis 2b: Jacobian norm by token
    print("\n2b. Jacobian Norm by Token...")
    try:
        jac_results = jacobian_norm_by_token(traj, labels)
        results['jacobian_by_token'] = jac_results
        print(f"   Jacobian peak at position {jac_results['jacobian_peak_position']}")
    except Exception as e:
        print(f"   ERROR: {e}")
        results['jacobian_by_token'] = {'error': str(e)}

    return results, traj, labels


def main():
    parser = argparse.ArgumentParser(description='Additional analyses for surface vs reasoning')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with trajectory files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--models', type=str, default='olmo3_base', help='Comma-separated model names')
    parser.add_argument('--tasks', type=str, default='gsm8k,logiqa', help='Comma-separated task names')
    parser.add_argument('--max-samples', type=int, default=200, help='Max samples per file')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(',')]
    tasks = [t.strip() for t in args.tasks.split(',')]

    all_results = {}
    task_data = {}  # Store for cross-domain analysis

    # Run per-task analyses
    for model in models:
        for task in tasks:
            filepath = data_dir / model / f"{task}_trajectories.h5"
            if not filepath.exists():
                print(f"\nSkipping {model}/{task}: file not found")
                continue

            key = f"{model}/{task}"
            results, traj, labels = analyze_single_file(filepath, key, args.max_samples)
            all_results[key] = results

            if not results.get('skipped'):
                task_data[key] = (traj, labels)

            # Save individual results
            output_file = output_dir / f"{model}_{task}_additional.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved: {output_file}")

    # Cross-domain subspace alignment
    print("\n" + "="*60)
    print("CROSS-DOMAIN SUBSPACE ALIGNMENT")
    print("="*60)

    cross_domain_results = []
    keys = list(task_data.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key1, key2 = keys[i], keys[j]
            traj1, labels1 = task_data[key1]
            traj2, labels2 = task_data[key2]

            try:
                alignment = cross_domain_subspace_alignment(
                    traj1, labels1, traj2, labels2, key1, key2
                )
                cross_domain_results.append(alignment)
                print(f"   {key1} vs {key2}:")
                print(f"      Cosine sim: {alignment['direction_cosine_similarity']:.3f}")
                print(f"      Transfer d: {alignment['transfer_effect_size']['d']:.3f}")
            except Exception as e:
                print(f"   ERROR for {key1} vs {key2}: {e}")

    all_results['cross_domain_alignment'] = cross_domain_results

    # Save combined results
    combined_output = output_dir / "additional_analyses_all.json"
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined: {combined_output}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\n| Task | Path Sig AUC | Token Peak | Signal Loc | Cross-Domain |")
    print("|------|--------------|------------|------------|--------------|")

    for key, r in sorted(all_results.items()):
        if key == 'cross_domain_alignment':
            continue
        if r.get('skipped'):
            continue

        ps_auc = r.get('path_signatures', {}).get('classifier_auc', {}).get('mean', 'N/A')
        tok_peak = r.get('token_position', {}).get('peak_token_fraction', 'N/A')
        sig_loc = r.get('token_position', {}).get('signal_location', 'N/A')

        ps_str = f"{ps_auc:.3f}" if isinstance(ps_auc, float) else str(ps_auc)[:7]
        tok_str = f"{tok_peak:.2f}" if isinstance(tok_peak, float) else str(tok_peak)[:5]

        print(f"| {key:20s} | {ps_str:>12s} | {tok_str:>10s} | {sig_loc:>10s} | — |")

    if cross_domain_results:
        print("\nCross-Domain Alignment:")
        for cd in cross_domain_results:
            print(f"  {cd['task1']} vs {cd['task2']}: cos={cd['direction_cosine_similarity']:.3f}, transfer_d={cd['transfer_effect_size']['d']:.3f}")

    print(f"\nComplete. Results in {output_dir}")


if __name__ == '__main__':
    main()
