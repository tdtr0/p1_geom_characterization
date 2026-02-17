#!/usr/bin/env python3
"""
Path Signature Analysis v2 — Fixed Projection Methods

Replaces PCA with principled dimensionality reduction:
- Random projections (Johnson-Lindenstrauss): preserves pairwise distances without
  variance bias. Doesn't systematically discard tail components where RLVR features live.
- Velocity-space projection: when reducing velocity fields, project in velocity space
  (not activation space) so we analyze the right object.
- Cross-validated probe-informed projection: uses logistic probe weight as one
  projection dimension, ensuring correctness-relevant direction is preserved.

Why PCA was wrong:
1. Superposition: neurons encode superpositions of features; PCA captures high-variance
   directions which are dominated by frequent features, not correctness-relevant ones.
2. Gradient-eigenvector disconnect: PCA eigenvectors of activation covariance ≠ Jacobian
   singular vectors (functionally important directions). No reason they should align.
3. Empirical evidence: SVD experiment showed RLVR changes tail eigenvectors 3-8x more
   than top eigenvectors — PCA→32 discards exactly the refinement that matters.

Three projection methods tested:
- Method A: GaussianRandomProjection (JL lemma, ε≈0.3 for n=500, k=64)
- Method B: Velocity-space PCA (PCA on v=x_{l+1}-x_l, not on x)
- Method C: Probe-informed (CV logistic probe weight + random orthogonal complement)

Usage:
    python path_signature_analysis_v2.py \
        --data-dir /path/to/trajectories_0shot \
        --models olmo3_base,olmo3_sft,olmo3_rl_zero,olmo3_think \
        --output-dir results/v2_signatures \
        --max-samples 500
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA

# Try to import signatory
try:
    import torch
    import signatory
    HAS_SIGNATORY = True
except ImportError:
    HAS_SIGNATORY = False
    print("WARNING: signatory not installed. Install with: pip install signatory")


# ============================================================================
# UTILITIES
# ============================================================================

def load_trajectories(filepath, max_samples=None):
    """Load trajectories and labels from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        trajectories = f['trajectories'][:]
        if 'is_correct' in f:
            labels = f['is_correct'][:]
        elif 'correct' in f:
            labels = f['correct'][:]
        else:
            raise KeyError(f"No correctness labels. Keys: {list(f.keys())}")

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


# ============================================================================
# PROJECTION METHODS
# ============================================================================

def make_random_projector(d_model, n_components=64, seed=42):
    """
    Method A: Gaussian random projection (Johnson-Lindenstrauss).

    JL lemma guarantees: for n points, projection to k dims preserves all
    pairwise distances within (1±ε) where ε ≈ sqrt(8*log(n)/k).
    For n=500, k=64: ε ≈ 0.88 (theoretical worst case; empirical much better).
    For n=500, k=128: ε ≈ 0.62.

    Key advantage: No bias toward high-variance directions.
    """
    rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
    # Fit on dummy data to initialize the projection matrix
    rp.fit(np.zeros((1, d_model)))
    return rp


def make_velocity_pca_projector(velocities_flat, n_components=64):
    """
    Method B: PCA on velocity field (not activation field).

    If we must use PCA, at least do it on the right object: the velocity
    v = x_{l+1} - x_l, not the activation x. The velocity field has different
    principal directions that capture "where computation moves most."
    """
    pca = PCA(n_components=min(n_components, velocities_flat.shape[1], velocities_flat.shape[0]))
    pca.fit(velocities_flat)
    return pca


def make_probe_informed_projector(X_train, y_train, d_model, n_components=64, seed=42):
    """
    Method C: Cross-validated probe-informed projection.

    Uses logistic probe weight as dim 0 (known correctness direction),
    fills remaining dims with random orthogonal directions.

    MUST be cross-validated: train probe on fold A, project fold B.
    """
    probe = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', random_state=seed)
    probe.fit(X_train, y_train)
    w = probe.coef_[0].copy()
    w = w / (np.linalg.norm(w) + 1e-10)

    # Build orthonormal basis: w as dim 0, random orthogonal complement
    rng = np.random.RandomState(seed)
    basis = np.zeros((n_components, d_model))
    basis[0] = w

    # Generate random directions, orthogonalize against w
    if n_components > 1:
        random_dirs = rng.randn(n_components - 1, d_model)
        # Remove component along w
        random_dirs -= (random_dirs @ w[:, None]) * w[None, :]
        # QR decomposition for orthonormal basis
        Q, _ = np.linalg.qr(random_dirs.T)
        n_fill = min(n_components - 1, Q.shape[1])
        basis[1:1 + n_fill] = Q[:, :n_fill].T

    return basis


# ============================================================================
# PATH SIGNATURE COMPUTATION
# ============================================================================

def compute_signatures_with_projection(trajectories, labels, projector, projector_name,
                                        view='layers_as_path', depth=3,
                                        target_layer=8, max_seq_len=128):
    """
    Compute path signatures using a given projection method.

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean
        projector: sklearn transformer with .transform() or numpy basis matrix
        projector_name: string identifier
        view: 'layers_as_path' or 'sequence_as_path'
        depth: signature depth
        target_layer: layer index for sequence_as_path
        max_seq_len: truncation for sequence_as_path
    """
    if not HAS_SIGNATORY:
        return {'error': 'signatory not installed', 'method': projector_name}

    n_samples, seq_len, n_layers, d_model = trajectories.shape

    signatures = []
    for i in range(n_samples):
        if view == 'layers_as_path':
            # Mean across tokens, path through layers
            path = trajectories[i].mean(axis=0)  # (n_layers, d_model)
        else:
            # Path through tokens at target layer
            sl = min(seq_len, max_seq_len)
            path = trajectories[i, :sl, target_layer, :]  # (seq_len, d_model)

        # Project
        if hasattr(projector, 'transform'):
            path_reduced = projector.transform(path)
        elif isinstance(projector, np.ndarray):
            # Basis matrix: (n_components, d_model)
            path_reduced = path @ projector.T
        else:
            raise ValueError(f"Unknown projector type: {type(projector)}")

        # Compute signature
        path_tensor = torch.tensor(path_reduced, dtype=torch.float32).unsqueeze(0)
        sig = signatory.signature(path_tensor, depth=depth)
        signatures.append(sig.squeeze().numpy())

    signatures = np.array(signatures)

    # Compute signature norms
    sig_norms = np.linalg.norm(signatures, axis=1)
    correct_norms = sig_norms[labels]
    incorrect_norms = sig_norms[~labels]

    d_norm = cohens_d(correct_norms, incorrect_norms)
    if len(correct_norms) > 0 and len(incorrect_norms) > 0:
        _, p_norm = stats.mannwhitneyu(correct_norms, incorrect_norms, alternative='two-sided')
    else:
        p_norm = 1.0

    # Train classifier on signatures
    auc_mean, auc_std = 0.5, 0.0
    if labels.sum() >= 5 and (~labels).sum() >= 5:
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver='lbfgs')
        try:
            cv_scores = cross_val_score(clf, signatures, labels, cv=5, scoring='roc_auc')
            auc_mean = float(cv_scores.mean())
            auc_std = float(cv_scores.std())
        except Exception:
            pass

    return {
        'method': projector_name,
        'view': view,
        'signature_depth': depth,
        'signature_dim': int(signatures.shape[1]),
        'n_projection_dims': int(path_reduced.shape[1]) if len(signatures) > 0 else 0,
        'norm_d': float(d_norm),
        'norm_p': float(p_norm),
        'correct_norm_mean': float(correct_norms.mean()) if len(correct_norms) > 0 else 0,
        'incorrect_norm_mean': float(incorrect_norms.mean()) if len(incorrect_norms) > 0 else 0,
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'n_correct': int(labels.sum()),
        'n_incorrect': int((~labels).sum()),
        'signatures': signatures,
        'labels': labels
    }


# ============================================================================
# CROSS-DOMAIN TRANSFER
# ============================================================================

def cross_domain_transfer(train_sigs, train_labels, test_sigs, test_labels):
    """Train on source domain, test on target. Returns AUC."""
    if train_labels.sum() < 5 or (~train_labels).sum() < 5:
        return {'error': 'insufficient training samples'}
    if test_labels.sum() < 5 or (~test_labels).sum() < 5:
        return {'error': 'insufficient test samples'}

    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver='lbfgs')
    clf.fit(train_sigs, train_labels)

    from sklearn.metrics import roc_auc_score
    probs = clf.predict_proba(test_sigs)[:, 1]
    auc = roc_auc_score(test_labels, probs)

    return {
        'transfer_auc': float(auc),
        'train_n': int(len(train_labels)),
        'test_n': int(len(test_labels))
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis(data_dir, models, output_dir, max_samples=500, n_proj_dims=64, depth=3):
    """Run path signature analysis with all three projection methods."""
    if not HAS_SIGNATORY:
        print("ERROR: signatory not installed. Cannot run analysis.")
        return

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ['gsm8k', 'humaneval', 'logiqa']
    all_results = []
    signatures_cache = {}  # For cross-domain transfer

    for model in models:
        for task in tasks:
            h5_path = data_dir / model / f"{task}_trajectories.h5"
            if not h5_path.exists():
                print(f"  SKIP: {model}/{task} - file not found")
                continue

            print(f"\n{'='*60}")
            print(f"  Processing {model}/{task}...")
            print(f"{'='*60}")

            try:
                trajectories, labels = load_trajectories(h5_path, max_samples)
            except Exception as e:
                print(f"    ERROR loading: {e}")
                continue

            n_samples, seq_len, n_layers, d_model = trajectories.shape
            n_correct = labels.sum()
            n_incorrect = (~labels).sum()
            print(f"    Loaded {len(labels)} samples: {n_correct} correct, {n_incorrect} incorrect")
            print(f"    Shape: ({n_samples}, {seq_len}, {n_layers}, {d_model})")

            if n_correct < 5 or n_incorrect < 5:
                print(f"    SKIP: insufficient samples for analysis")
                continue

            # ==============================================================
            # Method A: Random Projection (JL)
            # ==============================================================
            print(f"\n    Method A: Random Projection (k={n_proj_dims})...")
            rp = make_random_projector(d_model, n_components=n_proj_dims)

            result_a = compute_signatures_with_projection(
                trajectories, labels, rp, 'random_projection',
                view='layers_as_path', depth=depth
            )
            print(f"      layers_as_path: AUC={result_a['auc_mean']:.3f}±{result_a['auc_std']:.3f}, d={result_a['norm_d']:.3f}")
            all_results.append({
                'model': model, 'task': task, **{k: v for k, v in result_a.items()
                                                   if k not in ('signatures', 'labels')}
            })
            signatures_cache[(model, task, 'rp_layers')] = (result_a['signatures'], result_a['labels'])

            # Sequence-as-path with random projection
            mid_layer = n_layers // 2
            result_a_seq = compute_signatures_with_projection(
                trajectories, labels, rp, 'random_projection',
                view='sequence_as_path', depth=depth, target_layer=mid_layer
            )
            print(f"      seq_as_path L{mid_layer}: AUC={result_a_seq['auc_mean']:.3f}±{result_a_seq['auc_std']:.3f}")
            all_results.append({
                'model': model, 'task': task, **{k: v for k, v in result_a_seq.items()
                                                   if k not in ('signatures', 'labels')}
            })

            # ==============================================================
            # Method B: Velocity-space PCA
            # ==============================================================
            print(f"\n    Method B: Velocity-space PCA (k={n_proj_dims})...")
            # Compute velocities first, then PCA on velocity field
            traj_mean = trajectories.mean(axis=1)  # (n_samples, n_layers, d_model)
            velocities = traj_mean[:, 1:, :] - traj_mean[:, :-1, :]  # (n_samples, n_layers-1, d_model)
            v_flat = velocities.reshape(-1, d_model)

            # Subsample for PCA fitting if needed
            if len(v_flat) > 50000:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(v_flat), 50000, replace=False)
                v_flat_sub = v_flat[idx]
            else:
                v_flat_sub = v_flat

            vpca = make_velocity_pca_projector(v_flat_sub, n_components=n_proj_dims)

            result_b = compute_signatures_with_projection(
                trajectories, labels, vpca, 'velocity_pca',
                view='layers_as_path', depth=depth
            )
            print(f"      layers_as_path: AUC={result_b['auc_mean']:.3f}±{result_b['auc_std']:.3f}, d={result_b['norm_d']:.3f}")
            all_results.append({
                'model': model, 'task': task, **{k: v for k, v in result_b.items()
                                                   if k not in ('signatures', 'labels')}
            })
            signatures_cache[(model, task, 'vpca_layers')] = (result_b['signatures'], result_b['labels'])

            # ==============================================================
            # Method C: Probe-informed projection (cross-validated)
            # ==============================================================
            print(f"\n    Method C: Probe-informed projection (CV, k={n_proj_dims})...")
            # Use 5-fold CV: train probe on fold, project held-out fold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            all_sigs_c = np.zeros((n_samples, 0))  # Will be filled after first fold
            sig_dim = None

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(trajectories, labels)):
                # Train probe on train fold (use mean activation at best layer from prior results)
                train_acts = trajectories[train_idx].mean(axis=(1, 2))  # (n_train, d_model)
                train_labels_fold = labels[train_idx]

                basis = make_probe_informed_projector(
                    train_acts, train_labels_fold, d_model,
                    n_components=n_proj_dims, seed=42 + fold_idx
                )

                # Compute signatures for held-out fold
                test_traj = trajectories[test_idx]
                for i, idx in enumerate(test_idx):
                    path = test_traj[i].mean(axis=0)  # (n_layers, d_model)
                    path_reduced = path @ basis.T
                    path_tensor = torch.tensor(path_reduced, dtype=torch.float32).unsqueeze(0)
                    sig = signatory.signature(path_tensor, depth=depth).squeeze().numpy()

                    if sig_dim is None:
                        sig_dim = len(sig)
                        all_sigs_c = np.zeros((n_samples, sig_dim))
                    all_sigs_c[idx] = sig

            # Compute stats on CV signatures
            sig_norms_c = np.linalg.norm(all_sigs_c, axis=1)
            correct_norms_c = sig_norms_c[labels]
            incorrect_norms_c = sig_norms_c[~labels]
            d_norm_c = cohens_d(correct_norms_c, incorrect_norms_c)

            clf_c = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver='lbfgs')
            try:
                cv_scores_c = cross_val_score(clf_c, all_sigs_c, labels, cv=5, scoring='roc_auc')
                auc_c = float(cv_scores_c.mean())
                auc_c_std = float(cv_scores_c.std())
            except Exception:
                auc_c, auc_c_std = 0.5, 0.0

            print(f"      layers_as_path: AUC={auc_c:.3f}±{auc_c_std:.3f}, d={d_norm_c:.3f}")
            all_results.append({
                'model': model, 'task': task, 'method': 'probe_informed_cv',
                'view': 'layers_as_path', 'signature_depth': depth,
                'signature_dim': sig_dim or 0, 'n_projection_dims': n_proj_dims,
                'norm_d': float(d_norm_c), 'auc_mean': auc_c, 'auc_std': auc_c_std,
                'n_correct': int(n_correct), 'n_incorrect': int(n_incorrect)
            })
            signatures_cache[(model, task, 'probe_layers')] = (all_sigs_c, labels)

    # ==================================================================
    # Cross-domain transfer tests
    # ==================================================================
    print(f"\n{'='*60}")
    print("  Cross-domain transfer tests")
    print(f"{'='*60}")

    transfer_results = []
    for method_tag in ['rp_layers', 'vpca_layers', 'probe_layers']:
        method_name = {'rp_layers': 'random_projection', 'vpca_layers': 'velocity_pca',
                       'probe_layers': 'probe_informed_cv'}[method_tag]

        for model in models:
            task_pairs = [('gsm8k', 'humaneval'), ('humaneval', 'gsm8k'),
                          ('gsm8k', 'logiqa'), ('logiqa', 'gsm8k'),
                          ('humaneval', 'logiqa'), ('logiqa', 'humaneval')]

            for train_task, test_task in task_pairs:
                key_train = (model, train_task, method_tag)
                key_test = (model, test_task, method_tag)

                if key_train not in signatures_cache or key_test not in signatures_cache:
                    continue

                train_sigs, train_labels = signatures_cache[key_train]
                test_sigs, test_labels = signatures_cache[key_test]

                transfer = cross_domain_transfer(train_sigs, train_labels, test_sigs, test_labels)
                if 'transfer_auc' in transfer:
                    transfer_results.append({
                        'model': model, 'method': method_name,
                        'train_task': train_task, 'test_task': test_task,
                        'transfer_auc': transfer['transfer_auc']
                    })
                    print(f"    [{method_name}] {model}: {train_task}->{test_task} AUC={transfer['transfer_auc']:.3f}")

    # ==================================================================
    # Save results
    # ==================================================================
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = output_dir / 'h2_path_signatures_v2.csv'
        df.to_csv(output_path, index=False)
        print(f"\n  Saved main results to {output_path}")

        # Print comparison table
        print(f"\n  === Method Comparison (layers_as_path) ===")
        lap = df[df['view'] == 'layers_as_path']
        pivot = lap.pivot_table(index=['model', 'task'], columns='method',
                                values='auc_mean', aggfunc='first')
        print(pivot.to_string())

    if transfer_results:
        df_transfer = pd.DataFrame(transfer_results)
        transfer_path = output_dir / 'h2_path_signatures_v2_transfer.csv'
        df_transfer.to_csv(transfer_path, index=False)
        print(f"\n  Saved transfer results to {transfer_path}")

    # Save full results as JSON
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_proj_dims': n_proj_dims,
            'depth': depth,
            'max_samples': max_samples
        },
        'main_results': all_results,
        'transfer_results': transfer_results
    }
    json_path = output_dir / 'h2_path_signatures_v2.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved JSON to {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Path Signature Analysis v2 (fixed projections)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing trajectory HDF5 files')
    parser.add_argument('--models', type=str,
                        default='olmo3_base,olmo3_sft,olmo3_rl_zero,olmo3_think',
                        help='Comma-separated list of models')
    parser.add_argument('--output-dir', type=str, default='results/v2_signatures',
                        help='Output directory')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Maximum samples per task')
    parser.add_argument('--n-proj-dims', type=int, default=64,
                        help='Projection dimensionality (default: 64)')
    parser.add_argument('--depth', type=int, default=3,
                        help='Signature depth (default: 3)')

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(',')]

    print(f"\n{'='*60}")
    print("Path Signature Analysis v2 — Fixed Projections")
    print(f"{'='*60}")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Data directory: {args.data_dir}")
    print(f"Models: {models}")
    print(f"Projection dims: {args.n_proj_dims}")
    print(f"Signature depth: {args.depth}")
    print(f"Max samples: {args.max_samples}")
    print(f"Methods: random_projection, velocity_pca, probe_informed_cv")
    print(f"{'='*60}")

    run_analysis(
        data_dir=args.data_dir,
        models=models,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        n_proj_dims=args.n_proj_dims,
        depth=args.depth
    )

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
