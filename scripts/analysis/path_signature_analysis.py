#!/usr/bin/env python3
"""
Path Signature Analysis - Phase 3 Extension

Computes reparameterization-invariant path signatures from layer activations.
Path signatures capture the "dynamical structure" of computation flow.

Two views:
1. Layers-as-Path: Mean activation across tokens → path through layers
2. Sequence-as-Path: Activation at a specific layer → path through tokens

Tests:
- Cohen's d between correct/incorrect signature norms
- Logistic regression AUC on signatures
- Cross-domain transfer (train GSM8K → test HumanEval)

Usage:
    python path_signature_analysis.py \
        --data-dir /path/to/trajectories_0shot \
        --models olmo3_base,olmo3_sft,olmo3_rl_zero \
        --output-dir results
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Try to import signatory
try:
    import torch
    import signatory
    HAS_SIGNATORY = True
except ImportError:
    HAS_SIGNATORY = False
    print("WARNING: signatory not installed. Install with: pip install signatory")


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


def compute_path_signature(path, depth=3, n_components=32):
    """
    Compute path signature from a path.

    Args:
        path: (length, d_model) - the path to compute signature for
        depth: signature depth (higher = more features)
        n_components: PCA dimensionality for efficiency

    Returns:
        signature: 1D array of signature features
    """
    if not HAS_SIGNATORY:
        return None

    # Reduce dimensionality via PCA for efficiency
    if path.shape[1] > n_components:
        pca = PCA(n_components=n_components)
        path_reduced = pca.fit_transform(path)
    else:
        path_reduced = path

    # Convert to torch tensor with batch dim: (1, length, channels)
    path_tensor = torch.tensor(path_reduced, dtype=torch.float32).unsqueeze(0)

    # Compute signature
    sig = signatory.signature(path_tensor, depth=depth)

    return sig.squeeze().numpy()


def layers_as_path_analysis(trajectories, labels, depth=3, n_components=32):
    """
    View: Mean activation across tokens forms a path through layers.

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean
        depth: signature depth
        n_components: PCA components

    Returns:
        dict with signature analysis results
    """
    if not HAS_SIGNATORY:
        return {'error': 'signatory not installed'}

    n_samples, seq_len, n_layers, d_model = trajectories.shape
    print(f"    Computing layers-as-path signatures (depth={depth}, n_components={n_components})...")
    print(f"    Shape: {n_samples} samples, {n_layers} layers, {d_model} dims")

    # Average over sequence tokens first: (n_samples, n_layers, d_model)
    print(f"    Averaging over {seq_len} tokens...")
    traj_mean = trajectories.mean(axis=1)

    # Compute shared PCA on all data
    print(f"    Fitting PCA...")
    all_activations = traj_mean.reshape(-1, d_model)
    pca = PCA(n_components=min(n_components, d_model))
    pca.fit(all_activations)

    print(f"    Computing signatures for {n_samples} samples...")
    signatures = []
    for i in range(n_samples):
        if i % 20 == 0:
            print(f"      Sample {i}/{n_samples}...")

        # Transform this sample's path
        path = traj_mean[i]  # (n_layers, d_model)
        path_reduced = pca.transform(path)  # (n_layers, n_components)

        # Compute signature
        path_tensor = torch.tensor(path_reduced, dtype=torch.float32).unsqueeze(0)
        sig = signatory.signature(path_tensor, depth=depth)
        signatures.append(sig.squeeze().numpy())

    signatures = np.array(signatures)
    print(f"    Signature shape: {signatures.shape}")

    # Compute signature norms
    sig_norms = np.linalg.norm(signatures, axis=1)
    correct_norms = sig_norms[labels]
    incorrect_norms = sig_norms[~labels]

    d_norm = cohens_d(correct_norms, incorrect_norms)
    _, p_norm = stats.mannwhitneyu(correct_norms, incorrect_norms, alternative='two-sided')

    # Train classifier on signatures
    clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    if labels.sum() >= 5 and (~labels).sum() >= 5:
        cv_scores = cross_val_score(clf, signatures, labels, cv=5, scoring='roc_auc')
        auc_mean = float(cv_scores.mean())
        auc_std = float(cv_scores.std())
    else:
        auc_mean = 0.5
        auc_std = 0.0

    return {
        'view': 'layers_as_path',
        'signature_depth': depth,
        'signature_dim': int(signatures.shape[1]),
        'norm_d': float(d_norm),
        'norm_p': float(p_norm),
        'correct_norm_mean': float(correct_norms.mean()),
        'incorrect_norm_mean': float(incorrect_norms.mean()),
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'n_correct': int(labels.sum()),
        'n_incorrect': int((~labels).sum()),
        'signatures': signatures,
        'labels': labels
    }


def sequence_as_path_analysis(trajectories, labels, target_layer=8, depth=3, n_components=32, max_seq_len=128):
    """
    View: Activation at a specific layer forms a path through sequence positions.

    Args:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean
        target_layer: which layer to analyze
        depth: signature depth
        n_components: PCA components
        max_seq_len: truncate sequences for efficiency

    Returns:
        dict with signature analysis results
    """
    if not HAS_SIGNATORY:
        return {'error': 'signatory not installed'}

    n_samples, seq_len, n_layers, d_model = trajectories.shape
    seq_len = min(seq_len, max_seq_len)

    print(f"    Computing sequence-as-path signatures at layer {target_layer}...")

    # Extract activations at target layer: (n_samples, seq_len, d_model)
    layer_acts = trajectories[:, :seq_len, target_layer, :]

    # Compute shared PCA
    all_activations = layer_acts.reshape(-1, d_model)
    pca = PCA(n_components=min(n_components, d_model))
    pca.fit(all_activations)

    signatures = []
    for i in range(n_samples):
        path = layer_acts[i]  # (seq_len, d_model)
        path_reduced = pca.transform(path)

        path_tensor = torch.tensor(path_reduced, dtype=torch.float32).unsqueeze(0)
        sig = signatory.signature(path_tensor, depth=depth)
        signatures.append(sig.squeeze().numpy())

    signatures = np.array(signatures)
    print(f"    Signature shape: {signatures.shape}")

    # Compute norms and statistics
    sig_norms = np.linalg.norm(signatures, axis=1)
    correct_norms = sig_norms[labels]
    incorrect_norms = sig_norms[~labels]

    d_norm = cohens_d(correct_norms, incorrect_norms)
    _, p_norm = stats.mannwhitneyu(correct_norms, incorrect_norms, alternative='two-sided')

    # Classifier
    clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    if labels.sum() >= 5 and (~labels).sum() >= 5:
        cv_scores = cross_val_score(clf, signatures, labels, cv=5, scoring='roc_auc')
        auc_mean = float(cv_scores.mean())
        auc_std = float(cv_scores.std())
    else:
        auc_mean = 0.5
        auc_std = 0.0

    return {
        'view': 'sequence_as_path',
        'target_layer': target_layer,
        'signature_depth': depth,
        'signature_dim': int(signatures.shape[1]),
        'norm_d': float(d_norm),
        'norm_p': float(p_norm),
        'correct_norm_mean': float(correct_norms.mean()),
        'incorrect_norm_mean': float(incorrect_norms.mean()),
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'n_correct': int(labels.sum()),
        'n_incorrect': int((~labels).sum()),
        'signatures': signatures,
        'labels': labels
    }


def cross_domain_transfer(train_sigs, train_labels, test_sigs, test_labels):
    """
    Test cross-domain transfer: train on source, test on target.

    Returns AUC on target domain.
    """
    clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')

    # Need enough samples of each class
    if train_labels.sum() < 5 or (~train_labels).sum() < 5:
        return {'error': 'insufficient training samples'}
    if test_labels.sum() < 5 or (~test_labels).sum() < 5:
        return {'error': 'insufficient test samples'}

    clf.fit(train_sigs, train_labels)

    # Predict probabilities and compute AUC
    from sklearn.metrics import roc_auc_score
    probs = clf.predict_proba(test_sigs)[:, 1]
    auc = roc_auc_score(test_labels, probs)

    return {
        'transfer_auc': float(auc),
        'train_n': int(len(train_labels)),
        'test_n': int(len(test_labels))
    }


def run_analysis(data_dir, models, output_dir, max_samples=100):
    """Run path signature analysis for all model/task combinations."""
    if not HAS_SIGNATORY:
        print("ERROR: signatory not installed. Cannot run analysis.")
        return

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ['gsm8k', 'humaneval', 'logiqa']
    results = []

    # Store signatures for cross-domain transfer
    signatures_cache = {}

    for model in models:
        for task in tasks:
            # Construct file path
            h5_path = data_dir / model / f"{task}_trajectories.h5"

            if not h5_path.exists():
                print(f"  SKIP: {model}/{task} - file not found")
                continue

            print(f"\n  Processing {model}/{task}...")

            try:
                trajectories, labels = load_trajectories(h5_path, max_samples)
            except Exception as e:
                print(f"    ERROR loading: {e}")
                continue

            n_correct = labels.sum()
            n_incorrect = (~labels).sum()
            print(f"    Loaded {len(labels)} samples: {n_correct} correct, {n_incorrect} incorrect")

            if n_correct < 5 or n_incorrect < 5:
                print(f"    SKIP: insufficient samples for analysis")
                continue

            # Run layers-as-path analysis
            lap_result = layers_as_path_analysis(trajectories, labels)

            if 'error' not in lap_result:
                results.append({
                    'model': model,
                    'task': task,
                    'view': 'layers_as_path',
                    'sig_norm_d': lap_result['norm_d'],
                    'sig_norm_p': lap_result['norm_p'],
                    'auc_mean': lap_result['auc_mean'],
                    'auc_std': lap_result['auc_std'],
                    'n_correct': lap_result['n_correct'],
                    'n_incorrect': lap_result['n_incorrect']
                })

                # Cache for cross-domain
                signatures_cache[(model, task, 'layers')] = (
                    lap_result['signatures'],
                    lap_result['labels']
                )

            # Run sequence-as-path analysis (middle layer)
            n_layers = trajectories.shape[2]
            mid_layer = n_layers // 2
            sap_result = sequence_as_path_analysis(trajectories, labels, target_layer=mid_layer)

            if 'error' not in sap_result:
                results.append({
                    'model': model,
                    'task': task,
                    'view': f'sequence_as_path_L{mid_layer}',
                    'sig_norm_d': sap_result['norm_d'],
                    'sig_norm_p': sap_result['norm_p'],
                    'auc_mean': sap_result['auc_mean'],
                    'auc_std': sap_result['auc_std'],
                    'n_correct': sap_result['n_correct'],
                    'n_incorrect': sap_result['n_incorrect']
                })

    # Cross-domain transfer tests
    print("\n  Testing cross-domain transfer...")
    transfer_results = []

    for model in models:
        # GSM8K -> HumanEval
        key_train = (model, 'gsm8k', 'layers')
        key_test = (model, 'humaneval', 'layers')

        if key_train in signatures_cache and key_test in signatures_cache:
            train_sigs, train_labels = signatures_cache[key_train]
            test_sigs, test_labels = signatures_cache[key_test]

            transfer = cross_domain_transfer(train_sigs, train_labels, test_sigs, test_labels)
            if 'transfer_auc' in transfer:
                transfer_results.append({
                    'model': model,
                    'train_task': 'gsm8k',
                    'test_task': 'humaneval',
                    'transfer_auc': transfer['transfer_auc']
                })
                print(f"    {model}: GSM8K->HumanEval AUC = {transfer['transfer_auc']:.3f}")

        # HumanEval -> GSM8K
        if key_train in signatures_cache and key_test in signatures_cache:
            train_sigs, train_labels = signatures_cache[key_test]
            test_sigs, test_labels = signatures_cache[key_train]

            transfer = cross_domain_transfer(train_sigs, train_labels, test_sigs, test_labels)
            if 'transfer_auc' in transfer:
                transfer_results.append({
                    'model': model,
                    'train_task': 'humaneval',
                    'test_task': 'gsm8k',
                    'transfer_auc': transfer['transfer_auc']
                })
                print(f"    {model}: HumanEval->GSM8K AUC = {transfer['transfer_auc']:.3f}")

    # Save results
    if results:
        df = pd.DataFrame(results)
        output_path = output_dir / 'h2_path_signatures.csv'
        df.to_csv(output_path, index=False)
        print(f"\n  Saved main results to {output_path}")

        # Print summary table
        print("\n  === Path Signature Results ===")
        print(df.to_string(index=False))

    if transfer_results:
        df_transfer = pd.DataFrame(transfer_results)
        transfer_path = output_dir / 'h2_path_signatures_transfer.csv'
        df_transfer.to_csv(transfer_path, index=False)
        print(f"\n  Saved transfer results to {transfer_path}")


def main():
    parser = argparse.ArgumentParser(description='Path Signature Analysis')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing trajectory HDF5 files')
    parser.add_argument('--models', type=str, default='olmo3_base,olmo3_sft,olmo3_rl_zero',
                        help='Comma-separated list of models to analyze')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Maximum samples per task')

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]

    print(f"\n{'='*60}")
    print("Path Signature Analysis")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Models: {models}")
    print(f"Max samples: {args.max_samples}")
    print(f"{'='*60}")

    run_analysis(
        data_dir=args.data_dir,
        models=models,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
