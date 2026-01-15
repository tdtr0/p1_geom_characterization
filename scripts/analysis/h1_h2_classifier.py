#!/usr/bin/env python3
"""
H1/H2 Hypothesis Testing: Trajectory-based Correctness Classification

H1: Can we distinguish correct vs incorrect reasoning trajectories within a domain?
H2: Does a classifier trained on one domain transfer to another?

Usage:
    python scripts/analysis/h1_h2_classifier.py --data-dir data/trajectories --model olmo3_sft
"""

import sys
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import argparse
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Feature Extraction
# =============================================================================

def extract_trajectory_features(trajectory: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract geometric statistics from a single trajectory.

    Args:
        trajectory: (seq_len, n_layers, d_model) activation path

    Returns:
        Dict of feature arrays, total ~124 features
    """
    # Convert to float32 to avoid overflow with float16 data
    trajectory = trajectory.astype(np.float32)
    seq_len, n_layers, d_model = trajectory.shape

    # 1. Mean activation norm per layer
    mean_per_layer = trajectory.mean(axis=0)  # (n_layers, d_model)
    mean_norm = np.linalg.norm(mean_per_layer, axis=1)  # (n_layers,)

    # 2. Variance per layer
    var_per_layer = trajectory.var(axis=(0, 2))  # (n_layers,)

    # 3. Velocity: layer-to-layer difference
    velocity = np.diff(mean_per_layer, axis=0)  # (n_layers-1, d_model)
    velocity_norm = np.linalg.norm(velocity, axis=1)  # (n_layers-1,)

    # 4. Curvature: second derivative
    curvature = np.diff(velocity, axis=0)  # (n_layers-2, d_model)
    curvature_norm = np.linalg.norm(curvature, axis=1)  # (n_layers-2,)

    # 5. Cosine similarity between consecutive layers
    cos_sim = np.array([
        np.dot(mean_per_layer[i], mean_per_layer[i+1]) /
        (np.linalg.norm(mean_per_layer[i]) * np.linalg.norm(mean_per_layer[i+1]) + 1e-8)
        for i in range(n_layers - 1)
    ])

    # 6. Activation entropy per layer
    def layer_entropy(acts):
        acts_abs = np.abs(acts).mean(axis=0)
        acts_norm = acts_abs / (acts_abs.sum() + 1e-8)
        return entropy(acts_norm)

    ent_per_layer = np.array([layer_entropy(trajectory[:, i, :]) for i in range(n_layers)])

    # 7. First/last token difference
    first_last_diff = np.linalg.norm(trajectory[-1] - trajectory[0], axis=1)

    # 8. Max activation per layer
    max_per_layer = np.abs(trajectory).max(axis=(0, 2))

    return {
        'mean_norm': mean_norm,
        'variance': var_per_layer,
        'velocity': velocity_norm,
        'curvature': curvature_norm,
        'cos_sim': cos_sim,
        'entropy': ent_per_layer,
        'first_last_diff': first_last_diff,
        'max_activation': max_per_layer,
    }


def features_to_vector(features: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten feature dict to single vector."""
    return np.concatenate([
        features['mean_norm'],
        features['variance'],
        features['velocity'],
        features['curvature'],
        features['cos_sim'],
        features['entropy'],
        features['first_last_diff'],
        features['max_activation'],
    ])


def get_feature_names(n_layers: int = 16) -> List[str]:
    """Get names for all features."""
    names = []
    names.extend([f'mean_norm_L{i}' for i in range(n_layers)])
    names.extend([f'variance_L{i}' for i in range(n_layers)])
    names.extend([f'velocity_L{i}' for i in range(n_layers - 1)])
    names.extend([f'curvature_L{i}' for i in range(n_layers - 2)])
    names.extend([f'cos_sim_L{i}' for i in range(n_layers - 1)])
    names.extend([f'entropy_L{i}' for i in range(n_layers)])
    names.extend([f'first_last_diff_L{i}' for i in range(n_layers)])
    names.extend([f'max_activation_L{i}' for i in range(n_layers)])
    return names


# =============================================================================
# Data Loading
# =============================================================================

def load_trajectories_with_labels(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load trajectories and correctness labels from HDF5 file.

    Returns:
        trajectories: (n_samples, seq_len, n_layers, d_model)
        labels: (n_samples,) boolean array
        indices: (n_samples,) sample indices
    """
    print(f"  Loading {filepath}...", flush=True)

    with h5py.File(filepath, 'r') as f:
        # Get trajectories
        trajectories = f['trajectories'][:]  # (n_samples, seq_len, n_layers, d_model)

        # Get correctness labels
        if 'is_correct' in f:
            labels = f['is_correct'][:]
        elif 'correct' in f:
            labels = f['correct'][:]
        elif 'labels' in f:
            labels = f['labels'][:]
        else:
            raise KeyError(f"No correctness labels found in {filepath}. Keys: {list(f.keys())}")

        # Get indices if available
        if 'indices' in f:
            indices = f['indices'][:]
        else:
            indices = np.arange(len(labels))

    print(f"    Shape: {trajectories.shape}", flush=True)
    print(f"    Correct: {labels.sum()}/{len(labels)} ({100*labels.mean():.1f}%)", flush=True)

    return trajectories, labels.astype(bool), indices


def extract_features_batch(trajectories: np.ndarray, n_jobs: int = 32) -> np.ndarray:
    """Extract features for all trajectories with parallel processing."""
    from joblib import Parallel, delayed
    import multiprocessing

    n_samples = trajectories.shape[0]
    max_workers = multiprocessing.cpu_count()
    n_jobs = min(n_jobs, max_workers)

    print(f"  Extracting features with {n_jobs} workers (max: {max_workers})...", flush=True)

    def extract_single(i):
        features = extract_trajectory_features(trajectories[i])
        return features_to_vector(features)

    # Parallel feature extraction
    features_list = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(
        delayed(extract_single)(i) for i in range(n_samples)
    )

    # Convert to array and clean up any NaN/Inf values
    X = np.array(features_list)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    return X


# =============================================================================
# Classification
# =============================================================================

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_type: str = 'logistic'
) -> Dict:
    """Train classifier and evaluate."""

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create classifier with class balancing
    if classifier_type == 'logistic':
        clf = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=42)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    # Train
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, 'predict_proba') else None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    # Feature importance
    if classifier_type == 'logistic':
        importance = np.abs(clf.coef_[0])
    else:
        importance = clf.feature_importances_

    return {
        'accuracy': accuracy,
        'auc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'importance': importance,
        'classifier': clf,
        'scaler': scaler,
    }


def cross_validate(X: np.ndarray, y: np.ndarray, classifier_type: str = 'logistic', n_folds: int = 5) -> Dict:
    """Cross-validation within a single dataset."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if classifier_type == 'logistic':
        clf = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')

    return {
        'accuracy_mean': scores.mean(),
        'accuracy_std': scores.std(),
        'auc_mean': auc_scores.mean(),
        'auc_std': auc_scores.std(),
        'fold_accuracies': scores.tolist(),
    }


# =============================================================================
# H1 Test: Within-Domain Classification
# =============================================================================

def test_h1(
    data_dir: str,
    model: str,
    tasks: List[str] = ['gsm8k', 'humaneval', 'logiqa']
) -> Dict:
    """
    H1: Can we distinguish correct vs incorrect within each domain?

    For each task, run 5-fold CV to measure classification accuracy.
    """
    print("\n" + "="*70)
    print("H1 TEST: Within-Domain Classification")
    print("="*70)

    results = {}

    for task in tasks:
        filepath = Path(data_dir) / model / f"{task}_trajectories.h5"
        if not filepath.exists():
            print(f"\n⚠ {task}: File not found at {filepath}")
            continue

        print(f"\n{'─'*70}")
        print(f"Task: {task.upper()}")
        print(f"{'─'*70}")

        # Load data
        trajectories, labels, _ = load_trajectories_with_labels(str(filepath))

        # Check class balance
        n_correct = labels.sum()
        n_incorrect = len(labels) - n_correct
        print(f"  Class balance: {n_correct} correct, {n_incorrect} incorrect")

        if n_correct < 10 or n_incorrect < 10:
            print(f"  ⚠ Skipping: insufficient samples in one class")
            continue

        # Extract features
        print("  Extracting features...")
        X = extract_features_batch(trajectories)
        y = labels.astype(int)
        print(f"  Feature matrix: {X.shape}")

        # Cross-validation
        task_results = {}
        for clf_type in ['logistic', 'rf']:
            print(f"\n  {clf_type.upper()}:")
            cv_results = cross_validate(X, y, classifier_type=clf_type)
            task_results[clf_type] = cv_results
            print(f"    Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
            print(f"    AUC:      {cv_results['auc_mean']:.3f} ± {cv_results['auc_std']:.3f}")

        results[task] = task_results

    return results


# =============================================================================
# H2 Test: Cross-Domain Transfer
# =============================================================================

def test_h2(
    data_dir: str,
    model: str,
    tasks: List[str] = ['gsm8k', 'humaneval', 'logiqa']
) -> Dict:
    """
    H2: Does a classifier trained on one domain transfer to another?

    Train on each domain, test on all others.
    """
    print("\n" + "="*70)
    print("H2 TEST: Cross-Domain Transfer")
    print("="*70)

    # Load all data
    data = {}
    for task in tasks:
        filepath = Path(data_dir) / model / f"{task}_trajectories.h5"
        if not filepath.exists():
            print(f"⚠ {task}: File not found")
            continue

        trajectories, labels, _ = load_trajectories_with_labels(str(filepath))
        X = extract_features_batch(trajectories)
        y = labels.astype(int)
        data[task] = {'X': X, 'y': y}

    if len(data) < 2:
        print("⚠ Need at least 2 tasks for transfer test")
        return {}

    results = {}

    for clf_type in ['logistic', 'rf']:
        print(f"\n{'─'*70}")
        print(f"Classifier: {clf_type.upper()}")
        print(f"{'─'*70}")

        transfer_matrix = {}

        for train_task in data.keys():
            transfer_matrix[train_task] = {}
            X_train, y_train = data[train_task]['X'], data[train_task]['y']

            # Standardize based on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Train classifier
            if clf_type == 'logistic':
                clf = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=42)
            else:
                clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)

            clf.fit(X_train_scaled, y_train)

            for test_task in data.keys():
                X_test, y_test = data[test_task]['X'], data[test_task]['y']
                X_test_scaled = scaler.transform(X_test)

                y_pred = clf.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)

                try:
                    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                except:
                    auc = None

                transfer_matrix[train_task][test_task] = {
                    'accuracy': acc,
                    'auc': auc,
                }

        # Print transfer matrix
        print(f"\n  Transfer Matrix (rows=train, cols=test):")
        print(f"  {'':12}", end='')
        for task in data.keys():
            print(f"{task:12}", end='')
        print()

        for train_task in data.keys():
            print(f"  {train_task:12}", end='')
            for test_task in data.keys():
                acc = transfer_matrix[train_task][test_task]['accuracy']
                marker = '*' if train_task == test_task else ' '
                print(f"{acc:.3f}{marker}      ", end='')
            print()

        results[clf_type] = transfer_matrix

    return results


# =============================================================================
# Feature Importance Analysis
# =============================================================================

def analyze_feature_importance(
    data_dir: str,
    model: str,
    task: str = 'gsm8k'
) -> Dict:
    """Analyze which features are most predictive."""

    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    filepath = Path(data_dir) / model / f"{task}_trajectories.h5"
    trajectories, labels, _ = load_trajectories_with_labels(str(filepath))
    X = extract_features_batch(trajectories)
    y = labels.astype(int)

    # Get feature names
    n_layers = trajectories.shape[2]
    feature_names = get_feature_names(n_layers)

    # Train logistic regression on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=42)
    clf.fit(X_scaled, y)

    # Get importance
    importance = np.abs(clf.coef_[0])

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]

    print(f"\nTop 20 most predictive features for {task}:")
    print("-" * 50)
    for i, idx in enumerate(sorted_idx[:20]):
        print(f"  {i+1:2}. {feature_names[idx]:25} {importance[idx]:.4f}")

    return {
        'feature_names': feature_names,
        'importance': importance.tolist(),
        'sorted_idx': sorted_idx.tolist(),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="H1/H2 Hypothesis Testing")
    parser.add_argument('--data-dir', type=str, default='data/trajectories',
                       help='Directory containing trajectory data')
    parser.add_argument('--model', type=str, default='olmo3_sft',
                       help='Model to analyze')
    parser.add_argument('--tasks', type=str, default='gsm8k,humaneval,logiqa',
                       help='Comma-separated tasks')
    parser.add_argument('--output', type=str, default='results/h1_h2_results.json',
                       help='Output file for results')
    parser.add_argument('--skip-h1', action='store_true', help='Skip H1 test')
    parser.add_argument('--skip-h2', action='store_true', help='Skip H2 test')

    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(',')]

    print("="*70)
    print("H1/H2 HYPOTHESIS TESTING")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Tasks: {tasks}")
    print(f"Data dir: {args.data_dir}")

    results = {
        'model': args.model,
        'tasks': tasks,
    }

    # H1 Test
    if not args.skip_h1:
        h1_results = test_h1(args.data_dir, args.model, tasks)
        results['h1'] = h1_results

    # H2 Test
    if not args.skip_h2:
        h2_results = test_h2(args.data_dir, args.model, tasks)
        results['h2'] = h2_results

    # Feature importance
    if 'gsm8k' in tasks:
        importance_results = analyze_feature_importance(args.data_dir, args.model, 'gsm8k')
        results['feature_importance'] = importance_results

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if 'h1' in results:
        print("\nH1 (Within-Domain):")
        for task, task_results in results['h1'].items():
            lr_acc = task_results['logistic']['accuracy_mean']
            rf_acc = task_results['rf']['accuracy_mean']
            status = "✓ PASS" if max(lr_acc, rf_acc) > 0.60 else "✗ FAIL"
            print(f"  {task:12} LR: {lr_acc:.3f}, RF: {rf_acc:.3f}  {status}")

    if 'h2' in results and results['h2']:
        print("\nH2 (Cross-Domain Transfer):")
        lr_matrix = results['h2'].get('logistic', {})
        for train_task, test_results in lr_matrix.items():
            transfers = []
            for test_task, metrics in test_results.items():
                if train_task != test_task:
                    transfers.append(metrics['accuracy'])
            if transfers:
                mean_transfer = np.mean(transfers)
                status = "✓ PASS" if mean_transfer > 0.55 else "✗ FAIL"
                print(f"  Train on {train_task:10} → Others: {mean_transfer:.3f}  {status}")


if __name__ == "__main__":
    main()
