#!/usr/bin/env python3
"""
H1 Analysis: Do correct vs incorrect solutions have distinguishable trajectory dynamics?

Tests whether geometric features of activation trajectories can predict correctness.
Uses 8-shot GSM8K and LogiQA data (4 models × 2 tasks = 8 datasets).
"""

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Data paths
BASE_8SHOT = '/data/thanhdo/trajectories_8shot'
MODELS = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']
TASKS = ['gsm8k', 'logiqa']

def load_data(model, task):
    """Load trajectories and labels from HDF5 file."""
    h5_path = f'{BASE_8SHOT}/{model}/{task}_trajectories_8shot.h5'
    try:
        with h5py.File(h5_path, 'r') as f:
            trajectories = f['trajectories'][:]  # (n_samples, seq_len, n_layers, d_model)
            labels = f['is_correct'][:]
        return trajectories, labels
    except Exception as e:
        print(f'  Error loading {model}/{task}: {e}')
        return None, None

def compute_geometric_features(trajectories):
    """Compute geometric features from trajectories."""
    n_samples, seq_len, n_layers, d_model = trajectories.shape
    features = []
    
    for i in range(n_samples):
        traj = trajectories[i]  # (seq_len, n_layers, d_model)
        
        # 1. Layer-wise velocity (mean across tokens)
        layer_diffs = np.diff(traj, axis=1)  # (seq_len, n_layers-1, d_model)
        layer_velocities = np.linalg.norm(layer_diffs, axis=2).mean(axis=0)  # (n_layers-1,)
        
        # 2. Token-wise evolution (how much each token changes across layers)
        token_evolution = np.linalg.norm(traj[:, -1, :] - traj[:, 0, :], axis=1).mean()
        
        # 3. Trajectory length (total path length through layer space)
        path_length = np.sum(np.linalg.norm(layer_diffs, axis=2))
        
        # 4. Curvature proxy: second derivative magnitude
        if n_layers > 2:
            second_diffs = np.diff(layer_diffs, axis=1)  # (seq_len, n_layers-2, d_model)
            curvatures = np.linalg.norm(second_diffs, axis=2).mean(axis=0)  # (n_layers-2,)
        else:
            curvatures = np.array([0])
        
        # 5. Layer-wise variance (how spread out activations are at each layer)
        layer_vars = traj.var(axis=0).mean(axis=1)  # (n_layers,)
        
        # 6. Final layer norm (output representation magnitude)
        final_norm = np.linalg.norm(traj[:, -1, :], axis=1).mean()
        
        # 7. Trajectory stability (variance of velocities)
        velocity_stability = np.var(layer_velocities)
        
        # Concatenate all features
        feat = np.concatenate([
            layer_velocities,          # (n_layers-1,)
            [token_evolution],         # (1,)
            [path_length],             # (1,)
            curvatures,                # (n_layers-2,)
            layer_vars,                # (n_layers,)
            [final_norm],              # (1,)
            [velocity_stability]       # (1,)
        ])
        features.append(feat)
    
    return np.array(features)

def evaluate_h1(model, task, trajectories, labels):
    """Test H1: can geometric features predict correctness?"""
    print(f'\n  Computing features for {model}/{task}...')
    
    # Compute geometric features
    X = compute_geometric_features(trajectories)
    y = labels.astype(int)
    
    # Handle class imbalance info
    n_correct = y.sum()
    n_total = len(y)
    balance = n_correct / n_total
    print(f'  Class balance: {n_correct}/{n_total} correct ({balance*100:.1f}%)')
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test with Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
    
    # Test with Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='accuracy')
    
    # Compute AUC-ROC
    try:
        rf_auc = cross_val_score(rf, X_scaled, y, cv=cv, scoring='roc_auc').mean()
        lr_auc = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc').mean()
    except:
        rf_auc = lr_auc = 0.5
    
    # Baseline: predict majority class
    majority_baseline = max(balance, 1 - balance)
    
    return {
        'model': model,
        'task': task,
        'n_samples': n_total,
        'balance': balance,
        'rf_accuracy': rf_scores.mean(),
        'rf_std': rf_scores.std(),
        'rf_auc': rf_auc,
        'lr_accuracy': lr_scores.mean(),
        'lr_std': lr_scores.std(),
        'lr_auc': lr_auc,
        'baseline': majority_baseline,
        'rf_lift': rf_scores.mean() - majority_baseline,
        'n_features': X.shape[1]
    }

def main():
    print('='*70)
    print('H1 ANALYSIS: Distinguishing Correct vs Incorrect Trajectories')
    print('='*70)
    
    results = []
    
    for model in MODELS:
        for task in TASKS:
            print(f'\nProcessing {model}/{task}...')
            
            trajectories, labels = load_data(model, task)
            if trajectories is None:
                continue
            
            result = evaluate_h1(model, task, trajectories, labels)
            results.append(result)
            
            print(f'  RF Accuracy: {result["rf_accuracy"]*100:.1f}% (+/- {result["rf_std"]*100:.1f}%)')
            print(f'  RF AUC-ROC: {result["rf_auc"]:.3f}')
            print(f'  LR Accuracy: {result["lr_accuracy"]*100:.1f}% (+/- {result["lr_std"]*100:.1f}%)')
            print(f'  Baseline: {result["baseline"]*100:.1f}%')
            print(f'  Lift over baseline: {result["rf_lift"]*100:.1f}%')
    
    # Summary table
    print('\n' + '='*70)
    print('H1 RESULTS SUMMARY')
    print('='*70)
    print(f'{"Model":<15} {"Task":<10} {"RF Acc":<10} {"Baseline":<10} {"Lift":<10} {"AUC":<8}')
    print('-'*70)
    
    for r in results:
        print(f'{r["model"]:<15} {r["task"]:<10} {r["rf_accuracy"]*100:.1f}%     {r["baseline"]*100:.1f}%     {r["rf_lift"]*100:+.1f}%    {r["rf_auc"]:.3f}')
    
    # Aggregate statistics
    print('\n' + '='*70)
    print('AGGREGATE ANALYSIS')
    print('='*70)
    
    avg_lift = np.mean([r['rf_lift'] for r in results])
    avg_auc = np.mean([r['rf_auc'] for r in results])
    
    print(f'Mean lift over baseline: {avg_lift*100:.1f}%')
    print(f'Mean AUC-ROC: {avg_auc:.3f}')
    
    # Per-model analysis
    print('\nPer-model average lift:')
    for model in MODELS:
        model_results = [r for r in results if r['model'] == model]
        if model_results:
            model_lift = np.mean([r['rf_lift'] for r in model_results])
            print(f'  {model}: {model_lift*100:+.1f}%')
    
    # Per-task analysis
    print('\nPer-task average lift:')
    for task in TASKS:
        task_results = [r for r in results if r['task'] == task]
        if task_results:
            task_lift = np.mean([r['rf_lift'] for r in task_results])
            print(f'  {task}: {task_lift*100:+.1f}%')
    
    # H1 conclusion
    print('\n' + '='*70)
    print('H1 CONCLUSION')
    print('='*70)
    if avg_lift > 0.05:
        print('✅ H1 SUPPORTED: Geometric features predict correctness above baseline')
        print(f'   Average lift: {avg_lift*100:.1f}% above majority baseline')
    elif avg_lift > 0:
        print('⚠️  H1 WEAKLY SUPPORTED: Small but positive lift over baseline')
        print(f'   Average lift: {avg_lift*100:.1f}%')
    else:
        print('❌ H1 NOT SUPPORTED: Geometric features do not predict correctness')
    
    if avg_auc > 0.6:
        print(f'   AUC-ROC {avg_auc:.3f} indicates discriminative power')
    elif avg_auc > 0.55:
        print(f'   AUC-ROC {avg_auc:.3f} indicates weak discriminative power')
    else:
        print(f'   AUC-ROC {avg_auc:.3f} indicates near-random classification')

if __name__ == '__main__':
    main()
