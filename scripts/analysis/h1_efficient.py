#!/usr/bin/env python3
"""
H1 Efficient Analysis: Memory-efficient trajectory analysis using batched processing.

Processes trajectories in chunks to avoid 100GB+ memory usage.
"""

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

BASE_8SHOT = '/data/thanhdo/trajectories_8shot'
BASE_0SHOT = '/data/thanhdo/trajectories_0shot'
MODELS = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']

def compute_features_single(traj):
    """Compute features for a single trajectory (seq_len, n_layers, d_model)."""
    seq_len, n_layers, d_model = traj.shape
    
    # Layer-wise velocity
    layer_diffs = np.diff(traj, axis=1)  # (seq_len, n_layers-1, d_model)
    layer_velocities = np.linalg.norm(layer_diffs, axis=2).mean(axis=0)  # (n_layers-1,)
    
    # Token evolution
    token_evolution = np.linalg.norm(traj[:, -1, :] - traj[:, 0, :], axis=1).mean()
    
    # Path length
    path_length = np.sum(np.linalg.norm(layer_diffs, axis=2))
    
    # Curvature
    if n_layers > 2:
        second_diffs = np.diff(layer_diffs, axis=1)
        curvatures = np.linalg.norm(second_diffs, axis=2).mean(axis=0)
    else:
        curvatures = np.array([0])
    
    # Layer variance
    layer_vars = traj.var(axis=0).mean(axis=1)
    
    # Final norm
    final_norm = np.linalg.norm(traj[:, -1, :], axis=1).mean()
    
    # Velocity stability
    velocity_stability = np.var(layer_velocities)
    
    return np.concatenate([
        layer_velocities, [token_evolution], [path_length],
        curvatures, layer_vars, [final_norm], [velocity_stability]
    ])

def extract_features_batched(h5_path, batch_size=20):
    """Extract features in batches to save memory."""
    with h5py.File(h5_path, 'r') as f:
        n_samples = f['trajectories'].shape[0]
        labels = f['is_correct'][:]
        
        features = []
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = f['trajectories'][i:end]
            
            for j in range(batch.shape[0]):
                feat = compute_features_single(batch[j])
                features.append(feat)
            
            print(f'    Processed {end}/{n_samples}', end='\r')
        
        print()
    return np.array(features), labels

def evaluate(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
    baseline = max(y.mean(), 1 - y.mean())
    return scores.mean(), scores.std(), baseline

def main():
    print('='*70)
    print('H1 EFFICIENT ANALYSIS')
    print('='*70)
    
    results = []
    
    # GSM8K 8-shot analysis
    print('\n--- GSM8K 8-shot ---')
    for model in MODELS:
        h5_path = f'{BASE_8SHOT}/{model}/gsm8k_trajectories_8shot.h5'
        print(f'\n{model}:')
        try:
            X, y = extract_features_batched(h5_path)
            y = y.astype(int)
            acc, std, base = evaluate(X, y)
            lift = acc - base
            print(f'  Accuracy: {acc*100:.1f}% +/- {std*100:.1f}%')
            print(f'  Baseline: {base*100:.1f}%, Lift: {lift*100:+.1f}%')
            results.append({'model': model, 'task': 'gsm8k', 'shot': '8shot', 'acc': acc, 'lift': lift})
        except Exception as e:
            print(f'  Error: {e}')
    
    # LogiQA 8-shot analysis
    print('\n--- LogiQA 8-shot ---')
    for model in MODELS:
        h5_path = f'{BASE_8SHOT}/{model}/logiqa_trajectories_8shot.h5'
        print(f'\n{model}:')
        try:
            X, y = extract_features_batched(h5_path)
            y = y.astype(int)
            acc, std, base = evaluate(X, y)
            lift = acc - base
            print(f'  Accuracy: {acc*100:.1f}% +/- {std*100:.1f}%')
            print(f'  Baseline: {base*100:.1f}%, Lift: {lift*100:+.1f}%')
            results.append({'model': model, 'task': 'logiqa', 'shot': '8shot', 'acc': acc, 'lift': lift})
        except Exception as e:
            print(f'  Error: {e}')
    
    # GSM8K 0-shot analysis (for comparison)
    print('\n--- GSM8K 0-shot (comparison) ---')
    for model in MODELS:
        h5_path = f'{BASE_0SHOT}/{model}/gsm8k_trajectories.h5'
        print(f'\n{model}:')
        try:
            X, y = extract_features_batched(h5_path)
            y = y.astype(int)
            acc, std, base = evaluate(X, y)
            lift = acc - base
            print(f'  Accuracy: {acc*100:.1f}% +/- {std*100:.1f}%')
            print(f'  Baseline: {base*100:.1f}%, Lift: {lift*100:+.1f}%')
            results.append({'model': model, 'task': 'gsm8k', 'shot': '0shot', 'acc': acc, 'lift': lift})
        except Exception as e:
            print(f'  Error: {e}')
    
    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    
    print(f'\n{"Model":<15} {"Task":<10} {"Shot":<8} {"Accuracy":<12} {"Lift":<10}')
    print('-'*60)
    for r in results:
        print(f'{r["model"]:<15} {r["task"]:<10} {r["shot"]:<8} {r["acc"]*100:.1f}%        {r["lift"]*100:+.1f}%')
    
    # 0-shot vs 8-shot comparison
    print('\n--- 0-shot vs 8-shot comparison (GSM8K) ---')
    for model in MODELS:
        r0 = next((r for r in results if r['model']==model and r['task']=='gsm8k' and r['shot']=='0shot'), None)
        r8 = next((r for r in results if r['model']==model and r['task']=='gsm8k' and r['shot']=='8shot'), None)
        if r0 and r8:
            diff = r0['lift'] - r8['lift']
            print(f'{model}: 0-shot lift={r0["lift"]*100:+.1f}%, 8-shot lift={r8["lift"]*100:+.1f}%, diff={diff*100:+.1f}%')

if __name__ == '__main__':
    main()
