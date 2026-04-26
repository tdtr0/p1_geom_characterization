#!/usr/bin/env python3
"""
H1 Comparison: 0-shot vs 8-shot trajectory analysis

Tests whether 0-shot (no examples) shows stronger geometric signals than 8-shot.
Hypothesis: 0-shot may show larger trajectory differences because examples provide guidance.
"""

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

BASE_0SHOT = '/data/thanhdo/trajectories_0shot'
BASE_8SHOT = '/data/thanhdo/trajectories_8shot'
MODELS = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']

def load_data(base_path, model, task, suffix=''):
    h5_path = f'{base_path}/{model}/{task}_trajectories{suffix}.h5'
    try:
        with h5py.File(h5_path, 'r') as f:
            trajectories = f['trajectories'][:]
            labels = f['is_correct'][:]
        return trajectories, labels
    except Exception as e:
        return None, None

def compute_features(trajectories):
    n_samples, seq_len, n_layers, d_model = trajectories.shape
    features = []
    
    for i in range(n_samples):
        traj = trajectories[i]
        layer_diffs = np.diff(traj, axis=1)
        layer_velocities = np.linalg.norm(layer_diffs, axis=2).mean(axis=0)
        token_evolution = np.linalg.norm(traj[:, -1, :] - traj[:, 0, :], axis=1).mean()
        path_length = np.sum(np.linalg.norm(layer_diffs, axis=2))
        
        if n_layers > 2:
            second_diffs = np.diff(layer_diffs, axis=1)
            curvatures = np.linalg.norm(second_diffs, axis=2).mean(axis=0)
        else:
            curvatures = np.array([0])
        
        layer_vars = traj.var(axis=0).mean(axis=1)
        final_norm = np.linalg.norm(traj[:, -1, :], axis=1).mean()
        velocity_stability = np.var(layer_velocities)
        
        feat = np.concatenate([
            layer_velocities, [token_evolution], [path_length],
            curvatures, layer_vars, [final_norm], [velocity_stability]
        ])
        features.append(feat)
    
    return np.array(features)

def evaluate(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
    baseline = max(y.mean(), 1 - y.mean())
    return scores.mean(), scores.std(), baseline

def main():
    print('='*75)
    print('H1 COMPARISON: 0-shot vs 8-shot Trajectory Analysis')
    print('='*75)
    print('\nHypothesis: 0-shot may show stronger geometric signals')
    print('(8-shot examples may guide trajectories, reducing variation)')
    
    results = []
    
    # GSM8K comparison (all 4 models)
    print('\n' + '-'*75)
    print('GSM8K COMPARISON (all 4 models)')
    print('-'*75)
    
    for model in MODELS:
        print(f'\n{model}:')
        
        # 0-shot
        traj_0, labels_0 = load_data(BASE_0SHOT, model, 'gsm8k')
        if traj_0 is not None:
            X_0 = compute_features(traj_0)
            acc_0, std_0, base_0 = evaluate(X_0, labels_0.astype(int))
            lift_0 = acc_0 - base_0
            print(f'  0-shot: Acc={acc_0*100:.1f}%, Baseline={base_0*100:.1f}%, Lift={lift_0*100:+.1f}%')
        else:
            acc_0, lift_0 = None, None
            print(f'  0-shot: MISSING')
        
        # 8-shot
        traj_8, labels_8 = load_data(BASE_8SHOT, model, 'gsm8k', '_8shot')
        if traj_8 is not None:
            X_8 = compute_features(traj_8)
            acc_8, std_8, base_8 = evaluate(X_8, labels_8.astype(int))
            lift_8 = acc_8 - base_8
            print(f'  8-shot: Acc={acc_8*100:.1f}%, Baseline={base_8*100:.1f}%, Lift={lift_8*100:+.1f}%')
        else:
            acc_8, lift_8 = None, None
            print(f'  8-shot: MISSING')
        
        if lift_0 is not None and lift_8 is not None:
            diff = lift_0 - lift_8
            winner = '0-shot' if diff > 0 else '8-shot'
            print(f'  → {winner} has {abs(diff)*100:.1f}% higher lift')
            results.append({'model': model, 'task': 'gsm8k', 'lift_0shot': lift_0, 'lift_8shot': lift_8, 'diff': diff})
    
    # LogiQA comparison (only olmo3_base has 0-shot)
    print('\n' + '-'*75)
    print('LogiQA COMPARISON (only olmo3_base has 0-shot)')
    print('-'*75)
    
    model = 'olmo3_base'
    print(f'\n{model}:')
    
    traj_0, labels_0 = load_data(BASE_0SHOT, model, 'logiqa')
    if traj_0 is not None:
        X_0 = compute_features(traj_0)
        acc_0, std_0, base_0 = evaluate(X_0, labels_0.astype(int))
        lift_0 = acc_0 - base_0
        print(f'  0-shot: Acc={acc_0*100:.1f}%, Baseline={base_0*100:.1f}%, Lift={lift_0*100:+.1f}%')
    else:
        lift_0 = None
        print(f'  0-shot: MISSING')
    
    traj_8, labels_8 = load_data(BASE_8SHOT, model, 'logiqa', '_8shot')
    if traj_8 is not None:
        X_8 = compute_features(traj_8)
        acc_8, std_8, base_8 = evaluate(X_8, labels_8.astype(int))
        lift_8 = acc_8 - base_8
        print(f'  8-shot: Acc={acc_8*100:.1f}%, Baseline={base_8*100:.1f}%, Lift={lift_8*100:+.1f}%')
    else:
        lift_8 = None
        print(f'  8-shot: MISSING')
    
    if lift_0 is not None and lift_8 is not None:
        diff = lift_0 - lift_8
        winner = '0-shot' if diff > 0 else '8-shot'
        print(f'  → {winner} has {abs(diff)*100:.1f}% higher lift')
        results.append({'model': model, 'task': 'logiqa', 'lift_0shot': lift_0, 'lift_8shot': lift_8, 'diff': diff})
    
    # Summary
    print('\n' + '='*75)
    print('SUMMARY: 0-shot vs 8-shot')
    print('='*75)
    
    if results:
        avg_0shot = np.mean([r['lift_0shot'] for r in results])
        avg_8shot = np.mean([r['lift_8shot'] for r in results])
        avg_diff = np.mean([r['diff'] for r in results])
        
        print(f'Average 0-shot lift: {avg_0shot*100:+.1f}%')
        print(f'Average 8-shot lift: {avg_8shot*100:+.1f}%')
        print(f'Average difference (0shot - 8shot): {avg_diff*100:+.1f}%')
        
        if avg_diff > 0.02:
            print('\n✅ 0-shot shows STRONGER geometric signals than 8-shot')
            print('   (Examples may guide trajectories, reducing distinguishability)')
        elif avg_diff < -0.02:
            print('\n⚠️  8-shot shows STRONGER signals than 0-shot')
            print('   (Unexpected - examples may amplify pattern differences)')
        else:
            print('\n~  Similar signal strength in both conditions')

if __name__ == '__main__':
    main()
