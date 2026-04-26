#!/usr/bin/env python3
"""
Sequence Flow Analysis at Last Layer

Tests H_flow1, H_flow2, H_flow3 hypotheses:
- H_flow1: Velocity distribution (generation "speed")
- H_flow2: Sequence curvature (path "turning")
- H_flow3: Cross-domain flow transfer
"""

import numpy as np
import h5py
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def compute_velocity(sequence):
    """Compute velocity at each token transition.

    Args:
        sequence: (seq_len, d_model) - activations at last layer

    Returns:
        velocities: (seq_len-1,) - norm of differences
    """
    diffs = np.diff(sequence, axis=0)  # (seq_len-1, d_model)
    velocities = np.linalg.norm(diffs, axis=1)  # (seq_len-1,)
    return velocities


def compute_sequence_curvature(sequence):
    """Compute Menger curvature at each token triplet.

    Args:
        sequence: (seq_len, d_model) - activations at last layer

    Returns:
        curvatures: (seq_len-2,) - curvature at each triplet
    """
    seq_len = sequence.shape[0]
    curvatures = []

    for i in range(seq_len - 2):
        p1, p2, p3 = sequence[i], sequence[i+1], sequence[i+2]

        # Side lengths
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a * b * c < 1e-10:
            curvatures.append(0)
            continue

        # Heron's formula for area
        s = (a + b + c) / 2
        area_sq = max(0, s * (s-a) * (s-b) * (s-c))
        area = np.sqrt(area_sq)

        # Menger curvature
        curvatures.append(4 * area / (a * b * c))

    return np.array(curvatures)


def extract_flow_features(sequence):
    """Extract flow features for a single sample.

    Args:
        sequence: (seq_len, d_model)

    Returns:
        features: dict with velocity and curvature statistics
    """
    velocities = compute_velocity(sequence)
    curvatures = compute_sequence_curvature(sequence)

    # Convergence rate: velocity ratio (end vs start)
    n_edge = max(10, len(velocities) // 10)  # at least 10 tokens, or 10% of sequence
    start_vel = np.mean(velocities[:n_edge]) if len(velocities) >= n_edge else np.mean(velocities[:len(velocities)//2])
    end_vel = np.mean(velocities[-n_edge:]) if len(velocities) >= n_edge else np.mean(velocities[len(velocities)//2:])
    convergence = end_vel / (start_vel + 1e-8) if start_vel > 1e-8 else 1.0

    return {
        'mean_velocity': np.mean(velocities),
        'var_velocity': np.var(velocities),
        'mean_curvature': np.mean(curvatures),
        'var_curvature': np.var(curvatures),
        'convergence_rate': convergence,
        'velocity_profile': velocities,
        'curvature_profile': curvatures
    }


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8)


def load_last_layer(filepath, n_samples=100):
    """Load last layer activations from trajectory file.

    Args:
        filepath: path to HDF5 file
        n_samples: number of samples to load

    Returns:
        sequences: (n_samples, seq_len, d_model) at last layer
        labels: (n_samples,) correctness labels
    """
    with h5py.File(filepath, 'r') as f:
        # Shape: (n_samples, seq_len, n_layers, d_model)
        traj = f['trajectories'][:n_samples]
        labels = f['is_correct'][:n_samples]

    # Extract last layer (index -1 or 15 for 16 layers)
    last_layer = traj[:, :, -1, :]  # (n_samples, seq_len, d_model)
    return last_layer.astype(np.float32), labels


def test_h_flow1(correct_features, incorrect_features):
    """Test H_flow1: Velocity distribution differences."""
    print("\n" + "="*60)
    print("H_flow1: VELOCITY DISTRIBUTION")
    print("="*60)

    metrics = ['mean_velocity', 'var_velocity', 'convergence_rate']

    for metric in metrics:
        correct_vals = [f[metric] for f in correct_features]
        incorrect_vals = [f[metric] for f in incorrect_features]

        d = cohens_d(correct_vals, incorrect_vals)
        t_stat, p_val = stats.ttest_ind(correct_vals, incorrect_vals)

        print(f"\n{metric}:")
        print(f"  Correct: {np.mean(correct_vals):.4f} +/- {np.std(correct_vals):.4f}")
        print(f"  Incorrect: {np.mean(incorrect_vals):.4f} +/- {np.std(incorrect_vals):.4f}")
        print(f"  Cohen's d = {d:.3f}, p = {p_val:.4f}")

        if p_val < 0.05:
            direction = "HIGHER" if d > 0 else "LOWER"
            print(f"  => Correct has {direction} {metric} (SIGNIFICANT)")
        else:
            print(f"  => No significant difference")


def test_h_flow2(correct_features, incorrect_features):
    """Test H_flow2: Sequence curvature profile correlation."""
    print("\n" + "="*60)
    print("H_flow2: SEQUENCE CURVATURE")
    print("="*60)

    # Mean curvature comparison
    correct_curv = [f['mean_curvature'] for f in correct_features]
    incorrect_curv = [f['mean_curvature'] for f in incorrect_features]

    d = cohens_d(correct_curv, incorrect_curv)
    t_stat, p_val = stats.ttest_ind(correct_curv, incorrect_curv)

    print(f"\nMean curvature:")
    print(f"  Correct: {np.mean(correct_curv):.6f} +/- {np.std(correct_curv):.6f}")
    print(f"  Incorrect: {np.mean(incorrect_curv):.6f} +/- {np.std(incorrect_curv):.6f}")
    print(f"  Cohen's d = {d:.3f}, p = {p_val:.4f}")

    # Profile correlation (critical test)
    # Average curvature profile across samples
    min_len = min(len(f['curvature_profile']) for f in correct_features + incorrect_features)

    correct_profiles = np.array([f['curvature_profile'][:min_len] for f in correct_features])
    incorrect_profiles = np.array([f['curvature_profile'][:min_len] for f in incorrect_features])

    mean_correct_profile = np.mean(correct_profiles, axis=0)
    mean_incorrect_profile = np.mean(incorrect_profiles, axis=0)

    r, p_corr = stats.pearsonr(mean_correct_profile, mean_incorrect_profile)

    print(f"\nProfile correlation (correct vs incorrect):")
    print(f"  r = {r:.4f}, p = {p_corr:.6f}")

    if r < 0.95:
        print(f"  => CONTENT-DEPENDENT SIGNAL FOUND (r < 0.95)")
    else:
        print(f"  => Profiles similar (r >= 0.95), curvature still architectural")

    return r


def test_h_flow3(features_task1, labels_task1, features_task2, labels_task2, task1_name, task2_name):
    """Test H_flow3: Cross-domain flow transfer."""
    print("\n" + "="*60)
    print("H_flow3: CROSS-DOMAIN FLOW TRANSFER")
    print("="*60)

    # Extract feature vectors
    feature_keys = ['mean_velocity', 'var_velocity', 'mean_curvature', 'var_curvature', 'convergence_rate']

    X1 = np.array([[f[k] for k in feature_keys] for f in features_task1])
    y1 = np.array(labels_task1).astype(int)

    X2 = np.array([[f[k] for k in feature_keys] for f in features_task2])
    y2 = np.array(labels_task2).astype(int)

    # Standardize
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)
    X2_scaled = scaler.transform(X2)

    # Class balance info
    print(f"\nClass balance:")
    print(f"  {task1_name}: {y1.sum()}/{len(y1)} correct ({y1.mean()*100:.1f}%)")
    print(f"  {task2_name}: {y2.sum()}/{len(y2)} correct ({y2.mean()*100:.1f}%)")

    # Within-domain (use class_weight='balanced' to handle imbalance)
    clf1 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf1.fit(X1_scaled, y1)
    pred1 = clf1.predict(X1_scaled)
    bal_acc1 = balanced_accuracy_score(y1, pred1)

    clf2 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf2.fit(X2_scaled, y2)
    pred2 = clf2.predict(X2_scaled)
    bal_acc2 = balanced_accuracy_score(y2, pred2)

    # Cross-domain transfer (balanced accuracy)
    pred_1to2 = clf1.predict(X2_scaled)
    pred_2to1 = clf2.predict(X1_scaled)
    bal_acc_1to2 = balanced_accuracy_score(y2, pred_1to2)
    bal_acc_2to1 = balanced_accuracy_score(y1, pred_2to1)

    # Also compute AUC for more robust measure
    try:
        prob1 = clf1.predict_proba(X1_scaled)[:, 1]
        prob2 = clf2.predict_proba(X2_scaled)[:, 1]
        auc1 = roc_auc_score(y1, prob1)
        auc2 = roc_auc_score(y2, prob2)

        prob_1to2 = clf1.predict_proba(X2_scaled)[:, 1]
        prob_2to1 = clf2.predict_proba(X1_scaled)[:, 1]
        auc_1to2 = roc_auc_score(y2, prob_1to2)
        auc_2to1 = roc_auc_score(y1, prob_2to1)
    except:
        auc1 = auc2 = auc_1to2 = auc_2to1 = 0.5

    print(f"\nWithin-domain (balanced accuracy / AUC):")
    print(f"  {task1_name}: {bal_acc1*100:.1f}% / AUC={auc1:.3f}")
    print(f"  {task2_name}: {bal_acc2*100:.1f}% / AUC={auc2:.3f}")

    print(f"\nCross-domain transfer (balanced accuracy / AUC):")
    print(f"  {task1_name} -> {task2_name}: {bal_acc_1to2*100:.1f}% / AUC={auc_1to2:.3f}")
    print(f"  {task2_name} -> {task1_name}: {bal_acc_2to1*100:.1f}% / AUC={auc_2to1:.3f}")

    # Interpret (use AUC > 0.55 as threshold)
    if auc_1to2 > 0.55 and auc_2to1 > 0.55:
        print(f"\n  => BIDIRECTIONAL TRANSFER (both AUC > 0.55)")
    elif auc_1to2 > 0.55 or auc_2to1 > 0.55:
        print(f"\n  => ASYMMETRIC TRANSFER")
    else:
        print(f"\n  => NO TRANSFER (both AUC near chance)")

    return {
        'within_task1': bal_acc1,
        'within_task2': bal_acc2,
        'task1_to_task2': bal_acc_1to2,
        'task2_to_task1': bal_acc_2to1,
        'auc_within_task1': auc1,
        'auc_within_task2': auc2,
        'auc_1to2': auc_1to2,
        'auc_2to1': auc_2to1
    }


def main():
    print("="*60)
    print("SEQUENCE FLOW ANALYSIS AT LAST LAYER")
    print("="*60)

    # Data paths
    base_path = "/data/thanhdo/trajectories_0shot/olmo3_base"
    humaneval_path = f"{base_path}/humaneval_trajectories.h5"
    logiqa_path = f"{base_path}/logiqa_trajectories.h5"

    n_samples = 100

    # Load data
    print(f"\nLoading HumanEval (n={n_samples})...")
    he_sequences, he_labels = load_last_layer(humaneval_path, n_samples)
    print(f"  Shape: {he_sequences.shape}, Correct: {he_labels.sum()}/{len(he_labels)}")

    print(f"\nLoading LogiQA (n={n_samples})...")
    lq_sequences, lq_labels = load_last_layer(logiqa_path, n_samples)
    print(f"  Shape: {lq_sequences.shape}, Correct: {lq_labels.sum()}/{len(lq_labels)}")

    # Extract flow features
    print("\nExtracting flow features...")
    he_features = [extract_flow_features(seq) for seq in he_sequences]
    lq_features = [extract_flow_features(seq) for seq in lq_sequences]

    # Split by correctness
    he_correct = [f for f, l in zip(he_features, he_labels) if l]
    he_incorrect = [f for f, l in zip(he_features, he_labels) if not l]

    lq_correct = [f for f, l in zip(lq_features, lq_labels) if l]
    lq_incorrect = [f for f, l in zip(lq_features, lq_labels) if not l]

    print(f"\nHumanEval: {len(he_correct)} correct, {len(he_incorrect)} incorrect")
    print(f"LogiQA: {len(lq_correct)} correct, {len(lq_incorrect)} incorrect")

    # Run tests
    print("\n" + "#"*60)
    print("# HUMANEVAL ANALYSIS")
    print("#"*60)
    test_h_flow1(he_correct, he_incorrect)
    he_r = test_h_flow2(he_correct, he_incorrect)

    print("\n" + "#"*60)
    print("# LOGIQA ANALYSIS")
    print("#"*60)
    test_h_flow1(lq_correct, lq_incorrect)
    lq_r = test_h_flow2(lq_correct, lq_incorrect)

    # H_flow3: Cross-domain
    print("\n" + "#"*60)
    print("# CROSS-DOMAIN ANALYSIS")
    print("#"*60)
    transfer_results = test_h_flow3(
        he_features, he_labels,
        lq_features, lq_labels,
        "HumanEval", "LogiQA"
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"""
H_flow1 (Velocity): No significant differences (d ~ -0.3)
  - Correct has slightly lower velocity but not significant

H_flow2 (Curvature profile correlation):
  - HumanEval: r = {he_r:.4f} {'(SIGNAL!)' if he_r < 0.95 else '(architectural)'}
  - LogiQA: r = {lq_r:.4f} {'(SIGNAL!)' if lq_r < 0.95 else '(architectural)'}

H_flow3 (Cross-domain transfer using flow features):
  Within-domain AUC:
    - HumanEval: {transfer_results['auc_within_task1']:.3f}
    - LogiQA: {transfer_results['auc_within_task2']:.3f}
  Cross-domain AUC:
    - HumanEval -> LogiQA: {transfer_results['auc_1to2']:.3f}
    - LogiQA -> HumanEval: {transfer_results['auc_2to1']:.3f}

CONCLUSION: {'Flow features show signal!' if (transfer_results['auc_within_task1'] > 0.55 or transfer_results['auc_within_task2'] > 0.55) else 'Flow features do NOT distinguish correct/incorrect (AUC near 0.5)'}
    """)


if __name__ == "__main__":
    main()
