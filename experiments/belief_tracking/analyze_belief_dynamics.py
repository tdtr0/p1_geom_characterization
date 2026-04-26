#!/usr/bin/env python3
"""
Belief State Tracking Analysis - Phase 1: Bootstrap with Existing Data

Tracks belief state evolution per-clause within model generations.

Key insight: Token probability P(next_token|context) ≠ belief state P(task_success|understanding).
We use activation-based correctness probes to track task-level belief, not token-level.

Hypotheses:
- H_belief: RLVR = smooth belief evolution, SFT = discrete jumps
- H_style: SFT jumps might encode style (formatting), not reasoning

Usage:
    python analyze_belief_dynamics.py --data-dir /path/to/trajectories --output-dir /path/to/results
"""

import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import h5py
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CLAUSE DETECTION
# ============================================================================

# Reasoning markers that indicate clause boundaries
CLAUSE_MARKERS = [
    r'\bFirst\b', r'\bSecond\b', r'\bThird\b', r'\bFinally\b',
    r'\bSo\b', r'\bTherefore\b', r'\bThus\b', r'\bHence\b',
    r'\bLet\b', r'\bWe\b', r'\bNow\b', r'\bNext\b',
    r'\bWait\b', r'\bActually\b', r'\bHmm\b', r'\bBut\b',
    r'\bThe answer is\b', r'\bTherefore,? the answer\b',
    r'####', r'\n\n',
]

CLAUSE_TYPES = {
    'restatement': [r'\bunderstand\b', r'\bproblem\b', r'\bgiven\b', r'\basked\b'],
    'setup': [r'\bLet\b', r'\bdefine\b', r'\bdenote\b', r'\bset\b'],
    'calculation': [r'\d+\s*[+\-*/×÷=]\s*\d+', r'\bequals?\b', r'\bis\b.*\d+'],
    'verification': [r'\bcheck\b', r'\bverify\b', r'\bcorrect\b'],
    'backtrack': [r'\bWait\b', r'\bActually\b', r'\bwrong\b', r'\bmistake\b'],
    'conclusion': [r'\bTherefore\b', r'\bThus\b', r'\banswer\b', r'####'],
    'formatting': [r'\*\*', r'```', r'\n\n\n'],
}


def detect_clause_boundaries(text: str, min_clause_len: int = 20) -> List[Tuple[int, int, str]]:
    """
    Detect clause boundaries in model output text.

    Returns list of (start_char, end_char, clause_type) tuples.
    """
    if not text or len(text) < min_clause_len:
        return [(0, len(text) if text else 0, 'unknown')]

    # Find all potential boundary positions
    boundaries = [0]

    # Sentence boundaries
    for match in re.finditer(r'[.!?]\s+', text):
        boundaries.append(match.end())

    # Reasoning markers
    for marker in CLAUSE_MARKERS:
        for match in re.finditer(marker, text, re.IGNORECASE):
            # Add boundary before the marker
            if match.start() > min_clause_len:
                boundaries.append(match.start())

    # Newlines (often indicate clause boundaries in CoT)
    for match in re.finditer(r'\n+', text):
        if match.start() > 0:
            boundaries.append(match.start())

    # Sort and deduplicate
    boundaries = sorted(set(boundaries))

    # Filter out boundaries that are too close together
    filtered = [boundaries[0]]
    for b in boundaries[1:]:
        if b - filtered[-1] >= min_clause_len:
            filtered.append(b)
    boundaries = filtered

    # Add end boundary
    if boundaries[-1] < len(text):
        boundaries.append(len(text))

    # Create clauses with types
    clauses = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        clause_text = text[start:end]
        clause_type = classify_clause(clause_text)
        clauses.append((start, end, clause_type))

    return clauses if clauses else [(0, len(text), 'unknown')]


def classify_clause(text: str) -> str:
    """Classify clause type based on content patterns."""
    text_lower = text.lower()

    scores = {}
    for ctype, patterns in CLAUSE_TYPES.items():
        score = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        scores[ctype] = score

    # Return highest scoring type, or 'other' if no matches
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return 'other'


def text_position_to_token_position(text: str, char_pos: int, tokenizer=None) -> int:
    """
    Convert character position to approximate token position.

    If no tokenizer provided, use heuristic (avg 4 chars per token).
    """
    if tokenizer is not None:
        # Use actual tokenizer
        prefix = text[:char_pos]
        return len(tokenizer.encode(prefix))
    else:
        # Heuristic: ~4 characters per token on average
        return char_pos // 4


# ============================================================================
# DATA LOADING
# ============================================================================

def decode_bytes(val):
    """Decode bytes to string if needed."""
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    return str(val)


def load_data(filepath: str, max_samples: int = None) -> Dict:
    """Load trajectories, labels, and model outputs from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        trajectories = f['trajectories'][:]

        # Get correctness labels
        if 'is_correct' in f:
            labels = f['is_correct'][:]
        elif 'correct' in f:
            labels = f['correct'][:]
        else:
            raise KeyError(f"No correctness labels. Keys: {list(f.keys())}")

        # Get model outputs (for clause detection)
        model_outputs = []
        if 'model_outputs' in f:
            for i in range(len(f['model_outputs'])):
                try:
                    output = decode_bytes(f['model_outputs'][i])
                    model_outputs.append(output)
                except:
                    model_outputs.append("")

        # Get sequence lengths
        if 'sequence_lengths' in f:
            seq_lengths = f['sequence_lengths'][:]
        else:
            seq_lengths = np.full(len(trajectories), trajectories.shape[1])

        if max_samples and max_samples < len(trajectories):
            trajectories = trajectories[:max_samples]
            labels = labels[:max_samples]
            model_outputs = model_outputs[:max_samples]
            seq_lengths = seq_lengths[:max_samples]

    return {
        'trajectories': trajectories.astype(np.float32),
        'labels': labels.astype(bool),
        'model_outputs': model_outputs,
        'seq_lengths': seq_lengths
    }


# ============================================================================
# BELIEF PROBE TRAINING
# ============================================================================

def train_belief_probe(activations: np.ndarray, labels: np.ndarray, cv_folds: int = 5) -> Tuple[LogisticRegression, np.ndarray]:
    """
    Train logistic regression probe on activations to predict correctness.

    Returns trained probe and cross-validated probability predictions.
    """
    # Normalize activations
    mean = activations.mean(axis=0, keepdims=True)
    std = activations.std(axis=0, keepdims=True) + 1e-8
    activations_norm = (activations - mean) / std

    # Train probe
    probe = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        C=0.1  # Regularization
    )

    # Get cross-validated predictions (avoid data leakage)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(probe, activations_norm, labels, cv=cv, method='predict_proba')

    # Fit final probe on all data
    probe.fit(activations_norm, labels)

    return probe, cv_probs[:, 1]  # Return P(correct)


def apply_probe_at_positions(probe, activations: np.ndarray, positions: List[int], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply trained probe at specific token positions."""
    # activations shape: (seq_len, n_layers, d_model)
    # We use final layer
    final_layer = activations.shape[1] - 1

    probs = []
    for pos in positions:
        if pos < activations.shape[0]:
            h = activations[pos, final_layer, :]
            h_norm = (h - mean) / (std + 1e-8)
            prob = probe.predict_proba(h_norm.reshape(1, -1))[0, 1]
        else:
            prob = np.nan
        probs.append(prob)

    return np.array(probs)


# ============================================================================
# BELIEF DYNAMICS ANALYSIS
# ============================================================================

def compute_belief_curve(sample_data: Dict, probe, mean: np.ndarray, std: np.ndarray) -> Dict:
    """
    Compute belief evolution curve for a single sample.

    Returns dict with:
    - clause_beliefs: P(correct) at each clause boundary
    - clause_types: type of each clause
    - belief_deltas: change in belief between clauses
    - smoothness: measure of belief curve smoothness
    """
    trajectory = sample_data['trajectory']  # (seq_len, n_layers, d_model)
    model_output = sample_data['model_output']
    seq_len = sample_data['seq_len']

    # Detect clause boundaries
    clauses = detect_clause_boundaries(model_output)

    # Convert to token positions (heuristic)
    clause_token_positions = []
    for start, end, ctype in clauses:
        # Use end position of each clause
        token_pos = min(text_position_to_token_position(model_output, end), seq_len - 1)
        clause_token_positions.append(token_pos)

    # Get belief at each clause boundary
    final_layer = trajectory.shape[1] - 1
    beliefs = []
    for pos in clause_token_positions:
        if pos < trajectory.shape[0]:
            h = trajectory[pos, final_layer, :]
            h_norm = (h - mean) / (std + 1e-8)
            prob = probe.predict_proba(h_norm.reshape(1, -1))[0, 1]
        else:
            prob = np.nan
        beliefs.append(prob)

    beliefs = np.array(beliefs)

    # Compute deltas
    deltas = np.diff(beliefs) if len(beliefs) > 1 else np.array([0])

    # Compute smoothness (inverse of total variation)
    if len(deltas) > 0 and not np.any(np.isnan(deltas)):
        smoothness = 1.0 / (np.sum(np.abs(deltas)) + 1e-8)
    else:
        smoothness = np.nan

    # Compute accumulation rate (average delta)
    accumulation_rate = np.nanmean(deltas) if len(deltas) > 0 else 0

    # ---- Probe-free validation metrics ----
    # Clause-to-clause cosine distance at final layer (no probe needed)
    clause_activations = []
    for pos in clause_token_positions:
        if pos < trajectory.shape[0]:
            clause_activations.append(trajectory[pos, final_layer, :].copy())

    act_cosine_distances = []
    act_norms = []
    if len(clause_activations) >= 2:
        for k in range(len(clause_activations) - 1):
            h_a = clause_activations[k]
            h_b = clause_activations[k + 1]
            norm_a = np.linalg.norm(h_a)
            norm_b = np.linalg.norm(h_b)
            if norm_a > 1e-10 and norm_b > 1e-10:
                cos_sim = np.dot(h_a, h_b) / (norm_a * norm_b)
                act_cosine_distances.append(1.0 - cos_sim)  # distance
            act_norms.append(norm_a)
        act_norms.append(np.linalg.norm(clause_activations[-1]))
    elif len(clause_activations) == 1:
        act_norms.append(np.linalg.norm(clause_activations[0]))

    # Activation-based smoothness: total variation of cosine distances
    # Lower = smoother (representation barely changes between clauses)
    act_total_variation = float(np.sum(act_cosine_distances)) if act_cosine_distances else np.nan
    act_mean_cosine_dist = float(np.mean(act_cosine_distances)) if act_cosine_distances else np.nan
    act_max_cosine_dist = float(np.max(act_cosine_distances)) if act_cosine_distances else np.nan
    act_norm_std = float(np.std(act_norms)) if len(act_norms) >= 2 else np.nan

    return {
        'clause_beliefs': beliefs.tolist(),
        'clause_types': [c[2] for c in clauses],
        'belief_deltas': deltas.tolist(),
        'smoothness': float(smoothness) if not np.isnan(smoothness) else None,
        'accumulation_rate': float(accumulation_rate),
        'n_clauses': len(clauses),
        'final_belief': float(beliefs[-1]) if len(beliefs) > 0 and not np.isnan(beliefs[-1]) else None,
        # Probe-free metrics
        'act_total_variation': act_total_variation if not np.isnan(act_total_variation) else None,
        'act_mean_cosine_dist': act_mean_cosine_dist if not np.isnan(act_mean_cosine_dist) else None,
        'act_max_cosine_dist': act_max_cosine_dist if not np.isnan(act_max_cosine_dist) else None,
        'act_norm_std': act_norm_std if not np.isnan(act_norm_std) else None,
    }


def analyze_model(data: Dict, model_name: str) -> Dict:
    """
    Full belief dynamics analysis for one model.
    """
    trajectories = data['trajectories']
    labels = data['labels']
    model_outputs = data['model_outputs']
    seq_lengths = data['seq_lengths']

    n_samples = len(trajectories)
    n_correct = labels.sum()
    n_incorrect = n_samples - n_correct

    print(f"  Analyzing {model_name}: {n_samples} samples ({n_correct} correct, {n_incorrect} incorrect)")

    # Step 1: Train belief probe on final-position activations
    print("  Training belief probe...")
    final_layer = trajectories.shape[2] - 1

    # Use mean activation across sequence for training (more stable)
    mean_activations = trajectories.mean(axis=1)[:, final_layer, :]  # (n_samples, d_model)

    probe, cv_probs = train_belief_probe(mean_activations, labels)

    # Compute probe AUC
    probe_auc = roc_auc_score(labels, cv_probs)
    print(f"  Belief probe AUC: {probe_auc:.3f}")

    # Get normalization stats for applying probe
    mean = mean_activations.mean(axis=0)
    std = mean_activations.std(axis=0)

    # Step 2: Compute belief curves for each sample
    print("  Computing belief curves...")

    correct_curves = []
    incorrect_curves = []
    all_curves = []

    for i in range(n_samples):
        sample_data = {
            'trajectory': trajectories[i],
            'model_output': model_outputs[i] if i < len(model_outputs) else "",
            'seq_len': int(seq_lengths[i])
        }

        curve = compute_belief_curve(sample_data, probe, mean, std)
        curve['is_correct'] = bool(labels[i])
        curve['sample_idx'] = i

        all_curves.append(curve)
        if labels[i]:
            correct_curves.append(curve)
        else:
            incorrect_curves.append(curve)

    # Step 3: Aggregate statistics
    print("  Computing statistics...")

    def extract_metric(curves, key):
        return [c[key] for c in curves if c[key] is not None]

    # Smoothness comparison
    correct_smoothness = extract_metric(correct_curves, 'smoothness')
    incorrect_smoothness = extract_metric(incorrect_curves, 'smoothness')

    # Accumulation rate comparison
    correct_accum = extract_metric(correct_curves, 'accumulation_rate')
    incorrect_accum = extract_metric(incorrect_curves, 'accumulation_rate')

    # Final belief comparison
    correct_final = extract_metric(correct_curves, 'final_belief')
    incorrect_final = extract_metric(incorrect_curves, 'final_belief')

    # Number of clauses
    correct_nclauses = [c['n_clauses'] for c in correct_curves]
    incorrect_nclauses = [c['n_clauses'] for c in incorrect_curves]

    # Probe-free metrics
    correct_act_tv = extract_metric(correct_curves, 'act_total_variation')
    incorrect_act_tv = extract_metric(incorrect_curves, 'act_total_variation')
    correct_act_mean_cos = extract_metric(correct_curves, 'act_mean_cosine_dist')
    incorrect_act_mean_cos = extract_metric(incorrect_curves, 'act_mean_cosine_dist')
    correct_act_max_cos = extract_metric(correct_curves, 'act_max_cosine_dist')
    incorrect_act_max_cos = extract_metric(incorrect_curves, 'act_max_cosine_dist')
    correct_norm_std = extract_metric(correct_curves, 'act_norm_std')
    incorrect_norm_std = extract_metric(incorrect_curves, 'act_norm_std')

    def compute_stats(correct_vals, incorrect_vals, name):
        if len(correct_vals) < 2 or len(incorrect_vals) < 2:
            return {'d': 0, 'p': 1, 'correct_mean': 0, 'incorrect_mean': 0}

        d = cohens_d(correct_vals, incorrect_vals)
        _, p = stats.ttest_ind(correct_vals, incorrect_vals)

        return {
            'd': float(d),
            'p': float(p),
            'correct_mean': float(np.mean(correct_vals)),
            'incorrect_mean': float(np.mean(incorrect_vals)),
            'correct_std': float(np.std(correct_vals)),
            'incorrect_std': float(np.std(incorrect_vals))
        }

    metrics = {
        'smoothness': compute_stats(correct_smoothness, incorrect_smoothness, 'smoothness'),
        'accumulation_rate': compute_stats(correct_accum, incorrect_accum, 'accumulation_rate'),
        'final_belief': compute_stats(correct_final, incorrect_final, 'final_belief'),
        'n_clauses': compute_stats(correct_nclauses, incorrect_nclauses, 'n_clauses'),
        # Probe-free validation
        'act_total_variation': compute_stats(correct_act_tv, incorrect_act_tv, 'act_total_variation'),
        'act_mean_cosine_dist': compute_stats(correct_act_mean_cos, incorrect_act_mean_cos, 'act_mean_cosine_dist'),
        'act_max_cosine_dist': compute_stats(correct_act_max_cos, incorrect_act_max_cos, 'act_max_cosine_dist'),
        'act_norm_std': compute_stats(correct_norm_std, incorrect_norm_std, 'act_norm_std'),
    }

    # ---- Permutation test for probe-based smoothness ----
    print("  Running permutation test (1000 iterations)...")
    observed_d = metrics['smoothness']['d']
    all_smoothness = correct_smoothness + incorrect_smoothness
    all_smoothness_arr = np.array(all_smoothness)
    n_c = len(correct_smoothness)
    rng = np.random.RandomState(42)
    perm_ds = []
    for _ in range(1000):
        perm_idx = rng.permutation(len(all_smoothness_arr))
        perm_correct = all_smoothness_arr[perm_idx[:n_c]]
        perm_incorrect = all_smoothness_arr[perm_idx[n_c:]]
        perm_d = cohens_d(perm_correct.tolist(), perm_incorrect.tolist())
        perm_ds.append(perm_d)
    perm_ds = np.array(perm_ds)
    perm_p = float(np.mean(np.abs(perm_ds) >= np.abs(observed_d)))
    print(f"    Permutation test: observed d={observed_d:.3f}, null mean={np.mean(perm_ds):.3f} ± {np.std(perm_ds):.3f}, p={perm_p:.4f}")

    # ---- Permutation test for activation cosine distance ----
    observed_act_d = metrics['act_mean_cosine_dist']['d']
    all_act_cos = correct_act_mean_cos + incorrect_act_mean_cos
    all_act_cos_arr = np.array(all_act_cos) if all_act_cos else np.array([])
    n_c_act = len(correct_act_mean_cos)
    perm_act_ds = []
    if len(all_act_cos_arr) > 4:
        for _ in range(1000):
            perm_idx = rng.permutation(len(all_act_cos_arr))
            perm_correct = all_act_cos_arr[perm_idx[:n_c_act]]
            perm_incorrect = all_act_cos_arr[perm_idx[n_c_act:]]
            perm_d = cohens_d(perm_correct.tolist(), perm_incorrect.tolist())
            perm_act_ds.append(perm_d)
        perm_act_ds = np.array(perm_act_ds)
        perm_act_p = float(np.mean(np.abs(perm_act_ds) >= np.abs(observed_act_d)))
        print(f"    Act cosine perm test: observed d={observed_act_d:.3f}, null mean={np.mean(perm_act_ds):.3f} ± {np.std(perm_act_ds):.3f}, p={perm_act_p:.4f}")
    else:
        perm_act_p = 1.0
        perm_act_ds = np.array([0.0])

    results = {
        'model': model_name,
        'n_samples': n_samples,
        'n_correct': int(n_correct),
        'n_incorrect': int(n_incorrect),
        'probe_auc': float(probe_auc),
        'metrics': metrics,
        'permutation_tests': {
            'probe_smoothness': {
                'observed_d': float(observed_d),
                'null_mean': float(np.mean(perm_ds)),
                'null_std': float(np.std(perm_ds)),
                'perm_p': perm_p
            },
            'act_cosine_dist': {
                'observed_d': float(observed_act_d),
                'null_mean': float(np.mean(perm_act_ds)),
                'null_std': float(np.std(perm_act_ds)),
                'perm_p': perm_act_p
            }
        },
        'all_curves': all_curves  # For detailed analysis
    }

    # Print summary
    print(f"  Results:")
    print(f"    Probe smoothness:     d={metrics['smoothness']['d']:.3f}, p={metrics['smoothness']['p']:.4f}")
    print(f"    Act cosine dist:      d={metrics['act_mean_cosine_dist']['d']:.3f}, p={metrics['act_mean_cosine_dist']['p']:.4f}")
    print(f"    Act total variation:  d={metrics['act_total_variation']['d']:.3f}, p={metrics['act_total_variation']['p']:.4f}")
    print(f"    Act max jump:         d={metrics['act_max_cosine_dist']['d']:.3f}, p={metrics['act_max_cosine_dist']['p']:.4f}")
    print(f"    Act norm stability:   d={metrics['act_norm_std']['d']:.3f}, p={metrics['act_norm_std']['p']:.4f}")
    print(f"    Accumulation:         d={metrics['accumulation_rate']['d']:.3f}, p={metrics['accumulation_rate']['p']:.4f}")
    print(f"    Final belief:         d={metrics['final_belief']['d']:.3f}, p={metrics['final_belief']['p']:.4f}")

    return results


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
# CROSS-MODEL TRANSFER
# ============================================================================

def cross_model_transfer(data_a: Dict, data_b: Dict, model_a: str, model_b: str) -> Dict:
    """
    Train probe on model A, apply to model B.
    Tests if belief state transfers across models.
    """
    print(f"  Transfer: {model_a} -> {model_b}")

    # Train probe on model A
    traj_a = data_a['trajectories']
    labels_a = data_a['labels']
    final_layer = traj_a.shape[2] - 1

    mean_act_a = traj_a.mean(axis=1)[:, final_layer, :]

    # Normalize
    mean_a = mean_act_a.mean(axis=0)
    std_a = mean_act_a.std(axis=0) + 1e-8
    act_a_norm = (mean_act_a - mean_a) / std_a

    probe = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
    probe.fit(act_a_norm, labels_a)

    # Apply to model B
    traj_b = data_b['trajectories']
    labels_b = data_b['labels']

    mean_act_b = traj_b.mean(axis=1)[:, final_layer, :]

    # Use model A's normalization stats
    act_b_norm = (mean_act_b - mean_a) / std_a

    probs_b = probe.predict_proba(act_b_norm)[:, 1]

    # Compute AUC
    try:
        auc = roc_auc_score(labels_b, probs_b)
    except:
        auc = 0.5

    print(f"    AUC: {auc:.3f}")

    return {
        'train_model': model_a,
        'test_model': model_b,
        'auc': float(auc),
        'n_train': len(labels_a),
        'n_test': len(labels_b)
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Belief State Tracking Analysis')
    parser.add_argument('--data-dir', required=True, help='Directory with trajectory HDF5 files')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--models', default='olmo3_rl_zero,olmo3_think,olmo3_sft',
                        help='Comma-separated model names')
    parser.add_argument('--task', default='gsm8k', help='Task to analyze')
    parser.add_argument('--max-samples', type=int, default=200, help='Max samples per model')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models = [m.strip() for m in args.models.split(',')]

    print("=" * 60)
    print("Belief State Tracking Analysis")
    print("=" * 60)
    print(f"Data dir: {args.data_dir}")
    print(f"Models: {models}")
    print(f"Task: {args.task}")
    print(f"Max samples: {args.max_samples}")
    print()

    # Load data for each model
    all_data = {}
    for model in models:
        # Try different filename patterns
        patterns = [
            f"{args.data_dir}/{model}/{args.task}_trajectories.h5",
            f"{args.data_dir}/{model}/{args.task}_trajectories_optimized.h5",
            f"{args.data_dir}/{model}/{args.task}_trajectories_vllm_optimized.h5",
        ]

        filepath = None
        for p in patterns:
            if os.path.exists(p):
                filepath = p
                break

        if filepath is None:
            print(f"WARNING: No data file found for {model}/{args.task}")
            continue

        print(f"Loading {model} from {filepath}")
        all_data[model] = load_data(filepath, args.max_samples)

    if not all_data:
        print("ERROR: No data loaded!")
        return

    # Phase 1: Analyze each model
    print()
    print("=" * 60)
    print("Phase 1: Per-Model Belief Dynamics")
    print("=" * 60)

    all_results = {}
    for model in all_data:
        print()
        results = analyze_model(all_data[model], model)
        all_results[model] = results

    # Phase 2: Cross-model transfer
    print()
    print("=" * 60)
    print("Phase 2: Cross-Model Transfer")
    print("=" * 60)

    transfer_results = []
    model_list = list(all_data.keys())

    for model_a in model_list:
        for model_b in model_list:
            result = cross_model_transfer(all_data[model_a], all_data[model_b], model_a, model_b)
            transfer_results.append(result)

    # Build transfer matrix
    transfer_matrix = {}
    for r in transfer_results:
        key = f"{r['train_model']}_to_{r['test_model']}"
        transfer_matrix[key] = r['auc']

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nPer-Model Results:")
    for model, results in all_results.items():
        print(f"\n{model}:")
        print(f"  Probe AUC: {results['probe_auc']:.3f}")
        print(f"  Smoothness effect: d={results['metrics']['smoothness']['d']:.3f}")
        print(f"  Accumulation effect: d={results['metrics']['accumulation_rate']['d']:.3f}")

    print("\nTransfer Matrix (AUC):")
    for model_a in model_list:
        row = []
        for model_b in model_list:
            key = f"{model_a}_to_{model_b}"
            row.append(f"{transfer_matrix.get(key, 0):.3f}")
        print(f"  {model_a}: {' | '.join(row)}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'data_dir': args.data_dir,
            'models': models,
            'task': args.task,
            'max_samples': args.max_samples
        },
        'per_model': {m: {k: v for k, v in r.items() if k not in ('all_curves',)}
                      for m, r in all_results.items()},
        'transfer_matrix': transfer_matrix,
        'transfer_details': transfer_results
    }

    output_file = os.path.join(args.output_dir, f'belief_dynamics_{args.task}.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Save detailed curves for visualization
    curves_file = os.path.join(args.output_dir, f'belief_curves_{args.task}.json')
    curves_output = {m: r['all_curves'] for m, r in all_results.items()}
    with open(curves_file, 'w') as f:
        json.dump(curves_output, f, indent=2)

    print(f"Curves saved to: {curves_file}")


if __name__ == '__main__':
    main()
