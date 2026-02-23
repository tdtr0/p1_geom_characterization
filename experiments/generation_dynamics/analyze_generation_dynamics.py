#!/usr/bin/env python3
"""
Generation Dynamics Analysis (Phases A, B, C)

Phase A: Descriptive — gen lengths, entropy profiles, top-k overlap
Phase B: Correctness prediction — entropy features, hidden state probes
Phase C: Base vs RL-Zero divergence — step-by-step comparison

Usage:
    python experiments/generation_dynamics/analyze_generation_dynamics.py \
        --data-dir data/generation_trajectories \
        --output-dir experiments/generation_dynamics/results
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import warnings
warnings.filterwarnings("ignore")

# Optional imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not available, skipping Phase B")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Data Loading
# =============================================================================

def load_generation_data(data_dir, model, task):
    """Load generation trajectory data from HDF5."""
    path = Path(data_dir) / model / f"{task}_generation.h5"
    if not path.exists():
        return None

    samples = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("sample_"):
                continue
            g = f[key]
            sample = {
                "key": key,
                "is_correct": bool(g.attrs.get("is_correct", False)),
                "gen_len": int(g.attrs.get("gen_len", 0)),
                "prompt_len": int(g.attrs.get("prompt_len", 0)),
                "was_truncated": bool(g.attrs.get("was_truncated", False)),
            }

            # Load entropy (small, always load)
            if "entropy" in g:
                sample["entropy"] = g["entropy"][:]

            # Load top-k probs (moderate size)
            if "top_k_probs" in g:
                sample["top_k_probs"] = g["top_k_probs"][:]
            if "top_k_tokens" in g:
                sample["top_k_tokens"] = g["top_k_tokens"][:]

            samples.append(sample)

    return samples


def load_hidden_states(data_dir, model, task, sample_keys=None, layer_idx=-1):
    """Load hidden states for specific samples (memory-efficient)."""
    path = Path(data_dir) / model / f"{task}_generation.h5"
    states = {}
    with h5py.File(path, "r") as f:
        keys = sample_keys or [k for k in sorted(f.keys()) if k.startswith("sample_")]
        for key in keys:
            g = f[key]
            if "hidden_states" in g:
                # shape: (gen_len, n_layers, d_model)
                hs = g["hidden_states"]
                # Load just the requested layer
                states[key] = hs[:, layer_idx, :]
    return states


# =============================================================================
# Phase A: Descriptive Analysis
# =============================================================================

def phase_a_descriptive(data_dir, models, tasks):
    """Phase A: Descriptive statistics."""
    print("\n" + "=" * 60)
    print("  PHASE A: DESCRIPTIVE ANALYSIS")
    print("=" * 60)

    results = {}

    for task in tasks:
        print(f"\n--- {task} ---")
        for model in models:
            samples = load_generation_data(data_dir, model, task)
            if not samples:
                print(f"  {model}: No data")
                continue

            correct = [s for s in samples if s["is_correct"]]
            incorrect = [s for s in samples if not s["is_correct"]]

            # Gen length stats
            gen_lens_c = [s["gen_len"] for s in correct] if correct else [0]
            gen_lens_i = [s["gen_len"] for s in incorrect] if incorrect else [0]

            # Entropy stats
            ent_means_c, ent_means_i = [], []
            ent_vars_c, ent_vars_i = [], []
            ent_slopes_c, ent_slopes_i = [], []

            for s in correct:
                if "entropy" in s and len(s["entropy"]) > 1:
                    e = s["entropy"]
                    # Filter out zero-entropy steps (padding)
                    e_valid = e[e > 0]
                    if len(e_valid) > 1:
                        ent_means_c.append(np.mean(e_valid))
                        ent_vars_c.append(np.var(e_valid))
                        # Slope: linear fit
                        x = np.arange(len(e_valid))
                        slope = np.polyfit(x, e_valid, 1)[0]
                        ent_slopes_c.append(slope)

            for s in incorrect:
                if "entropy" in s and len(s["entropy"]) > 1:
                    e = s["entropy"]
                    e_valid = e[e > 0]
                    if len(e_valid) > 1:
                        ent_means_i.append(np.mean(e_valid))
                        ent_vars_i.append(np.var(e_valid))
                        x = np.arange(len(e_valid))
                        slope = np.polyfit(x, e_valid, 1)[0]
                        ent_slopes_i.append(slope)

            # Cohen's d
            def cohens_d(a, b):
                if len(a) < 2 or len(b) < 2:
                    return 0.0
                na, nb = np.array(a), np.array(b)
                pooled = np.sqrt((np.var(na) * (len(na) - 1) + np.var(nb) * (len(nb) - 1)) / (len(na) + len(nb) - 2))
                if pooled == 0:
                    return 0.0
                return (np.mean(na) - np.mean(nb)) / pooled

            d_gen_len = cohens_d(gen_lens_c, gen_lens_i)
            d_ent_mean = cohens_d(ent_means_c, ent_means_i) if ent_means_c and ent_means_i else 0
            d_ent_var = cohens_d(ent_vars_c, ent_vars_i) if ent_vars_c and ent_vars_i else 0
            d_ent_slope = cohens_d(ent_slopes_c, ent_slopes_i) if ent_slopes_c and ent_slopes_i else 0

            # p-values
            p_gen_len = stats.mannwhitneyu(gen_lens_c, gen_lens_i, alternative="two-sided").pvalue if HAS_SCIPY and len(gen_lens_c) > 1 and len(gen_lens_i) > 1 else float("nan")
            p_ent_mean = stats.mannwhitneyu(ent_means_c, ent_means_i, alternative="two-sided").pvalue if HAS_SCIPY and len(ent_means_c) > 1 and len(ent_means_i) > 1 else float("nan")

            truncated = sum(1 for s in samples if s["was_truncated"])

            print(f"\n  {model}:")
            print(f"    Samples: {len(samples)} ({len(correct)} correct, {len(incorrect)} incorrect, {truncated} truncated)")
            print(f"    Gen length:  correct={np.mean(gen_lens_c):.0f} vs incorrect={np.mean(gen_lens_i):.0f}  (d={d_gen_len:.3f}, p={p_gen_len:.4f})")
            print(f"    Entropy mean: correct={np.mean(ent_means_c):.3f} vs incorrect={np.mean(ent_means_i):.3f}  (d={d_ent_mean:.3f})" if ent_means_c and ent_means_i else "    Entropy: insufficient data")
            print(f"    Entropy var:  correct={np.mean(ent_vars_c):.3f} vs incorrect={np.mean(ent_vars_i):.3f}  (d={d_ent_var:.3f})" if ent_vars_c and ent_vars_i else "")
            print(f"    Entropy slope: correct={np.mean(ent_slopes_c):.5f} vs incorrect={np.mean(ent_slopes_i):.5f}  (d={d_ent_slope:.3f})" if ent_slopes_c and ent_slopes_i else "")

            results[f"{model}/{task}"] = {
                "n_samples": len(samples),
                "n_correct": len(correct),
                "n_incorrect": len(incorrect),
                "n_truncated": truncated,
                "gen_len_correct_mean": float(np.mean(gen_lens_c)),
                "gen_len_incorrect_mean": float(np.mean(gen_lens_i)),
                "gen_len_d": float(d_gen_len),
                "gen_len_p": float(p_gen_len),
                "entropy_mean_d": float(d_ent_mean),
                "entropy_var_d": float(d_ent_var),
                "entropy_slope_d": float(d_ent_slope),
            }

    return results


# =============================================================================
# Phase A2: Entropy Profile Over Generation Steps
# =============================================================================

def phase_a2_entropy_profiles(data_dir, models, tasks, max_steps=200):
    """Compute average entropy profile over generation steps."""
    print("\n" + "=" * 60)
    print("  PHASE A2: ENTROPY PROFILES OVER GENERATION STEPS")
    print("=" * 60)

    results = {}

    for task in tasks:
        print(f"\n--- {task} ---")
        for model in models:
            samples = load_generation_data(data_dir, model, task)
            if not samples:
                continue

            correct = [s for s in samples if s["is_correct"] and "entropy" in s]
            incorrect = [s for s in samples if not s["is_correct"] and "entropy" in s]

            # Bin entropy by generation step
            for label, group in [("correct", correct), ("incorrect", incorrect)]:
                if not group:
                    continue
                # Collect entropy at each step position
                step_entropies = defaultdict(list)
                for s in group:
                    e = s["entropy"]
                    for t in range(min(len(e), max_steps)):
                        if e[t] > 0:  # skip padding
                            step_entropies[t].append(float(e[t]))

                # Compute mean at each step
                steps = sorted(step_entropies.keys())
                means = [np.mean(step_entropies[t]) for t in steps]

                # Report key positions
                if means:
                    print(f"  {model}/{label} (n={len(group)}): step0={means[0]:.2f}, step10={means[min(10,len(means)-1)]:.2f}, step50={means[min(50,len(means)-1)]:.2f}, final={means[-1]:.2f}")

                    results[f"{model}/{task}/{label}"] = {
                        "n": len(group),
                        "entropy_step0": float(means[0]),
                        "entropy_step10": float(means[min(10, len(means) - 1)]),
                        "entropy_step50": float(means[min(50, len(means) - 1)]),
                        "entropy_final": float(means[-1]),
                        "entropy_mean_all": float(np.mean(means)),
                    }

    return results


# =============================================================================
# Phase B: Correctness Prediction
# =============================================================================

def phase_b_correctness_prediction(data_dir, models, tasks):
    """Phase B: Predict correctness from generation dynamics."""
    if not HAS_SKLEARN:
        print("\nSkipping Phase B (sklearn not available)")
        return {}

    print("\n" + "=" * 60)
    print("  PHASE B: CORRECTNESS PREDICTION")
    print("=" * 60)

    results = {}

    for task in tasks:
        print(f"\n--- {task} ---")
        for model in models:
            samples = load_generation_data(data_dir, model, task)
            if not samples:
                continue

            # Build feature matrix from entropy
            features = []
            labels = []
            for s in samples:
                if "entropy" not in s or s["gen_len"] < 2:
                    continue
                e = s["entropy"]
                e_valid = e[e > 0]
                if len(e_valid) < 2:
                    continue

                feat = [
                    np.mean(e_valid),           # mean entropy
                    np.var(e_valid),             # entropy variance
                    np.min(e_valid),             # min entropy (most confident)
                    np.max(e_valid),             # max entropy (most uncertain)
                    e_valid[0],                  # first step entropy
                    e_valid[-1],                 # last step entropy
                    np.polyfit(np.arange(len(e_valid)), e_valid, 1)[0],  # slope
                    float(s["gen_len"]),          # generation length
                    float(len(e_valid)),          # non-padding steps
                ]
                features.append(feat)
                labels.append(int(s["is_correct"]))

            if len(features) < 20 or sum(labels) < 5:
                print(f"  {model}: insufficient data ({len(features)} samples, {sum(labels)} correct)")
                continue

            X = np.array(features)
            y = np.array(labels)

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Cross-validated AUC
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            clf = LogisticRegression(max_iter=1000, random_state=42)

            try:
                aucs = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
            except Exception as e:
                print(f"  {model}: AUC failed ({e})")
                continue

            # Feature importance (fit on all data)
            clf.fit(X_scaled, y)
            feat_names = ["ent_mean", "ent_var", "ent_min", "ent_max", "ent_first", "ent_last", "ent_slope", "gen_len", "n_steps"]
            importances = dict(zip(feat_names, clf.coef_[0]))

            # Sort by absolute importance
            sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)

            print(f"  {model}: AUC = {mean_auc:.3f} +/- {std_auc:.3f} (n={len(features)}, {sum(labels)} correct)")
            print(f"    Top features: {sorted_imp[0][0]}={sorted_imp[0][1]:.3f}, {sorted_imp[1][0]}={sorted_imp[1][1]:.3f}, {sorted_imp[2][0]}={sorted_imp[2][1]:.3f}")

            results[f"{model}/{task}"] = {
                "auc_mean": float(mean_auc),
                "auc_std": float(std_auc),
                "n_samples": len(features),
                "n_correct": sum(labels),
                "feature_importance": {k: float(v) for k, v in sorted_imp},
            }

    # Phase B2: Hidden state probes at final layer (streaming from HDF5)
    print("\n  --- Hidden State Probes (Final Layer) ---")
    for task in tasks:
        for model in models:
            path = Path(data_dir) / model / f"{task}_generation.h5"
            if not path.exists():
                continue

            X, y = [], []
            with h5py.File(path, "r") as f:
                keys = sorted([k for k in f.keys() if k.startswith("sample_")])
                for key in keys:
                    g = f[key]
                    gen_len = int(g.attrs.get("gen_len", 0))
                    if gen_len < 2 or "hidden_states" not in g:
                        continue
                    label = int(g.attrs.get("is_correct", False))
                    # Stream: read only last layer, mean-pool over steps
                    hs = g["hidden_states"][:, -1, :].astype(np.float32)  # (gen_len, d_model)
                    valid = np.any(hs != 0, axis=1)
                    hs_valid = hs[valid]
                    if len(hs_valid) > 0:
                        X.append(np.mean(hs_valid, axis=0))
                        y.append(label)

            if len(X) < 20 or sum(y) < 5:
                print(f"  {model}/{task}: insufficient data ({len(X)} samples, {sum(y)} correct)")
                continue

            X = np.array(X, dtype=np.float32)
            y = np.array(y)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)

            try:
                aucs = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                print(f"  {model}/{task} hidden probe: AUC = {mean_auc:.3f} +/- {std_auc:.3f}")
                results[f"{model}/{task}/hidden_probe"] = {
                    "auc_mean": float(mean_auc),
                    "auc_std": float(std_auc),
                    "n_samples": len(X),
                    "n_correct": int(sum(y)),
                }
            except Exception as e:
                print(f"  {model}/{task} hidden probe failed: {e}")

    return results


# =============================================================================
# Phase C: Base vs RL-Zero Divergence
# =============================================================================

def phase_c_divergence(data_dir, tasks):
    """Phase C: Compare base vs rl_zero during generation."""
    print("\n" + "=" * 60)
    print("  PHASE C: BASE vs RL-ZERO DIVERGENCE")
    print("=" * 60)

    results = {}

    for task in tasks:
        print(f"\n--- {task} ---")

        base_samples = load_generation_data(data_dir, "olmo3_base", task)
        rlz_samples = load_generation_data(data_dir, "olmo3_rl_zero", task)

        if not base_samples or not rlz_samples:
            print("  Missing data for one model")
            continue

        # Match by prompt hash (or index since same seed)
        base_by_key = {}
        for s in base_samples:
            base_by_key[s["key"]] = s

        rlz_by_key = {}
        for s in rlz_samples:
            rlz_by_key[s["key"]] = s

        # Find matching keys
        common_keys = sorted(set(base_by_key.keys()) & set(rlz_by_key.keys()))
        print(f"  Matched samples: {len(common_keys)}")

        if len(common_keys) < 10:
            continue

        # C1: Entropy divergence
        both_correct, rlz_wins, base_wins, both_wrong = [], [], [], []
        ent_div_by_outcome = defaultdict(list)

        for key in common_keys:
            b = base_by_key[key]
            r = rlz_by_key[key]

            bc, rc = b["is_correct"], r["is_correct"]

            if bc and rc:
                both_correct.append(key)
                outcome = "both_correct"
            elif rc and not bc:
                rlz_wins.append(key)
                outcome = "rlz_wins"
            elif bc and not rc:
                base_wins.append(key)
                outcome = "base_wins"
            else:
                both_wrong.append(key)
                outcome = "both_wrong"

            # Entropy divergence
            if "entropy" in b and "entropy" in r:
                be = b["entropy"]
                re = r["entropy"]
                min_len = min(len(be[be > 0]), len(re[re > 0]))
                if min_len > 0:
                    be_v = be[be > 0][:min_len]
                    re_v = re[re > 0][:min_len]
                    ent_diff = float(np.mean(np.abs(be_v - re_v)))
                    ent_div_by_outcome[outcome].append(ent_diff)

        print(f"  Outcomes: both_correct={len(both_correct)}, rlz_wins={len(rlz_wins)}, base_wins={len(base_wins)}, both_wrong={len(both_wrong)}")

        for outcome, diffs in sorted(ent_div_by_outcome.items()):
            if diffs:
                print(f"    Entropy |base-rl_zero| ({outcome}): {np.mean(diffs):.4f} +/- {np.std(diffs):.4f} (n={len(diffs)})")

        # C2: Generation length comparison
        gen_len_diffs = []
        for key in common_keys:
            b = base_by_key[key]
            r = rlz_by_key[key]
            gen_len_diffs.append(r["gen_len"] - b["gen_len"])

        print(f"  Gen length diff (rl_zero - base): {np.mean(gen_len_diffs):.1f} +/- {np.std(gen_len_diffs):.1f}")

        # C3: Top-k token overlap at first step
        top_overlaps = []
        for key in common_keys:
            b = base_by_key[key]
            r = rlz_by_key[key]
            if "top_k_tokens" in b and "top_k_tokens" in r:
                bt = set(b["top_k_tokens"][0].tolist())
                rt = set(r["top_k_tokens"][0].tolist())
                overlap = len(bt & rt) / max(len(bt | rt), 1)
                top_overlaps.append(overlap)

        if top_overlaps:
            print(f"  Top-100 token overlap (step 0): {np.mean(top_overlaps):.3f} +/- {np.std(top_overlaps):.3f}")

        # C4: Hidden state divergence at final layer (streaming)
        print(f"  Computing hidden state divergence (streaming)...")
        base_path = Path(data_dir) / "olmo3_base" / f"{task}_generation.h5"
        rlz_path = Path(data_dir) / "olmo3_rl_zero" / f"{task}_generation.h5"

        cos_sims_by_outcome = defaultdict(list)
        with h5py.File(base_path, "r") as fb, h5py.File(rlz_path, "r") as fr:
            for key in common_keys:
                if key not in fb or key not in fr:
                    continue
                gb = fb[key]
                gr = fr[key]
                if "hidden_states" not in gb or "hidden_states" not in gr:
                    continue

                bc = base_by_key[key]["is_correct"]
                rc = rlz_by_key[key]["is_correct"]

                if bc and rc:
                    outcome = "both_correct"
                elif rc and not bc:
                    outcome = "rlz_wins"
                elif bc and not rc:
                    outcome = "base_wins"
                else:
                    outcome = "both_wrong"

                # Stream: read only last layer for each sample
                bh = gb["hidden_states"][:, -1, :].astype(np.float32)
                rh = gr["hidden_states"][:, -1, :].astype(np.float32)

                min_len = min(len(bh), len(rh))
                if min_len == 0:
                    continue

                bh_v = bh[:min_len]
                rh_v = rh[:min_len]

                # Filter zero rows
                valid = np.any(bh_v != 0, axis=1) & np.any(rh_v != 0, axis=1)
                if valid.sum() == 0:
                    continue

                bh_v = bh_v[valid]
                rh_v = rh_v[valid]

                # Cosine similarity per step
                dot = np.sum(bh_v * rh_v, axis=1)
                norm_b = np.linalg.norm(bh_v, axis=1)
                norm_r = np.linalg.norm(rh_v, axis=1)
                cos = dot / (norm_b * norm_r + 1e-8)

                cos_sims_by_outcome[outcome].append({
                    "mean_cos": float(np.mean(cos)),
                    "min_cos": float(np.min(cos)),
                    "n_steps": int(len(cos)),
                    "cos_step0": float(cos[0]),
                    "cos_final": float(cos[-1]),
                })

        print(f"\n  Hidden state cos_sim (base vs rl_zero) at final layer:")
        for outcome in ["both_correct", "rlz_wins", "base_wins", "both_wrong"]:
            entries = cos_sims_by_outcome.get(outcome, [])
            if entries:
                mean_cos = np.mean([e["mean_cos"] for e in entries])
                min_cos = np.mean([e["min_cos"] for e in entries])
                cos0 = np.mean([e["cos_step0"] for e in entries])
                cosf = np.mean([e["cos_final"] for e in entries])
                print(f"    {outcome:15s} (n={len(entries):3d}): mean={mean_cos:.5f}, min={min_cos:.5f}, step0={cos0:.5f}, final={cosf:.5f}")

        results[task] = {
            "n_matched": len(common_keys),
            "both_correct": len(both_correct),
            "rlz_wins": len(rlz_wins),
            "base_wins": len(base_wins),
            "both_wrong": len(both_wrong),
            "entropy_div_by_outcome": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in ent_div_by_outcome.items()},
            "gen_len_diff_mean": float(np.mean(gen_len_diffs)),
            "top100_overlap_step0": float(np.mean(top_overlaps)) if top_overlaps else None,
            "hidden_cos_by_outcome": {k: {"mean_cos": float(np.mean([e["mean_cos"] for e in v])), "n": len(v)} for k, v in cos_sims_by_outcome.items()},
        }

    return results


# =============================================================================
# Phase D: Velocity-Space PCA on Generation Increments
# =============================================================================

def phase_d_velocity_pca(data_dir, models, tasks, max_steps=300, layer_idx=-1):
    """Phase D: PCA on Δx_t = x_{t+1} - x_t to find slow manifold / spectral gap.

    For each model/task:
    1. Stream hidden states at final layer
    2. Compute velocity increments Δx_t = x_{t+1} - x_t
    3. Collect across traces, split by correct/incorrect
    4. PCA on pooled velocities → eigenvalue spectrum
    5. Check for spectral gap (reasoning progress vs token noise)
    6. Compare PC alignment between correct/incorrect and between models
    """
    print("\n" + "=" * 60)
    print("  PHASE D: VELOCITY-SPACE PCA ON GENERATION INCREMENTS")
    print("=" * 60)

    results = {}

    for task in tasks:
        print(f"\n--- {task} ---")
        task_results = {}

        # Collect velocities per model
        model_velocities = {}  # model -> {"correct": ndarray, "incorrect": ndarray, "all": ndarray}

        for model in models:
            path = Path(data_dir) / model / f"{task}_generation.h5"
            if not path.exists():
                print(f"  {model}: No data")
                continue

            print(f"  {model}: Streaming velocity increments...")

            vels_correct = []
            vels_incorrect = []
            n_samples_used = 0

            with h5py.File(path, "r") as f:
                keys = sorted([k for k in f.keys() if k.startswith("sample_")])
                for key in keys:
                    g = f[key]
                    if "hidden_states" not in g:
                        continue
                    gen_len = int(g.attrs.get("gen_len", 0))
                    if gen_len < 3:
                        continue

                    is_correct = bool(g.attrs.get("is_correct", False))

                    # Stream final layer only: (gen_len, d_model)
                    hs = g["hidden_states"][:min(gen_len, max_steps), layer_idx, :].astype(np.float32)

                    # Filter zero rows
                    valid = np.any(hs != 0, axis=1)
                    hs = hs[valid]
                    if len(hs) < 3:
                        continue

                    # Compute velocity increments: Δx_t = x_{t+1} - x_t
                    deltas = np.diff(hs, axis=0)  # (T-1, d_model)

                    # Filter out near-zero deltas (padding artifacts)
                    norms = np.linalg.norm(deltas, axis=1)
                    valid_deltas = deltas[norms > 1e-6]

                    if len(valid_deltas) == 0:
                        continue

                    if is_correct:
                        vels_correct.append(valid_deltas)
                    else:
                        vels_incorrect.append(valid_deltas)

                    n_samples_used += 1

            if not vels_correct and not vels_incorrect:
                print(f"    No valid velocity data")
                continue

            # Stack into matrices
            V_correct = np.vstack(vels_correct) if vels_correct else np.zeros((0, hs.shape[1]))
            V_incorrect = np.vstack(vels_incorrect) if vels_incorrect else np.zeros((0, hs.shape[1]))
            V_all = np.vstack([V_correct, V_incorrect]) if (len(V_correct) + len(V_incorrect)) > 0 else np.zeros((0, hs.shape[1]))

            print(f"    Samples: {n_samples_used}, Velocity vectors: {len(V_all)} (correct={len(V_correct)}, incorrect={len(V_incorrect)})")

            model_velocities[model] = {
                "correct": V_correct,
                "incorrect": V_incorrect,
                "all": V_all,
            }

            # --- PCA on all velocities ---
            n_components = min(50, len(V_all) - 1, V_all.shape[1])
            if n_components < 2:
                print(f"    Insufficient data for PCA")
                continue

            # Center the data
            V_centered = V_all - np.mean(V_all, axis=0)

            # SVD-based PCA (memory efficient)
            print(f"    Running PCA (n={len(V_all)}, d={V_all.shape[1]}, k={n_components})...")
            U, S, Vt = np.linalg.svd(V_centered, full_matrices=False)
            eigenvalues = (S ** 2) / (len(V_all) - 1)
            eigenvalues = eigenvalues[:n_components]
            total_var = np.sum(eigenvalues)
            var_explained = np.cumsum(eigenvalues) / total_var

            # Spectral gap analysis: λ_k / λ_{k+1}
            ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-10)

            # Find largest gap
            gap_idx = int(np.argmax(ratios))
            gap_ratio = float(ratios[gap_idx])

            print(f"    Eigenvalue spectrum (top 10): {', '.join([f'{e:.2f}' for e in eigenvalues[:10]])}")
            print(f"    Variance explained: PC1={var_explained[0]:.3f}, PC5={var_explained[min(4,len(var_explained)-1)]:.3f}, PC10={var_explained[min(9,len(var_explained)-1)]:.3f}, PC20={var_explained[min(19,len(var_explained)-1)]:.3f}")
            print(f"    Spectral gap: largest at PC{gap_idx+1}/{gap_idx+2} (ratio={gap_ratio:.2f})")

            # --- Correct vs Incorrect on the PCs ---
            PCs = Vt[:n_components]  # (n_components, d_model)

            proj_correct, proj_incorrect = None, None
            if len(V_correct) > 5:
                V_c_centered = V_correct - np.mean(V_all, axis=0)
                proj_correct = V_c_centered @ PCs.T  # (n_correct_vels, n_components)

            if len(V_incorrect) > 5:
                V_i_centered = V_incorrect - np.mean(V_all, axis=0)
                proj_incorrect = V_i_centered @ PCs.T  # (n_incorrect_vels, n_components)

            if proj_correct is not None and proj_incorrect is not None:
                print(f"\n    Correct vs Incorrect projections onto top PCs:")
                print(f"    {'PC':>4s}  {'Correct var':>12s}  {'Incorrect var':>14s}  {'Ratio':>8s}  {'Mean diff':>10s}")
                pc_comparison = []
                for pc_i in range(min(10, n_components)):
                    var_c = float(np.var(proj_correct[:, pc_i]))
                    var_i = float(np.var(proj_incorrect[:, pc_i]))
                    ratio = var_c / (var_i + 1e-10)
                    mean_diff = float(np.mean(proj_correct[:, pc_i]) - np.mean(proj_incorrect[:, pc_i]))
                    print(f"    PC{pc_i+1:2d}  {var_c:12.4f}  {var_i:14.4f}  {ratio:8.3f}  {mean_diff:10.4f}")
                    pc_comparison.append({
                        "pc": pc_i + 1,
                        "var_correct": var_c,
                        "var_incorrect": var_i,
                        "var_ratio": float(ratio),
                        "mean_diff": mean_diff,
                    })

                # Statistical test: are projections separable?
                if HAS_SKLEARN and len(proj_correct) > 10 and len(proj_incorrect) > 10:
                    # Use top-k PCs as features for correctness classification
                    for k in [3, 5, 10]:
                        if k > n_components:
                            break
                        X_pc = np.vstack([proj_correct[:, :k], proj_incorrect[:, :k]])
                        y_pc = np.array([1] * len(proj_correct) + [0] * len(proj_incorrect))

                        # Subsample if too many velocity vectors (keep balanced)
                        max_per_class = 5000
                        if len(proj_correct) > max_per_class or len(proj_incorrect) > max_per_class:
                            idx_c = np.random.choice(len(proj_correct), min(max_per_class, len(proj_correct)), replace=False)
                            idx_i = np.random.choice(len(proj_incorrect), min(max_per_class, len(proj_incorrect)), replace=False)
                            X_pc = np.vstack([proj_correct[idx_c, :k], proj_incorrect[idx_i, :k]])
                            y_pc = np.array([1] * len(idx_c) + [0] * len(idx_i))

                        try:
                            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                            clf = LogisticRegression(max_iter=1000, random_state=42)
                            aucs = cross_val_score(clf, X_pc, y_pc, cv=cv, scoring="roc_auc")
                            print(f"    Velocity PC{k} → correctness AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
                        except Exception as e:
                            print(f"    Velocity PC{k} classification failed: {e}")

            # --- Separate PCA for correct vs incorrect ---
            if len(V_correct) > 50 and len(V_incorrect) > 50:
                print(f"\n    Separate PCA: correct vs incorrect")
                n_sub = min(20, len(V_correct) - 1, len(V_incorrect) - 1)

                V_c_c = V_correct - np.mean(V_correct, axis=0)
                _, S_c, Vt_c = np.linalg.svd(V_c_c, full_matrices=False)
                eig_c = (S_c ** 2) / (len(V_correct) - 1)
                eig_c = eig_c[:n_sub]
                var_c = np.cumsum(eig_c) / np.sum(eig_c)

                V_i_c = V_incorrect - np.mean(V_incorrect, axis=0)
                _, S_i, Vt_i = np.linalg.svd(V_i_c, full_matrices=False)
                eig_i = (S_i ** 2) / (len(V_incorrect) - 1)
                eig_i = eig_i[:n_sub]
                var_i = np.cumsum(eig_i) / np.sum(eig_i)

                print(f"    Correct  spectrum (top 5): {', '.join([f'{e:.2f}' for e in eig_c[:5]])}")
                print(f"    Incorrect spectrum (top 5): {', '.join([f'{e:.2f}' for e in eig_i[:5]])}")
                print(f"    Correct  var explained: PC1={var_c[0]:.3f}, PC5={var_c[min(4,len(var_c)-1)]:.3f}, PC10={var_c[min(9,len(var_c)-1)]:.3f}")
                print(f"    Incorrect var explained: PC1={var_i[0]:.3f}, PC5={var_i[min(4,len(var_i)-1)]:.3f}, PC10={var_i[min(9,len(var_i)-1)]:.3f}")

                # PC alignment: cosine sim between top PCs of correct vs incorrect
                PCs_c = Vt_c[:n_sub]
                PCs_i = Vt_i[:n_sub]
                alignment = np.abs(PCs_c @ PCs_i.T)  # (n_sub, n_sub) cosine similarities

                # Diagonal = alignment of matched PCs
                diag_alignment = np.diag(alignment)
                print(f"    PC alignment (correct↔incorrect): PC1={diag_alignment[0]:.3f}, PC2={diag_alignment[1]:.3f}, PC3={diag_alignment[2]:.3f}, PC5={diag_alignment[min(4,len(diag_alignment)-1)]:.3f}")

                # Effective dimensionality via participation ratio
                pr_c = np.sum(eig_c) ** 2 / (np.sum(eig_c ** 2) + 1e-10)
                pr_i = np.sum(eig_i) ** 2 / (np.sum(eig_i ** 2) + 1e-10)
                print(f"    Participation ratio: correct={pr_c:.1f}, incorrect={pr_i:.1f}")

                task_results[f"{model}/separate_pca"] = {
                    "correct_eigenvalues": eig_c.tolist(),
                    "incorrect_eigenvalues": eig_i.tolist(),
                    "correct_var_explained": var_c.tolist(),
                    "incorrect_var_explained": var_i.tolist(),
                    "pc_alignment_diag": diag_alignment.tolist(),
                    "participation_ratio_correct": float(pr_c),
                    "participation_ratio_incorrect": float(pr_i),
                }

            task_results[model] = {
                "n_samples": n_samples_used,
                "n_velocity_vectors": len(V_all),
                "n_correct_vels": len(V_correct),
                "n_incorrect_vels": len(V_incorrect),
                "eigenvalues": eigenvalues.tolist(),
                "var_explained": var_explained.tolist(),
                "spectral_gap_idx": gap_idx,
                "spectral_gap_ratio": gap_ratio,
                "pc_comparison": pc_comparison if proj_correct is not None and proj_incorrect is not None else None,
            }

        # --- Cross-model PC alignment ---
        if len(model_velocities) == 2 and all(m in model_velocities for m in models):
            print(f"\n  Cross-model velocity PC alignment ({models[0]} vs {models[1]}):")
            for label in ["all", "correct", "incorrect"]:
                V0 = model_velocities[models[0]][label]
                V1 = model_velocities[models[1]][label]
                if len(V0) < 50 or len(V1) < 50:
                    continue

                n_sub = min(10, len(V0) - 1, len(V1) - 1)

                V0_c = V0 - np.mean(V0, axis=0)
                _, _, Vt0 = np.linalg.svd(V0_c, full_matrices=False)
                PCs0 = Vt0[:n_sub]

                V1_c = V1 - np.mean(V1, axis=0)
                _, _, Vt1 = np.linalg.svd(V1_c, full_matrices=False)
                PCs1 = Vt1[:n_sub]

                alignment = np.abs(PCs0 @ PCs1.T)
                diag = np.diag(alignment)

                # Also: subspace overlap (Grassmann distance proxy)
                # How much of model0's top-k subspace is captured by model1's top-k
                for k in [3, 5, 10]:
                    if k > n_sub:
                        break
                    overlap = np.trace(PCs0[:k] @ PCs1[:k].T @ PCs1[:k] @ PCs0[:k].T)
                    overlap /= k  # Normalize to [0,1]
                    print(f"    {label}: top-{k} subspace overlap = {overlap:.3f}")

                print(f"    {label}: PC alignment = [{', '.join([f'{d:.3f}' for d in diag[:5]])}]")

                task_results[f"cross_model/{label}"] = {
                    "pc_alignment_diag": diag.tolist(),
                }

        results[task] = task_results

    return results


# =============================================================================
# Phase E: Probe-Informed CKA
# =============================================================================

def _train_probe(data_dir, model, task, layer_idx=-1, max_steps=300):
    """Train correctness probe on mean-pooled generation hidden states. Return (probe, scaler, w, auc)."""
    path = Path(data_dir) / model / f"{task}_generation.h5"
    X, y = [], []
    with h5py.File(path, "r") as f:
        for key in sorted(k for k in f.keys() if k.startswith("sample_")):
            g = f[key]
            if "hidden_states" not in g or int(g.attrs.get("gen_len", 0)) < 3:
                continue
            hs = g["hidden_states"][:max_steps, layer_idx, :].astype(np.float32)
            valid = np.any(hs != 0, axis=1)
            hs = hs[valid]
            if len(hs) > 0:
                X.append(np.mean(hs, axis=0))
                y.append(int(g.attrs.get("is_correct", False)))

    X, y = np.array(X, dtype=np.float32), np.array(y)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X_s, y, cv=cv, scoring="roc_auc")
    clf.fit(X_s, y)

    w = clf.coef_[0].copy()
    w = w / (np.linalg.norm(w) + 1e-10)

    return clf, scaler, w, float(np.mean(aucs)), X_s, y


def phase_e_probe_informed_cka(data_dir, tasks, layer_idx=-1, max_steps=300):
    """Phase E: Probe-Informed CKA — is divergence concentrated in correctness subspace?

    E1: Probe-score tracking — w·hidden at each generation step, by outcome
    E2: Subspace cos_sim — raw vs probe-direction vs complement
    E3: Probe transfer — base probe applied to rl_zero per step
    """
    if not HAS_SKLEARN:
        print("\nSkipping Phase E (sklearn not available)")
        return {}

    print("\n" + "=" * 60)
    print("  PHASE E: PROBE-INFORMED CKA")
    print("=" * 60)

    results = {}
    models = ["olmo3_base", "olmo3_rl_zero"]

    for task in tasks:
        print(f"\n--- {task} ---")
        task_results = {}

        # === Step 1: Train probes on each model ===
        probes = {}
        for model in models:
            path = Path(data_dir) / model / f"{task}_generation.h5"
            if not path.exists():
                continue
            clf, scaler, w, auc, X_s, y = _train_probe(data_dir, model, task, layer_idx, max_steps)
            probes[model] = {"clf": clf, "scaler": scaler, "w": w, "auc": auc}
            print(f"  {model} probe AUC: {auc:.3f} (w norm in scaled space: {np.linalg.norm(clf.coef_[0]):.3f})")

        if len(probes) < 2:
            print(f"  Need both models for CKA analysis")
            continue

        w_base = probes["olmo3_base"]["w"]
        w_rlz = probes["olmo3_rl_zero"]["w"]
        scaler_base = probes["olmo3_base"]["scaler"]

        # Probe direction alignment
        probe_cos = float(np.abs(np.dot(w_base, w_rlz)))
        print(f"  Probe direction alignment (base↔rl_zero): {probe_cos:.4f}")

        task_results["probe_alignment"] = probe_cos
        task_results["base_probe_auc"] = probes["olmo3_base"]["auc"]
        task_results["rlz_probe_auc"] = probes["olmo3_rl_zero"]["auc"]

        # === Step 2: Stream matched samples, compute per-step metrics ===
        base_path = Path(data_dir) / "olmo3_base" / f"{task}_generation.h5"
        rlz_path = Path(data_dir) / "olmo3_rl_zero" / f"{task}_generation.h5"

        # Load metadata for outcome classification
        base_meta = load_generation_data(data_dir, "olmo3_base", task)
        rlz_meta = load_generation_data(data_dir, "olmo3_rl_zero", task)
        if not base_meta or not rlz_meta:
            continue

        base_by_key = {s["key"]: s for s in base_meta}
        rlz_by_key = {s["key"]: s for s in rlz_meta}
        common_keys = sorted(set(base_by_key) & set(rlz_by_key))

        # Per-step accumulators by outcome
        step_data = {
            outcome: {"base_probe_scores": defaultdict(list),
                       "rlz_probe_scores": defaultdict(list),
                       "raw_cos": defaultdict(list),
                       "probe_cos": defaultdict(list),
                       "complement_cos": defaultdict(list)}
            for outcome in ["both_correct", "rlz_wins", "base_wins", "both_wrong"]
        }

        # Build complement projection (random 64D orthogonal to w_base)
        rng = np.random.RandomState(42)
        n_comp = 64
        random_dirs = rng.randn(n_comp, len(w_base))
        random_dirs -= (random_dirs @ w_base[:, None]) * w_base[None, :]
        Q, _ = np.linalg.qr(random_dirs.T)
        P_complement = Q[:, :n_comp].T  # (n_comp, d_model)

        print(f"  Streaming {len(common_keys)} matched samples...")
        n_processed = 0

        with h5py.File(base_path, "r") as fb, h5py.File(rlz_path, "r") as fr:
            for key in common_keys:
                if key not in fb or key not in fr:
                    continue
                gb, gr = fb[key], fr[key]
                if "hidden_states" not in gb or "hidden_states" not in gr:
                    continue

                bc = base_by_key[key]["is_correct"]
                rc = rlz_by_key[key]["is_correct"]
                if bc and rc:
                    outcome = "both_correct"
                elif rc and not bc:
                    outcome = "rlz_wins"
                elif bc and not rc:
                    outcome = "base_wins"
                else:
                    outcome = "both_wrong"

                bh = gb["hidden_states"][:max_steps, layer_idx, :].astype(np.float32)
                rh = gr["hidden_states"][:max_steps, layer_idx, :].astype(np.float32)

                min_len = min(len(bh), len(rh))
                if min_len < 3:
                    continue

                bh, rh = bh[:min_len], rh[:min_len]

                # Filter zero rows
                valid = np.any(bh != 0, axis=1) & np.any(rh != 0, axis=1)
                if valid.sum() < 3:
                    continue

                sd = step_data[outcome]

                for t in range(min_len):
                    if not valid[t]:
                        continue

                    b_vec = bh[t]  # (d_model,)
                    r_vec = rh[t]

                    # E1: Probe scores (dot product with probe direction)
                    b_score = float(np.dot(w_base, b_vec))
                    r_score = float(np.dot(w_base, r_vec))
                    sd["base_probe_scores"][t].append(b_score)
                    sd["rlz_probe_scores"][t].append(r_score)

                    # E2: Cosine similarity decomposition
                    # Raw cos
                    cos_raw = float(np.dot(b_vec, r_vec) / (np.linalg.norm(b_vec) * np.linalg.norm(r_vec) + 1e-10))
                    sd["raw_cos"][t].append(cos_raw)

                    # Probe-direction cos (1D)
                    b_proj = np.dot(w_base, b_vec)
                    r_proj = np.dot(w_base, r_vec)
                    # Sign agreement = cosine in 1D
                    if abs(b_proj) > 1e-8 and abs(r_proj) > 1e-8:
                        cos_probe = float(np.sign(b_proj) * np.sign(r_proj))
                    else:
                        cos_probe = 0.0
                    sd["probe_cos"][t].append(cos_probe)

                    # Complement cos (64D)
                    b_comp = P_complement @ b_vec
                    r_comp = P_complement @ r_vec
                    cos_comp = float(np.dot(b_comp, r_comp) / (np.linalg.norm(b_comp) * np.linalg.norm(r_comp) + 1e-10))
                    sd["complement_cos"][t].append(cos_comp)

                n_processed += 1

        print(f"  Processed {n_processed} matched pairs")

        # === Summarize per-step results ===
        step_bins = [0, 5, 10, 20, 50, 100, 200]

        print(f"\n  E1: Probe Score Tracking (base probe applied to both models)")
        print(f"  {'Outcome':>15s}  {'Step':>5s}  {'Base score':>12s}  {'RLZ score':>12s}  {'Diff':>8s}  {'n':>5s}")
        for outcome in ["both_correct", "rlz_wins", "base_wins", "both_wrong"]:
            sd = step_data[outcome]
            for t in step_bins:
                b_scores = sd["base_probe_scores"].get(t, [])
                r_scores = sd["rlz_probe_scores"].get(t, [])
                if b_scores and r_scores:
                    bm, rm = np.mean(b_scores), np.mean(r_scores)
                    print(f"  {outcome:>15s}  {t:>5d}  {bm:>12.4f}  {rm:>12.4f}  {rm-bm:>8.4f}  {len(b_scores):>5d}")

            e1_data = {}
            for t in sorted(sd["base_probe_scores"].keys()):
                if sd["base_probe_scores"][t]:
                    e1_data[str(t)] = {
                        "base_mean": float(np.mean(sd["base_probe_scores"][t])),
                        "rlz_mean": float(np.mean(sd["rlz_probe_scores"][t])),
                        "n": len(sd["base_probe_scores"][t]),
                    }
            task_results[f"e1/{outcome}"] = e1_data

        print(f"\n  E2: Subspace Cosine Similarity Decomposition")
        print(f"  {'Outcome':>15s}  {'Step':>5s}  {'Raw cos':>10s}  {'Probe sign':>11s}  {'Compl cos':>10s}  {'n':>5s}")
        for outcome in ["both_correct", "rlz_wins", "base_wins", "both_wrong"]:
            sd = step_data[outcome]
            for t in step_bins:
                raw = sd["raw_cos"].get(t, [])
                probe = sd["probe_cos"].get(t, [])
                comp = sd["complement_cos"].get(t, [])
                if raw and probe and comp:
                    print(f"  {outcome:>15s}  {t:>5d}  {np.mean(raw):>10.4f}  {np.mean(probe):>11.4f}  {np.mean(comp):>10.4f}  {len(raw):>5d}")

            e2_data = {}
            for t in sorted(sd["raw_cos"].keys()):
                if sd["raw_cos"][t]:
                    e2_data[str(t)] = {
                        "raw_cos": float(np.mean(sd["raw_cos"][t])),
                        "probe_cos": float(np.mean(sd["probe_cos"][t])),
                        "complement_cos": float(np.mean(sd["complement_cos"][t])),
                        "n": len(sd["raw_cos"][t]),
                    }
            task_results[f"e2/{outcome}"] = e2_data

        # === E3: Probe transfer — base probe correctness AUC at each step ===
        print(f"\n  E3: Per-Step Probe Transfer (base probe → rl_zero correctness)")
        e3_results = {}
        with h5py.File(rlz_path, "r") as fr:
            keys = sorted(k for k in fr.keys() if k.startswith("sample_"))
            # Collect per-step scores
            step_scores = defaultdict(lambda: {"scores": [], "labels": []})
            for key in keys:
                g = fr[key]
                if "hidden_states" not in g or int(g.attrs.get("gen_len", 0)) < 3:
                    continue
                label = int(g.attrs.get("is_correct", False))
                hs = g["hidden_states"][:max_steps, layer_idx, :].astype(np.float32)
                valid = np.any(hs != 0, axis=1)
                for t in range(len(hs)):
                    if valid[t]:
                        score = float(np.dot(w_base, hs[t]))
                        step_scores[t]["scores"].append(score)
                        step_scores[t]["labels"].append(label)

        for t in step_bins:
            ss = step_scores.get(t)
            if ss and len(ss["scores"]) > 20 and sum(ss["labels"]) > 3:
                scores_arr = np.array(ss["scores"])
                labels_arr = np.array(ss["labels"])
                try:
                    auc = roc_auc_score(labels_arr, scores_arr)
                    print(f"    Step {t:>5d}: AUC = {auc:.3f} (n={len(scores_arr)}, {sum(labels_arr)} correct)")
                    e3_results[str(t)] = {"auc": float(auc), "n": len(scores_arr)}
                except Exception:
                    pass
        task_results["e3_transfer"] = e3_results

        results[task] = task_results

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/generation_trajectories")
    parser.add_argument("--output-dir", default="experiments/generation_dynamics/results")
    parser.add_argument("--phases", default="abc", help="Which phases to run (a, b, c)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models = ["olmo3_base", "olmo3_rl_zero"]
    tasks = ["gsm8k", "humaneval", "logiqa"]

    all_results = {}

    if "a" in args.phases.lower():
        all_results["phase_a"] = phase_a_descriptive(args.data_dir, models, tasks)
        all_results["phase_a2"] = phase_a2_entropy_profiles(args.data_dir, models, tasks)

    if "b" in args.phases.lower():
        all_results["phase_b"] = phase_b_correctness_prediction(args.data_dir, models, tasks)

    if "c" in args.phases.lower():
        all_results["phase_c"] = phase_c_divergence(args.data_dir, tasks)

    if "d" in args.phases.lower():
        all_results["phase_d"] = phase_d_velocity_pca(args.data_dir, models, tasks)

    if "e" in args.phases.lower():
        all_results["phase_e"] = phase_e_probe_informed_cka(args.data_dir, tasks)

    # Save results
    output_path = Path(args.output_dir) / "generation_dynamics_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
