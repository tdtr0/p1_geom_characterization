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

    # Save results
    output_path = Path(args.output_dir) / "generation_dynamics_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
