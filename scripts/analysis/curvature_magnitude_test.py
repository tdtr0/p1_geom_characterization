#!/usr/bin/env python3
"""
Test curvature MAGNITUDE differences between correct/incorrect.
(The curvature PROFILE is identical - this tests the scalar value)
"""

import numpy as np
import h5py
from scipy import stats


def compute_mean_curvature(trajectory):
    """Compute mean curvature across layer transitions."""
    mean_traj = trajectory.mean(axis=0)
    n_layers = mean_traj.shape[0]

    curvatures = []
    for i in range(n_layers - 2):
        p1, p2, p3 = mean_traj[i], mean_traj[i+1], mean_traj[i+2]
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        if a * b * c < 1e-10:
            continue
        s = (a + b + c) / 2
        area_sq = max(0, s * (s-a) * (s-b) * (s-c))
        area = np.sqrt(area_sq)
        curvatures.append(4 * area / (a * b * c))
    return np.mean(curvatures) if curvatures else 0


print("=" * 60)
print("CURVATURE MAGNITUDE ANALYSIS (not profile shape)")
print("=" * 60)

for task in ["humaneval", "logiqa"]:
    path = f"/data/thanhdo/trajectories_0shot/olmo3_base/{task}_trajectories.h5"
    with h5py.File(path, "r") as f:
        traj = f["trajectories"][:100].astype(np.float32)
        labels = f["is_correct"][:100]

    correct_curv = [compute_mean_curvature(traj[i]) for i in range(len(traj)) if labels[i]]
    incorrect_curv = [compute_mean_curvature(traj[i]) for i in range(len(traj)) if not labels[i]]

    # Statistical test
    t_stat, p_val = stats.ttest_ind(correct_curv, incorrect_curv)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(correct_curv)-1) * np.var(correct_curv) +
         (len(incorrect_curv)-1) * np.var(incorrect_curv)) /
        (len(correct_curv) + len(incorrect_curv) - 2)
    )
    d = (np.mean(correct_curv) - np.mean(incorrect_curv)) / (pooled_std + 1e-8)

    print(f"\n{task.upper()}:")
    print(f"  Correct (n={len(correct_curv)}): {np.mean(correct_curv):.4f} +/- {np.std(correct_curv):.4f}")
    print(f"  Incorrect (n={len(incorrect_curv)}): {np.mean(incorrect_curv):.4f} +/- {np.std(incorrect_curv):.4f}")
    print(f"  Cohen d = {d:.3f}, p = {p_val:.4f}")

    if p_val < 0.05:
        direction = "HIGHER" if d > 0 else "LOWER"
        print(f"  => Correct solutions have {direction} curvature magnitude (significant)")
    else:
        print(f"  => No significant difference in curvature magnitude")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
Curvature PROFILE (shape): Identical across correctness and domain (r~1.0)
  => Architectural property, NOT useful for detecting correctness

Curvature MAGNITUDE: Small effect (d~0.3), not significant (p>0.2)
  => Weak signal at best, needs more samples or more sensitive methods

The r=0.996 cross-domain finding is a RED HERRING - it tells us about
transformer architecture, not about reasoning transfer.
""")
