#!/usr/bin/env python3
"""
Quick test of Lyapunov exponent computation.
Tests both fast (norm ratio) and full SVD methods.
"""

import numpy as np
import h5py
import time
from scipy import linalg

print("Testing Lyapunov computation with small samples...")

# Load a small subset
filepath = "/data/thanhdo/trajectories_0shot/olmo3_base/humaneval_trajectories.h5"
with h5py.File(filepath, "r") as f:
    traj = f["trajectories"][:20].astype(np.float32)
    labels = f["is_correct"][:20]

print(f"Loaded: {traj.shape}")
print(f"Correct samples: {labels.sum()}/{len(labels)}")


def compute_lyapunov_fast(trajectory):
    """Fast Lyapunov using Frobenius norm ratio as proxy."""
    seq_len, n_layers, d_model = trajectory.shape
    layer_lyapunov = []

    for l in range(n_layers - 1):
        x_l = trajectory[:, l, :]
        x_l1 = trajectory[:, l+1, :]
        # Frobenius norm ratio as proxy for expansion
        norm_ratio = np.linalg.norm(x_l1, 'fro') / (np.linalg.norm(x_l, 'fro') + 1e-8)
        expansion = np.log(norm_ratio + 1e-8)
        layer_lyapunov.append(expansion)

    return {
        'mean': float(np.mean(layer_lyapunov)),
        'max': float(np.max(layer_lyapunov)),
        'std': float(np.std(layer_lyapunov))
    }


def compute_lyapunov_svd(trajectory, max_seq=64):
    """Full SVD-based Lyapunov (slower but more accurate)."""
    seq_len, n_layers, d_model = trajectory.shape
    # Subsample sequence for speed
    if seq_len > max_seq:
        idx = np.linspace(0, seq_len-1, max_seq, dtype=int)
        trajectory = trajectory[idx, :, :]
        seq_len = max_seq

    layer_lyapunov = []

    for l in range(n_layers - 1):
        x_l = trajectory[:, l, :]      # (seq_len, d_model)
        x_l1 = trajectory[:, l+1, :]   # (seq_len, d_model)
        delta_x = x_l1 - x_l

        try:
            # Truncated SVD for speed (only top k singular values)
            _, s, _ = linalg.svd(delta_x, full_matrices=False)
            if len(s) > 1 and s[-1] > 1e-10:
                expansion = np.log(s[0] / (s[-1] + 1e-8))
            else:
                expansion = 0
        except:
            expansion = 0

        layer_lyapunov.append(expansion)

    return {
        'mean': float(np.mean(layer_lyapunov)),
        'max': float(np.max(layer_lyapunov)),
        'std': float(np.std(layer_lyapunov))
    }


# Test fast method
print("\n=== Fast Method (Frobenius norm ratio) ===")
results_fast = []
for i in range(len(traj)):
    t0 = time.time()
    lyap = compute_lyapunov_fast(traj[i])
    dt = time.time() - t0
    results_fast.append(lyap)
    label = "C" if labels[i] else "I"
    print(f"  Sample {i:2d} [{label}]: mean={lyap['mean']:.4f}, max={lyap['max']:.4f}, time={dt:.2f}s")

correct_fast = [r['mean'] for i, r in enumerate(results_fast) if labels[i]]
incorrect_fast = [r['mean'] for i, r in enumerate(results_fast) if not labels[i]]

print("\nSummary (Fast):")
if correct_fast:
    print(f"  Correct ({len(correct_fast)}): mean={np.mean(correct_fast):.4f} +/- {np.std(correct_fast):.4f}")
if incorrect_fast:
    print(f"  Incorrect ({len(incorrect_fast)}): mean={np.mean(incorrect_fast):.4f} +/- {np.std(incorrect_fast):.4f}")


# Test SVD method on first 5 samples
print("\n=== SVD Method (on 5 samples) ===")
results_svd = []
for i in range(min(5, len(traj))):
    t0 = time.time()
    lyap = compute_lyapunov_svd(traj[i], max_seq=64)
    dt = time.time() - t0
    results_svd.append(lyap)
    label = "C" if labels[i] else "I"
    print(f"  Sample {i:2d} [{label}]: mean={lyap['mean']:.4f}, max={lyap['max']:.4f}, time={dt:.2f}s")

print("\n[OK] Lyapunov test completed!")
print("\nConclusion:")
print("  - Fast method is ~instant per sample")
print("  - SVD method takes a few seconds per sample (with seq subsampling)")
print("  - For 50+ samples, use fast method or CPU-parallel SVD")
