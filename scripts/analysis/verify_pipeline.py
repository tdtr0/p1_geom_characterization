#!/usr/bin/env python3
"""
Pipeline Verification Script

Tests the activation collection pipeline on a small sample to ensure:
1. Model loads correctly
2. Activations can be extracted
3. Shapes are as expected
4. Storage works
5. Loading works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from activation_collector import ActivationCollector
from task_data import get_task_data
from geometric_measures import compute_effective_rank, compute_spectral_decay, analyze_spectrum
import numpy as np
import torch


def verify_model_loading(model_name: str = "gpt2"):
    """Test 1: Verify model loads correctly."""
    print(f"\n{'='*60}")
    print("TEST 1: Model Loading")
    print(f"{'='*60}")

    try:
        collector = ActivationCollector(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16
        )
        print(f"✓ Model loaded: {collector.n_layers} layers, d_model={collector.d_model}")
        return collector
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        raise


def verify_activation_collection(collector: ActivationCollector):
    """Test 2: Verify activation extraction works."""
    print(f"\n{'='*60}")
    print("TEST 2: Activation Collection")
    print(f"{'='*60}")

    test_prompts = [
        "What is 2 + 2? Let me think step by step.",
        "Write a function to compute factorial.",
        "The sky is blue because",
    ]

    try:
        activations = collector.collect_activations(
            test_prompts,
            aggregation="last_token",
            batch_size=1
        )

        print(f"✓ Collected activations for {len(test_prompts)} prompts")

        # Verify shapes
        expected_shape = (len(test_prompts), collector.d_model)
        for name, arr in activations.items():
            if arr.shape != expected_shape:
                print(f"✗ Unexpected shape for {name}: {arr.shape} (expected {expected_shape})")
                return None
            # Check for NaN/Inf
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                print(f"✗ {name} contains NaN or Inf")
                return None

        print(f"✓ All shapes correct: {expected_shape}")
        print(f"✓ No NaN/Inf values detected")

        # Check activation magnitudes
        sample_key = list(activations.keys())[len(activations) // 2]  # Middle layer
        sample_act = activations[sample_key]
        mean_norm = np.linalg.norm(sample_act, axis=1).mean()
        print(f"✓ Sample activation norm: {mean_norm:.2f}")

        if mean_norm < 0.1 or mean_norm > 1000:
            print(f"⚠ Warning: Unusual activation magnitude")

        return activations

    except Exception as e:
        print(f"✗ Activation collection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_storage(collector: ActivationCollector, activations: dict):
    """Test 3: Verify HDF5 storage works."""
    print(f"\n{'='*60}")
    print("TEST 3: HDF5 Storage")
    print(f"{'='*60}")

    test_file = Path("test_activations.h5")

    try:
        # Save
        metadata = {
            "model": collector.model_name,
            "n_samples": 3,
            "test": True,
        }
        collector.save_to_hdf5(activations, str(test_file), metadata)
        print(f"✓ Saved to {test_file}")

        # Load
        loaded_acts, loaded_meta = collector.load_from_hdf5(str(test_file))
        print(f"✓ Loaded from {test_file}")

        # Verify
        for key in activations.keys():
            if not np.allclose(activations[key], loaded_acts[key], rtol=1e-3):
                print(f"✗ Data mismatch for {key}")
                return False

        print(f"✓ Data integrity verified")
        print(f"✓ Metadata: {loaded_meta}")

        # Cleanup
        test_file.unlink()
        print(f"✓ Cleaned up test file")

        return True

    except Exception as e:
        print(f"✗ Storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_geometric_measures(activations: dict):
    """Test 4: Verify geometric measures compute correctly."""
    print(f"\n{'='*60}")
    print("TEST 4: Geometric Measures")
    print(f"{'='*60}")

    try:
        # Take a sample layer
        sample_key = list(activations.keys())[len(activations) // 2]
        sample_act = activations[sample_key]

        print(f"Computing measures on {sample_key}...")

        # Effective rank
        eff_rank = compute_effective_rank(sample_act)
        print(f"✓ Effective rank: {eff_rank:.2f}")

        # Spectral decay
        alpha, s, r2 = compute_spectral_decay(sample_act)
        print(f"✓ Spectral decay α: {alpha:.3f} (R²={r2:.3f})")

        # Comprehensive analysis
        spectrum = analyze_spectrum(sample_act)
        print(f"✓ Participation ratio: {spectrum['participation_ratio']:.2f}")
        print(f"✓ Stable rank: {spectrum['stable_rank']:.2f}")
        print(f"✓ 90% energy: {spectrum['n_components_90pct']} components")

        # Sanity checks
        n_samples, d_model = sample_act.shape
        max_rank = min(n_samples, d_model)

        if not (1 <= eff_rank <= max_rank):
            print(f"✗ Effective rank out of range: {eff_rank}")
            return False

        if alpha < 0:
            print(f"✗ Negative spectral decay: {alpha}")
            return False

        print(f"✓ All geometric measures computed successfully")
        return True

    except Exception as e:
        print(f"✗ Geometric measures failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_task_data():
    """Test 5: Verify task data loading."""
    print(f"\n{'='*60}")
    print("TEST 5: Task Data Loading")
    print(f"{'='*60}")

    tasks_to_test = ["gsm8k", "humaneval"]  # Skip logiqa for now
    n_samples = 5

    for task in tasks_to_test:
        try:
            data = get_task_data(task, n_samples=n_samples)
            if len(data) != n_samples:
                print(f"✗ {task}: Expected {n_samples} samples, got {len(data)}")
                return False

            # Verify structure
            prompt, answer, metadata = data[0]
            if not isinstance(prompt, str) or not isinstance(answer, str):
                print(f"✗ {task}: Invalid data types")
                return False

            print(f"✓ {task}: Loaded {len(data)} samples")
            print(f"  Sample prompt length: {len(prompt)} chars")

        except Exception as e:
            print(f"✗ {task} loading failed: {e}")
            return False

    print(f"✓ All task data loading successful")
    return True


def main():
    """Run all verification tests."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use")
    args = parser.parse_args()

    # Set GPU if specified
    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU {args.gpu}")

    print("\n" + "="*60)
    print("PIPELINE VERIFICATION")
    print("="*60)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available, using CPU")

    # Test 5: Task data (doesn't require model)
    if not verify_task_data():
        print("\n✗ VERIFICATION FAILED: Task data loading")
        sys.exit(1)

    # Test 1: Model loading
    try:
        # Use a small model for testing
        model_name = "gpt2"  # Small and fast to download
        print(f"\nUsing test model: {model_name}")
        collector = verify_model_loading(model_name)
    except Exception as e:
        print(f"\n✗ VERIFICATION FAILED: Model loading")
        sys.exit(1)

    # Test 2: Activation collection
    activations = verify_activation_collection(collector)
    if activations is None:
        print("\n✗ VERIFICATION FAILED: Activation collection")
        sys.exit(1)

    # Test 3: Storage
    if not verify_storage(collector, activations):
        print("\n✗ VERIFICATION FAILED: Storage")
        sys.exit(1)

    # Test 4: Geometric measures
    if not verify_geometric_measures(activations):
        print("\n✗ VERIFICATION FAILED: Geometric measures")
        sys.exit(1)

    # Final summary
    print("\n" + "="*60)
    print("✓ ALL VERIFICATION TESTS PASSED")
    print("="*60)
    print("\nPipeline is ready for data collection!")
    print("\nNext steps:")
    print("  1. Run activation collection: python scripts/collect_activations.py")
    print("  2. Run analysis: python scripts/run_analysis.py")


if __name__ == "__main__":
    main()
