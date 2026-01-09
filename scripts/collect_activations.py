#!/usr/bin/env python3
"""
Fault-Tolerant Activation Collection Script

Features:
- Checkpointing every N samples
- Auto-resume from checkpoint on restart
- OOM handling with batch size reduction
- Progress tracking
- GPU selection
"""

import sys
import os
import argparse
from pathlib import Path

# Parse GPU argument BEFORE importing torch to set CUDA_VISIBLE_DEVICES
def parse_gpu_arg():
    """Quick parse of --gpu argument before torch import."""
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")
            return int(gpu_id)
    return None

gpu_id = parse_gpu_arg()

import yaml
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from activation_collector import ActivationCollector
from task_data import get_task_data
from checkpointing import CheckpointManager, create_partial_output_path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Collect activations with checkpointing")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to process (default: all from config)")
    parser.add_argument("--tasks", nargs="+", default=None,
                       help="Tasks to process (default: all from config)")
    parser.add_argument("--output-dir", type=str, default="data/activations",
                       help="Output directory")
    parser.add_argument("--config", type=str, default="configs/models.yaml",
                       help="Config file")
    parser.add_argument("--checkpoint-freq", type=int, default=50,
                       help="Save checkpoint every N samples")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for processing")
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU ID to use (default: auto)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")
    parser.add_argument("--force", action="store_true",
                       help="Force recompute even if output exists")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: only process 10 samples")
    return parser.parse_args()


def load_config(config_path: str):
    """Load model and task configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_gpu(gpu_id: int = None):
    """
    Set up GPU environment.
    Note: CUDA_VISIBLE_DEVICES already set at import time if --gpu was specified.
    """
    if gpu_id is not None:
        print(f"Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES already set)")

    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("⚠ CUDA not available, using CPU")

    return device


def collect_with_checkpointing(
    model_name: str,
    model_key: str,
    task_name: str,
    output_file: str,
    n_samples: int,
    batch_size: int,
    checkpoint_freq: int,
    device: str,
    resume: bool = True,
    checkpoint_manager: CheckpointManager = None
):
    """
    Collect activations with checkpointing.

    Handles:
    - Resume from checkpoint
    - OOM with batch size reduction
    - Regular checkpointing
    - Graceful failure handling
    """
    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager()

    # Check if already completed
    if checkpoint_manager.is_completed(model_key, task_name):
        print(f"✓ {model_key}/{task_name} already completed, skipping")
        return True

    # Check for existing output (if not forcing)
    output_path = Path(output_file)
    if output_path.exists() and not resume:
        print(f"✓ Output exists: {output_file}, skipping")
        return True

    # Load checkpoint if resuming
    start_idx = 0
    completed = 0
    if resume:
        resume_info = checkpoint_manager.get_resume_info(model_key, task_name)
        if resume_info:
            start_idx = resume_info["start_idx"]
            completed = resume_info["completed"]
            print(f"↻ Resuming from sample {start_idx} ({completed}/{n_samples} done)")

    # Load task data
    print(f"\nLoading task data: {task_name}")
    try:
        task_data = get_task_data(task_name, n_samples=n_samples)
        prompts = [item[0] for item in task_data]
    except Exception as e:
        print(f"✗ Failed to load task {task_name}: {e}")
        checkpoint_manager.save_checkpoint(
            model_key, task_name, 0, n_samples, -1, output_file, status="failed"
        )
        return False

    # Load model
    print(f"\nLoading model: {model_name}")
    try:
        collector = ActivationCollector(
            model_name,
            device=device,
            dtype=torch.float16
        )
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        checkpoint_manager.save_checkpoint(
            model_key, task_name, 0, n_samples, -1, output_file, status="failed"
        )
        return False

    # Collect activations in chunks
    all_activations = None
    current_batch_size = batch_size

    for chunk_start in range(start_idx, len(prompts), checkpoint_freq):
        chunk_end = min(chunk_start + checkpoint_freq, len(prompts))
        chunk = prompts[chunk_start:chunk_end]

        print(f"\n[{chunk_start:4d}-{chunk_end:4d}/{len(prompts)}] Processing {len(chunk)} samples...")

        # Try collecting with current batch size
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                chunk_acts = collector.collect_activations(
                    chunk,
                    aggregation="last_token",
                    batch_size=current_batch_size,
                    max_length=2048
                )

                # Merge with existing activations
                if all_activations is None:
                    all_activations = chunk_acts
                else:
                    for key in all_activations:
                        all_activations[key] = np.concatenate([
                            all_activations[key],
                            chunk_acts[key]
                        ], axis=0)

                # Success - save checkpoint
                completed = chunk_end
                checkpoint_manager.save_checkpoint(
                    model_key,
                    task_name,
                    completed,
                    n_samples,
                    chunk_end - 1,
                    output_file,
                    status="in_progress",
                    batch_size=current_batch_size
                )

                print(f"✓ Checkpoint saved ({completed}/{n_samples} samples)")
                break  # Success, move to next chunk

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # OOM - reduce batch size and retry
                    retry_count += 1
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"⚠ OOM! Reducing batch size to {current_batch_size}, retry {retry_count}/{max_retries}")
                    torch.cuda.empty_cache()

                    if retry_count >= max_retries:
                        print(f"✗ Failed after {max_retries} retries")
                        checkpoint_manager.save_checkpoint(
                            model_key, task_name, completed, n_samples,
                            chunk_start - 1, output_file, status="failed"
                        )
                        return False
                else:
                    # Other error
                    print(f"✗ Error: {e}")
                    checkpoint_manager.save_checkpoint(
                        model_key, task_name, completed, n_samples,
                        chunk_start - 1, output_file, status="failed"
                    )
                    return False

    # Save final output
    print(f"\n💾 Saving final output to {output_file}")
    metadata = {
        "model": model_name,
        "model_key": model_key,
        "task": task_name,
        "n_samples": len(prompts),
        "collection_date": datetime.now().isoformat(),
        "aggregation": "last_token",
    }

    try:
        collector.save_to_hdf5(all_activations, output_file, metadata)

        # Mark as completed
        checkpoint_manager.save_checkpoint(
            model_key, task_name, len(prompts), n_samples,
            len(prompts) - 1, output_file, status="completed"
        )

        print(f"✓ Completed {model_key}/{task_name}")
        return True

    except Exception as e:
        print(f"✗ Failed to save: {e}")
        return False
    finally:
        # Clean up
        del collector
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    print("="*80)
    print("ACTIVATION COLLECTION WITH CHECKPOINTING")
    print("="*80)

    # Setup
    device = setup_gpu(args.gpu)
    config = load_config(args.config)
    checkpoint_manager = CheckpointManager()

    # Determine models and tasks to process
    if args.models is None:
        # Use primary models from config
        models = config["primary"]
    else:
        models = {k: config["primary"][k] for k in args.models if k in config["primary"]}

    if args.tasks is None:
        tasks = list(config["tasks"].keys())
    else:
        tasks = args.tasks

    # Test mode
    n_samples_per_task = 10 if args.test else None

    print(f"\nModels to process: {list(models.keys())}")
    print(f"Tasks to process: {tasks}")
    if args.test:
        print("⚠ TEST MODE: Only processing 10 samples per task")
    print()

    # Process each model/task combination
    results = []

    for model_key, model_info in models.items():
        model_name = model_info["model_name"]

        # Create output directory for this model
        model_dir = Path(args.output_dir) / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        for task_name in tasks:
            task_config = config["tasks"][task_name]
            n_samples = n_samples_per_task or task_config["n_samples"]

            output_file = str(model_dir / f"{task_name}.h5")

            print("\n" + "="*80)
            print(f"Processing: {model_key} / {task_name}")
            print("="*80)

            success = collect_with_checkpointing(
                model_name=model_name,
                model_key=model_key,
                task_name=task_name,
                output_file=output_file,
                n_samples=n_samples,
                batch_size=args.batch_size,
                checkpoint_freq=args.checkpoint_freq,
                device=device,
                resume=args.resume,
                checkpoint_manager=checkpoint_manager
            )

            results.append({
                "model": model_key,
                "task": task_name,
                "success": success
            })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['model']:20s} / {result['task']:15s}")

    success_count = sum(1 for r in results if r["success"])
    print(f"\nCompleted: {success_count}/{len(results)}")

    if success_count == len(results):
        print("\n✓ ALL COLLECTIONS SUCCESSFUL")
    else:
        print(f"\n⚠ {len(results) - success_count} collections failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
