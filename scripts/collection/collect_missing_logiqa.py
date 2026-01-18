#!/usr/bin/env python3
"""Collect just the 3 missing LogiQA trajectory files.

Models to collect:
- olmo3_sft
- olmo3_rl_zero
- olmo3_think

Usage:
    PYTHONPATH=./src python scripts/collection/collect_missing_logiqa.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import everything from the main collection script
from collect_trajectories_with_labels import (
    TrajectoryCollector,
    check_logiqa_correct,
    prepare_logiqa,
    LAYERS_TO_COLLECT,
    N_SAMPLES,
    MAX_SEQ_LEN,
    MAX_NEW_TOKENS,
    OUTPUT_DIR,
    load_models_config
)

import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# Models to collect (the 3 missing ones)
MODELS_TO_COLLECT = ['olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']


def collect_for_model(model_key: str, model_config: dict, task_data: list, n_samples: int):
    """Collect LogiQA trajectories for a single model."""

    # Output file
    output_dir = OUTPUT_DIR / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "logiqa_trajectories.h5"

    # Skip if exists
    if output_file.exists():
        print(f"  {output_file} already exists, skipping")
        return

    print(f"  Output: {output_file}")

    # Initialize collector
    collector = TrajectoryCollector(
        model_name=model_config['model_name'],
        layers_to_collect=LAYERS_TO_COLLECT
    )

    # Track correctness
    n_correct = 0
    n_incorrect = 0

    # Create HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        f.create_dataset(
            'trajectories',
            shape=(n_samples, MAX_SEQ_LEN, len(LAYERS_TO_COLLECT), collector.d_model),
            dtype='float16',
            compression='gzip',
            compression_opts=4
        )
        f.create_dataset('sequence_lengths', shape=(n_samples,), dtype='int32')
        f.create_dataset('is_correct', shape=(n_samples,), dtype='bool')
        f.create_dataset('prompts', shape=(n_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset('model_outputs', shape=(n_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset('ground_truth', shape=(n_samples,), dtype=h5py.string_dtype(encoding='utf-8'))

        # Collect
        pbar = tqdm(range(n_samples), desc=f"  {model_key}/logiqa")

        for i in pbar:
            prompt, answer, metadata = task_data[i]

            try:
                # Generate with trajectory collection
                model_output, trajectory = collector.generate_with_trajectory(
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    max_seq_len=MAX_SEQ_LEN
                )

                # Check correctness
                is_correct = check_logiqa_correct(model_output, answer)

                # Get sequence length
                seq_len = trajectory.shape[0]

                # Pad or truncate trajectory
                if seq_len > MAX_SEQ_LEN:
                    trajectory = trajectory[:MAX_SEQ_LEN]
                    seq_len = MAX_SEQ_LEN
                elif seq_len < MAX_SEQ_LEN:
                    padding = np.zeros(
                        (MAX_SEQ_LEN - seq_len, len(LAYERS_TO_COLLECT), collector.d_model),
                        dtype=np.float32
                    )
                    trajectory = np.vstack([trajectory, padding])

                # Store
                f['trajectories'][i] = trajectory.astype(np.float16)
                f['sequence_lengths'][i] = seq_len
                f['is_correct'][i] = is_correct
                f['prompts'][i] = prompt
                f['model_outputs'][i] = model_output[:10000]
                f['ground_truth'][i] = str(answer)[:5000]

                # Update counters
                if is_correct:
                    n_correct += 1
                else:
                    n_incorrect += 1

                pbar.set_postfix(correct=n_correct, incorrect=n_incorrect)

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                # Store placeholder
                f['trajectories'][i] = np.zeros(
                    (MAX_SEQ_LEN, len(LAYERS_TO_COLLECT), collector.d_model),
                    dtype=np.float16
                )
                f['sequence_lengths'][i] = 0
                f['is_correct'][i] = False
                f['prompts'][i] = prompt
                f['model_outputs'][i] = f"ERROR: {str(e)}"
                f['ground_truth'][i] = str(answer)[:5000]
                n_incorrect += 1

    # Clean up
    del collector
    torch.cuda.empty_cache()

    # Report
    file_size_gb = output_file.stat().st_size / 1e9
    print(f"  Completed: {n_correct} correct, {n_incorrect} incorrect")
    print(f"  File size: {file_size_gb:.2f} GB")


def main():
    print("=" * 80)
    print("Collecting Missing LogiQA Trajectories")
    print("=" * 80)
    print(f"Models: {MODELS_TO_COLLECT}")
    print(f"Samples: {N_SAMPLES}")
    print()

    # Load model configs
    models = load_models_config()

    # Prepare LogiQA data once
    print("Preparing LogiQA data...")
    task_data = prepare_logiqa(n_samples=N_SAMPLES, split='test')
    print(f"Loaded {len(task_data)} samples")
    print()

    # Collect for each model
    for model_key in MODELS_TO_COLLECT:
        if model_key not in models:
            print(f"Model {model_key} not in config, skipping")
            continue

        model_config = models[model_key]
        print(f"\n{'=' * 60}")
        print(f"Model: {model_key} ({model_config['model_name']})")
        print(f"{'=' * 60}")

        try:
            collect_for_model(
                model_key=model_key,
                model_config=model_config,
                task_data=task_data,
                n_samples=N_SAMPLES
            )
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

        # Free GPU memory
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
