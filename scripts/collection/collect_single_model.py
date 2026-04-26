#!/usr/bin/env python3
"""
Single-model trajectory collection for parallel GPU execution.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/collect_single_model.py olmo3_base
    CUDA_VISIBLE_DEVICES=1 python scripts/collect_single_model.py olmo3_sft olmo3_think

This script collects trajectories for specified models only, allowing
parallel execution across multiple GPUs.
"""

import sys
import os

# Set CUDA device before importing torch
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import h5py
import torch
import numpy as np
import yaml
import json
from datetime import datetime
from tqdm import tqdm

from task_data import prepare_gsm8k, prepare_humaneval, prepare_logiqa

# Import the collection infrastructure from main script
from collect_trajectories_with_labels import (
    TrajectoryCollector,
    check_correctness,
    load_checkpoint,
    save_checkpoint,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    N_SAMPLES,
    MAX_SEQ_LEN,
    MAX_NEW_TOKENS,
    LAYERS_TO_COLLECT,
)


def load_models_config():
    """Load model configurations."""
    config_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Flatten primary and secondary into single dict
    models = {}
    for section in ['primary', 'secondary']:
        if section in config:
            models.update(config[section])
    return models


def collect_for_model(model_key: str, model_config: dict, n_samples: int = N_SAMPLES):
    """Collect trajectories for a single model across all tasks."""

    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"HF Name: {model_config['model_name']}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"{'='*60}")

    task_prep_fns = {
        'gsm8k': prepare_gsm8k,
        'humaneval': prepare_humaneval,
        'logiqa': prepare_logiqa
    }

    # Initialize collector once for all tasks
    collector = None

    for task_name, prep_fn in task_prep_fns.items():
        print(f"\nTask: {task_name}")

        # Check if already complete
        checkpoint = load_checkpoint(model_key, task_name)
        if checkpoint["completed_samples"] >= n_samples:
            print(f"  Already complete: {checkpoint['n_correct']} correct, {checkpoint['n_incorrect']} incorrect")
            continue

        # Load collector if not already loaded
        if collector is None:
            collector = TrajectoryCollector(
                model_name=model_config['model_name'],
                layers_to_collect=LAYERS_TO_COLLECT,
            )

        # Prepare task data
        task_data = prep_fn(n_samples=n_samples, split='test')

        # Output file
        output_file = OUTPUT_DIR / model_key / f"{task_name}_trajectories.h5"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        start_idx = checkpoint["completed_samples"]
        n_correct = checkpoint["n_correct"]
        n_incorrect = checkpoint["n_incorrect"]

        # Create HDF5 file if starting fresh
        if start_idx == 0:
            with h5py.File(output_file, 'w') as f:
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

                f.attrs['model'] = model_key
                f.attrs['task'] = task_name
                f.attrs['n_samples'] = n_samples
                f.attrs['max_seq_len'] = MAX_SEQ_LEN
                f.attrs['layers'] = LAYERS_TO_COLLECT
                f.attrs['d_model'] = collector.d_model
                f.attrs['collection_date'] = datetime.now().isoformat()

        # Collect
        print(f"  Collecting {n_samples - start_idx} samples (from {start_idx})...")

        with h5py.File(output_file, 'a') as f:
            pbar = tqdm(range(start_idx, n_samples), desc=f"  {model_key}/{task_name}")

            for i in pbar:
                prompt, answer, metadata = task_data[i]

                try:
                    model_output, trajectory = collector.generate_with_trajectory(
                        prompt=prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        max_seq_len=MAX_SEQ_LEN
                    )

                    is_correct = check_correctness(model_output, answer, task_name, metadata)
                    seq_len = trajectory.shape[0]

                    # Pad/truncate
                    if seq_len > MAX_SEQ_LEN:
                        trajectory = trajectory[:MAX_SEQ_LEN]
                        seq_len = MAX_SEQ_LEN
                    elif seq_len < MAX_SEQ_LEN:
                        padding = np.zeros(
                            (MAX_SEQ_LEN - seq_len, len(LAYERS_TO_COLLECT), collector.d_model),
                            dtype=np.float16
                        )
                        trajectory = np.vstack([trajectory, padding])

                    f['trajectories'][i] = trajectory.astype(np.float16)
                    f['sequence_lengths'][i] = seq_len
                    f['is_correct'][i] = is_correct
                    f['prompts'][i] = prompt
                    f['model_outputs'][i] = model_output[:10000]
                    f['ground_truth'][i] = str(answer)[:5000]

                    if is_correct:
                        n_correct += 1
                    else:
                        n_incorrect += 1

                    pbar.set_postfix({'correct': n_correct, 'incorrect': n_incorrect})

                except Exception as e:
                    print(f"\n  Error on sample {i}: {e}")
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

                if (i + 1) % 25 == 0:
                    save_checkpoint(model_key, task_name, i + 1, n_correct, n_incorrect)

        save_checkpoint(model_key, task_name, n_samples, n_correct, n_incorrect)
        print(f"  Complete: {n_correct} correct, {n_incorrect} incorrect")

    # Cleanup
    if collector is not None:
        del collector
        torch.cuda.empty_cache()


def main():
    if len(sys.argv) < 2:
        print("Usage: python collect_single_model.py <model_key> [model_key2] ...")
        print("Available models: olmo3_base, olmo3_sft, olmo3_rl_zero, olmo3_think, deepseek_r1")
        return

    model_keys = sys.argv[1:]
    models = load_models_config()

    for model_key in model_keys:
        if model_key not in models:
            print(f"Unknown model: {model_key}")
            print(f"Available: {list(models.keys())}")
            continue

        try:
            collect_for_model(model_key, models[model_key])
        except Exception as e:
            print(f"Error collecting {model_key}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
