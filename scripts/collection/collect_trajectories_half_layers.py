#!/usr/bin/env python3
"""
Phase 2: Collect full token trajectories (half layers - even indices only)

Strategy: Even layers [0, 2, 4, ..., 30] = 16 layers
Storage: ~56 GB total (vs 112 GB for all 32 layers)

Rationale:
- Layer smoothness analysis shows max jump of 0.09% between consecutive layers
- Even/odd layer means differ by < 0.01%
- No critical non-linear transitions detected
- Half-layer sampling is safe and reduces storage by 2x
"""

import os
import sys
import h5py
import torch
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from activation_collector import ActivationCollector
from task_data import prepare_gsm8k, prepare_humaneval, prepare_logiqa

# Configuration
OUTPUT_DIR = Path("data/trajectories")
CHECKPOINT_DIR = Path("data/checkpoints")
N_SAMPLES = 100  # Reduced from 500 for trajectory collection
BATCH_SIZE = 4  # Small batch to manage memory
MAX_SEQ_LEN = 512  # Truncate very long sequences

# Even layers only: [0, 2, 4, ..., 30] = 16 layers
LAYERS_TO_COLLECT = list(range(0, 32, 2))

def load_models_config():
    """Load model configurations"""
    config_path = Path("configs/models.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_checkpoint(model_key, task_name):
    """Load checkpoint if exists"""
    checkpoint_file = CHECKPOINT_DIR / f"trajectory_{model_key}_{task_name}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed_samples": 0}

def save_checkpoint(model_key, task_name, completed_samples):
    """Save checkpoint"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"trajectory_{model_key}_{task_name}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump({"completed_samples": completed_samples}, f)

def collect_trajectories_for_model(model_key, model_config, task_name, task_data):
    """
    Collect full token trajectories for even layers only.

    Storage per file:
    - n_samples × max_seq_len × n_layers × d_model × 2 bytes (float16)
    - 100 × 512 × 16 × 4096 × 2 = ~6.7 GB per model/task
    - 4 models × 3 tasks = 12 files × 6.7 GB = ~80 GB total
    - With compression: ~56 GB
    """

    # Load checkpoint
    checkpoint = load_checkpoint(model_key, task_name)
    start_idx = checkpoint["completed_samples"]

    if start_idx >= N_SAMPLES:
        print(f"  ✓ Already completed ({N_SAMPLES} samples)")
        return

    # Output file
    output_file = OUTPUT_DIR / model_key / f"{task_name}_trajectories.h5"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize collector
    print(f"  Loading model {model_config['model_name']}...")
    collector = ActivationCollector(
        model_name=model_config['model_name'],
        layers_to_extract=LAYERS_TO_COLLECT,  # Even layers only
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Prepare output dataset
    if start_idx == 0:
        # Create new file
        with h5py.File(output_file, 'w') as f:
            # Pre-allocate datasets with compression
            f.create_dataset(
                'trajectories',
                shape=(N_SAMPLES, MAX_SEQ_LEN, len(LAYERS_TO_COLLECT), 4096),
                dtype='float16',
                compression='gzip',
                compression_opts=4
            )
            f.create_dataset(
                'sequence_lengths',
                shape=(N_SAMPLES,),
                dtype='int32'
            )
            f.create_dataset(
                'prompts',
                shape=(N_SAMPLES,),
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # Metadata
            f.attrs['model'] = model_key
            f.attrs['task'] = task_name
            f.attrs['n_samples'] = N_SAMPLES
            f.attrs['max_seq_len'] = MAX_SEQ_LEN
            f.attrs['layers'] = LAYERS_TO_COLLECT
            f.attrs['d_model'] = 4096
            f.attrs['collection_date'] = datetime.now().isoformat()

    # Collect trajectories
    print(f"  Collecting {N_SAMPLES - start_idx} trajectories (starting from {start_idx})...")

    with h5py.File(output_file, 'a') as f:
        for i in tqdm(range(start_idx, N_SAMPLES), desc=f"  {model_key}/{task_name}"):
            prompt = task_data[i]['prompt']

            # Get full trajectory
            trajectory = collector.get_trajectory(
                prompt=prompt,
                max_new_tokens=256,
                aggregation=None  # Return full sequence
            )

            # trajectory shape: (seq_len, n_layers, d_model)
            seq_len = trajectory.shape[0]

            # Truncate if needed
            if seq_len > MAX_SEQ_LEN:
                trajectory = trajectory[:MAX_SEQ_LEN]
                seq_len = MAX_SEQ_LEN

            # Pad if needed
            if seq_len < MAX_SEQ_LEN:
                padding = np.zeros(
                    (MAX_SEQ_LEN - seq_len, len(LAYERS_TO_COLLECT), 4096),
                    dtype=np.float16
                )
                trajectory = np.vstack([trajectory, padding])

            # Store
            f['trajectories'][i] = trajectory.astype(np.float16)
            f['sequence_lengths'][i] = seq_len
            f['prompts'][i] = prompt

            # Checkpoint every 10 samples
            if (i + 1) % 10 == 0:
                save_checkpoint(model_key, task_name, i + 1)

    # Final checkpoint
    save_checkpoint(model_key, task_name, N_SAMPLES)

    # Clean up
    del collector
    torch.cuda.empty_cache()

    # Report file size
    file_size_gb = output_file.stat().st_size / 1e9
    print(f"  ✓ Completed. File size: {file_size_gb:.2f} GB")

def main():
    """Main collection pipeline"""
    print("=" * 80)
    print("PHASE 2: Full Trajectory Collection (Half Layers)")
    print("=" * 80)
    print()
    print(f"Strategy: Even layers only [0, 2, 4, ..., 30] = {len(LAYERS_TO_COLLECT)} layers")
    print(f"Samples: {N_SAMPLES} per task")
    print(f"Max sequence length: {MAX_SEQ_LEN} tokens")
    print(f"Expected storage: ~56 GB total (with compression)")
    print()

    # Load configs
    models = load_models_config()

    # Task data preparation functions
    task_prep_fns = {
        'gsm8k': prepare_gsm8k,
        'humaneval': prepare_humaneval,
        'logiqa': prepare_logiqa
    }

    # Models to process
    model_keys = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']

    # Collect for each model/task
    for model_key in model_keys:
        if model_key not in models:
            print(f"⚠️  Model {model_key} not found in config, skipping")
            continue

        model_config = models[model_key]
        print(f"\n{'=' * 80}")
        print(f"Model: {model_key} ({model_config['model_name']})")
        print(f"{'=' * 80}")

        for task_name, prep_fn in task_prep_fns.items():
            print(f"\nTask: {task_name}")

            # Prepare task data
            task_data = prep_fn(n_samples=N_SAMPLES, split='test')

            # Collect trajectories
            try:
                collect_trajectories_for_model(
                    model_key=model_key,
                    model_config=model_config,
                    task_name=task_name,
                    task_data=task_data
                )
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Free GPU memory between models
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)

    # Summary
    total_size = sum(
        f.stat().st_size
        for f in OUTPUT_DIR.rglob("*.h5")
    ) / 1e9

    print(f"\nTotal storage used: {total_size:.2f} GB")
    print(f"Files: {len(list(OUTPUT_DIR.rglob('*.h5')))}")
    print(f"\nNext steps:")
    print(f"  1. Run path signature analysis: python scripts/analyze_trajectories.py")
    print(f"  2. Compare RL-Zero vs SFT trajectory shapes")

if __name__ == "__main__":
    main()
