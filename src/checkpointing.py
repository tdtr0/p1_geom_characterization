"""
Checkpointing Module

Handles checkpointing and resumption for long-running collection jobs.
Enables recovery from OOM, crashes, or interruptions.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import time


class CheckpointManager:
    """
    Manages checkpoints for activation collection.

    Checkpoint structure:
    {
        "model": str,
        "task": str,
        "samples_completed": int,
        "total_samples": int,
        "last_sample_idx": int,
        "output_file": str,
        "timestamp": float,
        "status": "in_progress" | "completed" | "failed"
    }
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, model: str, task: str) -> Path:
        """Get checkpoint file path for a model/task combination."""
        safe_model = model.replace("/", "_").replace("-", "_")
        return self.checkpoint_dir / f"{safe_model}_{task}.json"

    def save_checkpoint(
        self,
        model: str,
        task: str,
        samples_completed: int,
        total_samples: int,
        last_sample_idx: int,
        output_file: str,
        status: str = "in_progress",
        **extra_info
    ):
        """Save checkpoint to disk."""
        checkpoint = {
            "model": model,
            "task": task,
            "samples_completed": samples_completed,
            "total_samples": total_samples,
            "last_sample_idx": last_sample_idx,
            "output_file": output_file,
            "timestamp": time.time(),
            "status": status,
            **extra_info
        }

        checkpoint_path = self.get_checkpoint_path(model, task)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        return checkpoint_path

    def load_checkpoint(self, model: str, task: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint from disk if it exists."""
        checkpoint_path = self.get_checkpoint_path(model, task)

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        return checkpoint

    def checkpoint_exists(self, model: str, task: str) -> bool:
        """Check if checkpoint exists."""
        return self.get_checkpoint_path(model, task).exists()

    def is_completed(self, model: str, task: str) -> bool:
        """Check if task is marked as completed."""
        checkpoint = self.load_checkpoint(model, task)
        return checkpoint is not None and checkpoint.get("status") == "completed"

    def delete_checkpoint(self, model: str, task: str):
        """Delete checkpoint (e.g., after successful completion)."""
        checkpoint_path = self.get_checkpoint_path(model, task)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    def list_checkpoints(self) -> list:
        """List all checkpoints."""
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                checkpoints.append(checkpoint)
        return checkpoints

    def get_resume_info(self, model: str, task: str) -> Optional[Dict[str, Any]]:
        """
        Get information needed to resume a job.

        Returns:
            Dict with:
            - start_idx: Index to start from
            - completed: Number of samples already done
            - output_file: Path to partial output file
        """
        checkpoint = self.load_checkpoint(model, task)

        if checkpoint is None:
            return None

        if checkpoint["status"] == "completed":
            print(f"Task {model}/{task} already completed")
            return None

        return {
            "start_idx": checkpoint["last_sample_idx"] + 1,
            "completed": checkpoint["samples_completed"],
            "output_file": checkpoint["output_file"],
            "checkpoint": checkpoint
        }


def create_partial_output_path(base_path: str, part_num: int = 0) -> str:
    """Create path for partial output file."""
    base = Path(base_path)
    if part_num == 0:
        return str(base)
    else:
        return str(base.parent / f"{base.stem}_part{part_num}{base.suffix}")


def merge_h5_files(input_files: list, output_file: str):
    """
    Merge multiple HDF5 files into one.

    Concatenates activations along the sample dimension.
    """
    import h5py
    import numpy as np

    if len(input_files) == 0:
        raise ValueError("No input files to merge")

    if len(input_files) == 1:
        # Just rename
        Path(input_files[0]).rename(output_file)
        return

    # Read all files and concatenate
    all_activations = {}
    all_metadata = {}

    for i, input_file in enumerate(input_files):
        with h5py.File(input_file, 'r') as f:
            # Load activations
            for key in f.keys():
                if key != 'metadata':
                    if key not in all_activations:
                        all_activations[key] = []
                    all_activations[key].append(f[key][:])

            # Load metadata (just use last file's metadata)
            if 'metadata' in f:
                meta_group = f['metadata']
                for key in meta_group.attrs:
                    all_metadata[key] = meta_group.attrs[key]
                for key in meta_group.keys():
                    all_metadata[key] = meta_group[key][:]

    # Concatenate and save
    with h5py.File(output_file, 'w') as f:
        for key, arrays in all_activations.items():
            concatenated = np.concatenate(arrays, axis=0)
            f.create_dataset(key, data=concatenated, compression='gzip')

        # Save metadata
        if all_metadata:
            meta_group = f.create_group('metadata')
            for key, value in all_metadata.items():
                if isinstance(value, str):
                    meta_group.attrs[key] = value
                elif isinstance(value, (int, float, bool)):
                    meta_group.attrs[key] = value
                else:
                    meta_group.create_dataset(key, data=value)

    # Clean up partial files
    for input_file in input_files:
        Path(input_file).unlink()

    print(f"✓ Merged {len(input_files)} files into {output_file}")
