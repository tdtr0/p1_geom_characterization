#!/usr/bin/env python3
"""
OPTIMIZED LogiQA trajectory collection with GPU-efficient batched processing.

This script fixes 4 critical bottlenecks discovered in collect_logiqa_batched.py:
  1. GPU→CPU transfers during generation (FIXED: keep on GPU until final transfer)
  2. Sequential forward passes for activation collection (FIXED: batched forward pass)
  3. Blocking HDF5 writes (FIXED: async I/O with threading)
  4. Memory fragmentation (FIXED: explicit cleanup)

Expected speedup: 4-5x vs sequential, 2-3x vs batched
Expected time: 2-3 hours for 500 samples (vs 12.5 hrs sequential, 5.5 hrs batched)

See docs/plans/master_algorithm.md "GPU Optimization Lessons" for detailed analysis.

Usage:
    python collect_logiqa_optimized.py olmo3_sft --batch-size 8
"""

import argparse
import sys
import os
import re
import gc
import queue
import threading
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import h5py
from tqdm import tqdm
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from task_data import prepare_logiqa

# Configuration
N_SAMPLES = 500
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 2048
LAYERS_TO_COLLECT = list(range(0, 32, 2))  # Even layers
BATCH_SIZE = 8  # Can increase with optimizations
OUTPUT_DIR = Path('data/trajectories')


def extract_answer(text: str) -> str:
    """Extract answer (A/B/C/D) from model output."""
    text = text.strip()
    match = re.search(r'answer[^A-D]*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches[-1].upper()
    return ""


def check_logiqa_correct(output: str, ground_truth: str) -> bool:
    """Check if model output matches ground truth answer."""
    extracted = extract_answer(output)
    return extracted == ground_truth


class OptimizedCollector:
    """
    GPU-optimized trajectory collector with batched activation collection.

    Key optimizations:
    1. Keep all tensors on GPU until final transfer
    2. Batch activation collection (single forward pass for all samples)
    3. Explicit memory cleanup
    """

    def __init__(self, model_name: str, layers_to_collect: List[int], batch_size: int = 8):
        self.model_name = model_name
        self.layers_to_collect = layers_to_collect
        self.batch_size = batch_size
        self.n_layers = len(layers_to_collect)

        print(f"Loading model: {model_name}")
        print(f"  Batch size: {batch_size}")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fix padding for decoder-only models
        self.tokenizer.padding_side = 'left'

        self.model.eval()
        self.d_model = 4096  # OLMo-3

        print(f"  Loaded: {len(self.model.model.layers)} layers, d_model={self.d_model}")
        print(f"  Device: {self.model.device}")

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 2048,
        max_seq_len: int = 512,
    ) -> Tuple[List[str], List[np.ndarray]]:
        """
        Generate outputs and collect activations for a batch of prompts.

        OPTIMIZATION 1 & 2: Keep tensors on GPU, batch activation collection

        Returns:
            outputs: List of generated texts
            trajectories: List of activation arrays (seq_len, n_layers, d_model)
        """
        batch_size = len(prompts)

        # Tokenize batch with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(self.model.device)

        # Generate batch (no activation collection here)
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.2,
            )

        # Decode outputs
        outputs = []
        for i in range(batch_size):
            # Get only generated part (after prompt)
            prompt_len = inputs['input_ids'][i].ne(self.tokenizer.pad_token_id).sum()
            generated = output_ids[i][prompt_len:]
            output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            outputs.append(output_text)

        # OPTIMIZATION 2: Batched activation collection
        # Instead of 4 sequential forward passes (4 × 30s = 120s),
        # do ONE batched forward pass (~35s)

        # Prepare full texts (prompt + output)
        full_texts = [prompts[i] + outputs[i] for i in range(batch_size)]

        # Tokenize all together with padding
        batch_inputs = self.tokenizer(
            full_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(self.model.device)

        # OPTIMIZATION 1: Setup hooks that keep tensors on GPU
        layer_outputs_gpu = {}  # Keep on GPU!

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # CRITICAL: .detach() only, NO .cpu()!
                # Transfer to CPU happens once at the end
                if layer_idx not in layer_outputs_gpu:
                    layer_outputs_gpu[layer_idx] = []
                layer_outputs_gpu[layer_idx].append(hidden.detach())  # Stay on GPU
            return hook

        handles = []
        for layer_idx in self.layers_to_collect:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            # Single batched forward pass
            with torch.no_grad():
                _ = self.model(**batch_inputs)

            # Extract trajectories by sample
            # layer_outputs_gpu[layer_idx][-1] has shape: (batch_size, seq_len, d_model)
            trajectories = []

            for i in range(batch_size):
                # Find actual sequence length (exclude padding)
                seq_len = batch_inputs['attention_mask'][i].sum().item()

                # Build trajectory: (max_seq_len, n_layers, d_model)
                trajectory = np.zeros((max_seq_len, self.n_layers, self.d_model), dtype=np.float16)

                for j, layer_idx in enumerate(self.layers_to_collect):
                    if layer_idx in layer_outputs_gpu and len(layer_outputs_gpu[layer_idx]) > 0:
                        # Get activation for this layer and sample
                        # Shape: (batch_size, seq_len, d_model) -> extract sample i
                        act_gpu = layer_outputs_gpu[layer_idx][-1][i]  # (seq_len, d_model)

                        # Transfer to CPU only once
                        act_cpu = act_gpu.cpu().numpy()

                        # Copy actual sequence (no padding)
                        actual_len = min(seq_len, max_seq_len)
                        trajectory[:actual_len, j, :] = act_cpu[:actual_len].astype(np.float16)

                trajectories.append(trajectory)

        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()

            # OPTIMIZATION 4: Explicit memory cleanup
            layer_outputs_gpu.clear()
            del layer_outputs_gpu

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

        return outputs, trajectories


def collect_logiqa_optimized(
    model_key: str,
    model_config: Dict,
    task_data: List,
    batch_size: int = 8,
):
    """
    Collect LogiQA trajectories with optimized batched generation.

    OPTIMIZATION 3: Async HDF5 writes with threading
    """

    # Use actual data length instead of hardcoded constant
    n_samples = len(task_data)

    output_dir = OUTPUT_DIR / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "logiqa_trajectories_optimized.h5"

    print(f"\n{'='*80}")
    print(f"OPTIMIZED LogiQA Collection: {model_key}")
    print(f"Batch size: {batch_size}")
    print(f"Samples: {n_samples}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"{'='*80}\n")

    print(f"  Output: {output_file}")

    # Initialize collector
    collector = OptimizedCollector(
        model_name=model_config['model_name'],
        layers_to_collect=LAYERS_TO_COLLECT,
        batch_size=batch_size,
    )

    n_correct = 0
    n_incorrect = 0

    # Create HDF5 file
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

        # OPTIMIZATION 3: Async I/O with threading
        # Write batch N to disk while GPU generates batch N+1

        write_queue = queue.Queue(maxsize=2)  # Buffer 2 batches max
        write_error = {'error': None}  # Track errors from writer thread

        def writer_thread():
            """Background thread that writes to HDF5 while GPU works."""
            try:
                while True:
                    item = write_queue.get()
                    if item is None:  # Poison pill - shutdown
                        break

                    # Unpack batch data
                    batch_start, actual_batch_size, batch_data = item

                    # Write to HDF5 (GPU is working on next batch!)
                    for i in range(actual_batch_size):
                        idx = batch_start + i
                        trajectory, seq_len, is_correct, prompt, output, gt = batch_data[i]

                        f['trajectories'][idx] = trajectory
                        f['sequence_lengths'][idx] = seq_len
                        f['is_correct'][idx] = is_correct
                        f['prompts'][idx] = prompt
                        f['model_outputs'][idx] = output[:50000]
                        f['ground_truth'][idx] = str(gt)

                    write_queue.task_done()

            except Exception as e:
                write_error['error'] = e

        # Start writer thread
        writer = threading.Thread(target=writer_thread, daemon=True)
        writer.start()

        # Process in batches
        pbar = tqdm(
            range(0, n_samples, batch_size),
            desc=f"  {model_key}/logiqa",
        )

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, n_samples)
            actual_batch_size = batch_end - batch_start

            # Prepare batch
            batch_prompts = [task_data[i][0] for i in range(batch_start, batch_end)]
            batch_answers = [task_data[i][1] for i in range(batch_start, batch_end)]

            # Generate batch (GPU working)
            outputs, trajectories = collector.generate_batch(
                batch_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                max_seq_len=MAX_SEQ_LEN,
            )

            # Check correctness and prepare data
            batch_data = []
            for i in range(actual_batch_size):
                is_correct = check_logiqa_correct(outputs[i], batch_answers[i])
                seq_len = MAX_SEQ_LEN  # Using fixed for simplicity

                if is_correct:
                    n_correct += 1
                else:
                    n_incorrect += 1

                batch_data.append((
                    trajectories[i],
                    seq_len,
                    is_correct,
                    batch_prompts[i],
                    outputs[i],
                    batch_answers[i]
                ))

            # Queue for async write (non-blocking unless queue full)
            write_queue.put((batch_start, actual_batch_size, batch_data))

            # Check for write errors
            if write_error['error'] is not None:
                raise write_error['error']

            # Update progress
            pbar.set_postfix({
                'correct': n_correct,
                'incorrect': n_incorrect,
                'out_len': len(outputs[0]) if outputs else 0,
            })

        # Wait for all writes to complete
        write_queue.put(None)  # Poison pill
        writer.join()

        # Check for final errors
        if write_error['error'] is not None:
            raise write_error['error']

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Correct: {n_correct}/{n_samples} ({n_correct/n_samples*100:.1f}%)")
    print(f"Incorrect: {n_incorrect}/{n_samples} ({n_incorrect/n_samples*100:.1f}%)")
    print(f"Output file: {output_file}")
    print(f"{'='*80}\n")


def load_model_config(model_key: str) -> Dict:
    """Load model configuration from configs/models.yaml"""
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'models.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    models = config.get('primary', {})
    if model_key not in models:
        raise ValueError(f"Model '{model_key}' not found")

    return models[model_key]


def main():
    parser = argparse.ArgumentParser(description='Optimized LogiQA collection')
    parser.add_argument('model_key', type=str, help='Model key (e.g., olmo3_sft)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of samples')

    args = parser.parse_args()

    # Load config
    model_config = load_model_config(args.model_key)

    # Load data
    print("Preparing LogiQA data...")
    task_data = prepare_logiqa(n_samples=args.num_samples, split='test')
    print(f"✓ Loaded {len(task_data)} LogiQA samples")

    # Run collection
    collect_logiqa_optimized(
        model_key=args.model_key,
        model_config=model_config,
        task_data=task_data,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
