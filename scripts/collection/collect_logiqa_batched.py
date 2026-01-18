#!/usr/bin/env python3
"""
LogiQA trajectory collection with simple HuggingFace batching.
No vLLM needed - just batches generation with transformers.

3-4x faster than sequential, fits on 24GB GPUs.

Usage:
    python collect_logiqa_batched.py olmo3_sft --batch-size 4
"""

import argparse
import sys
import os
import re
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
BATCH_SIZE = 4  # Safe for 24GB GPUs
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


class BatchedCollector:
    """Collect trajectories with batched generation."""

    def __init__(self, model_name: str, layers_to_collect: List[int], batch_size: int = 4):
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

        # Setup hooks for activation collection
        layer_outputs = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # Store batch: (batch_size, seq_len, d_model)
                if layer_idx not in layer_outputs:
                    layer_outputs[layer_idx] = []
                layer_outputs[layer_idx].append(hidden.detach().cpu())
            return hook

        handles = []
        for layer_idx in self.layers_to_collect:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            # Generate batch
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

            # Extract trajectories from collected activations
            # Note: We collected activations during generation, but they're scattered
            # For simplicity, run one more forward pass to get clean trajectories
            trajectories = []

            for i in range(batch_size):
                # Tokenize prompt + output for this sample
                full_text = prompts[i] + outputs[i]
                sample_inputs = self.tokenizer(
                    full_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_seq_len,
                ).to(self.model.device)

                # Clear layer outputs
                layer_outputs.clear()

                # Forward pass
                with torch.no_grad():
                    _ = self.model(**sample_inputs)

                # Build trajectory: (seq_len, n_layers, d_model)
                seq_len = sample_inputs['input_ids'].shape[1]
                trajectory = np.zeros((max_seq_len, self.n_layers, self.d_model), dtype=np.float16)

                for j, layer_idx in enumerate(self.layers_to_collect):
                    if layer_idx in layer_outputs and len(layer_outputs[layer_idx]) > 0:
                        # Get last stored activation for this layer
                        act = layer_outputs[layer_idx][-1][0].numpy()  # [seq_len, d_model]
                        actual_len = min(seq_len, max_seq_len)
                        trajectory[:actual_len, j, :] = act[:actual_len].astype(np.float16)

                trajectories.append(trajectory)

        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return outputs, trajectories


def collect_logiqa_batched(
    model_key: str,
    model_config: Dict,
    task_data: List,
    batch_size: int = 4,
):
    """Collect LogiQA trajectories with batched generation."""

    output_dir = OUTPUT_DIR / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "logiqa_trajectories_batched.h5"

    print(f"\n{'='*80}")
    print(f"Batched LogiQA Collection: {model_key}")
    print(f"Batch size: {batch_size}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"{'='*80}\n")

    print(f"  Output: {output_file}")

    # Initialize collector
    collector = BatchedCollector(
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
            shape=(N_SAMPLES, MAX_SEQ_LEN, len(LAYERS_TO_COLLECT), collector.d_model),
            dtype='float16',
            compression='gzip',
            compression_opts=4
        )
        f.create_dataset('sequence_lengths', shape=(N_SAMPLES,), dtype='int32')
        f.create_dataset('is_correct', shape=(N_SAMPLES,), dtype='bool')
        f.create_dataset('prompts', shape=(N_SAMPLES,), dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset('model_outputs', shape=(N_SAMPLES,), dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset('ground_truth', shape=(N_SAMPLES,), dtype=h5py.string_dtype(encoding='utf-8'))

        # Process in batches
        pbar = tqdm(
            range(0, N_SAMPLES, batch_size),
            desc=f"  {model_key}/logiqa",
        )

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, N_SAMPLES)
            actual_batch_size = batch_end - batch_start

            # Prepare batch
            batch_prompts = [task_data[i][0] for i in range(batch_start, batch_end)]
            batch_answers = [task_data[i][1] for i in range(batch_start, batch_end)]

            # Generate batch
            outputs, trajectories = collector.generate_batch(
                batch_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                max_seq_len=MAX_SEQ_LEN,
            )

            # Store results
            for i in range(actual_batch_size):
                idx = batch_start + i

                is_correct = check_logiqa_correct(outputs[i], batch_answers[i])
                seq_len = MAX_SEQ_LEN  # Using fixed for simplicity

                if is_correct:
                    n_correct += 1
                else:
                    n_incorrect += 1

                # Write to HDF5
                f['trajectories'][idx] = trajectories[i]
                f['sequence_lengths'][idx] = seq_len
                f['is_correct'][idx] = is_correct
                f['prompts'][idx] = batch_prompts[i]
                f['model_outputs'][idx] = outputs[i][:50000]
                f['ground_truth'][idx] = str(batch_answers[i])

                # Update progress
                pbar.set_postfix({
                    'correct': n_correct,
                    'incorrect': n_incorrect,
                    'out_len': len(outputs[i]),
                })

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Correct: {n_correct}/{N_SAMPLES} ({n_correct/N_SAMPLES*100:.1f}%)")
    print(f"Incorrect: {n_incorrect}/{N_SAMPLES} ({n_incorrect/N_SAMPLES*100:.1f}%)")
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
    parser = argparse.ArgumentParser(description='Batched LogiQA collection')
    parser.add_argument('model_key', type=str, help='Model key (e.g., olmo3_sft)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of samples')

    args = parser.parse_args()

    # Load config
    model_config = load_model_config(args.model_key)

    # Load data
    print("Preparing LogiQA data...")
    task_data = prepare_logiqa(n_samples=args.num_samples, split='test')
    print(f"✓ Loaded {len(task_data)} LogiQA samples")

    # Run collection
    collect_logiqa_batched(
        model_key=args.model_key,
        model_config=model_config,
        task_data=task_data,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
