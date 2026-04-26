#!/usr/bin/env python3
"""Collect LogiQA trajectories for a single model on a specific GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./src python scripts/collection/collect_single_logiqa.py olmo3_sft
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./src python scripts/collection/collect_single_logiqa.py olmo3_rl_zero
"""

import sys
import os

# Add src and scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import yaml

from task_data import prepare_logiqa

# Configuration - INCREASED TOKEN LIMIT
N_SAMPLES = 500
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 2048  # Sufficient for most outputs; repetition_penalty prevents loops
LAYERS_TO_COLLECT = list(range(0, 32, 2))  # Even layers: [0, 2, 4, ..., 30]
OUTPUT_DIR = Path("data/trajectories")

# Try to use Flash Attention
USE_FLASH_ATTN = True


def load_models_config():
    """Load model configurations."""
    config_path = Path("configs/models.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get('primary', {})


def extract_logiqa_answer(model_output: str) -> str:
    """Extract A/B/C/D answer from model output."""
    import re
    patterns = [
        r'(?:answer|choice|option)\s*(?:is|:)?\s*[:\s]*([A-D])',
        r'\b([A-D])\s*(?:is correct|is the answer)',
        r'(?:^|\n)\s*([A-D])\s*[.)\s]',
        r'\\boxed\{([A-D])\}',
        r'\*\*([A-D])\*\*',
    ]
    for pattern in patterns:
        match = re.search(pattern, model_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    matches = re.findall(r'\b([A-D])\b', model_output)
    if matches:
        return matches[-1].upper()
    return ""


def check_logiqa_correct(model_output: str, ground_truth: str) -> bool:
    """Check if LogiQA answer is correct."""
    try:
        gt_idx = int(ground_truth)
        gt_letter = chr(65 + gt_idx)
    except (ValueError, TypeError):
        gt_letter = str(ground_truth).upper()
    model_answer = extract_logiqa_answer(model_output)
    return model_answer == gt_letter


class SingleGPUCollector:
    """Collects trajectories on a single GPU."""

    def __init__(self, model_name: str, layers_to_collect: list):
        self.model_name = model_name
        self.layers_to_collect = layers_to_collect

        print(f"Loading model: {model_name}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load on single GPU
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            print(f"  GPU 0: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

            try:
                if USE_FLASH_ATTN:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                    ).cuda()
                    print("  Using Flash Attention 2")
                else:
                    raise ImportError("Flash Attention disabled")
            except (ImportError, ValueError) as e:
                print(f"  Flash Attention not available: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                ).cuda()
        else:
            raise RuntimeError("CUDA not available")

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size

        print(f"  Loaded: {self.n_layers} layers, d_model={self.d_model}")
        print(f"  Device: {self.device}")

    def generate_with_trajectory(self, prompt: str, max_new_tokens: int, max_seq_len: int) -> tuple:
        """Generate response and collect trajectory."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[1]

        layer_outputs = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                layer_outputs[layer_idx] = hidden.detach().cpu()
            return hook

        handles = []
        for layer_idx in self.layers_to_collect:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            with torch.no_grad():
                _ = self.model(**inputs)

                prompt_trajectory = []
                for layer_idx in self.layers_to_collect:
                    act = layer_outputs[layer_idx][0].numpy().astype(np.float16)
                    prompt_trajectory.append(act)

                trajectory = np.stack(prompt_trajectory, axis=0)
                trajectory = np.transpose(trajectory, (1, 0, 2))

                output_ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.2,  # Prevent repetition loops
                )

                generated_text = self.tokenizer.decode(
                    output_ids[0][prompt_len:],
                    skip_special_tokens=True
                )

        finally:
            for handle in handles:
                handle.remove()

        return generated_text, trajectory

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect_logiqa(model_key: str, model_config: dict, task_data: list):
    """Collect LogiQA trajectories for a single model."""

    output_dir = OUTPUT_DIR / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "logiqa_trajectories.h5"

    if output_file.exists():
        print(f"  {output_file} already exists, skipping")
        return

    print(f"  Output: {output_file}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")

    collector = SingleGPUCollector(
        model_name=model_config['model_name'],
        layers_to_collect=LAYERS_TO_COLLECT
    )

    n_correct = 0
    n_incorrect = 0

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

        pbar = tqdm(range(N_SAMPLES), desc=f"  {model_key}/logiqa")

        for i in pbar:
            prompt, answer, metadata = task_data[i]

            try:
                model_output, trajectory = collector.generate_with_trajectory(
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    max_seq_len=MAX_SEQ_LEN
                )

                is_correct = check_logiqa_correct(model_output, answer)
                seq_len = trajectory.shape[0]

                # Log output length periodically
                if i < 5 or i % 50 == 0:
                    print(f"\n  Sample {i}: output_len={len(model_output)}, seq_len={seq_len}, correct={is_correct}")
                    if len(model_output) > 500:
                        print(f"    First 200 chars: {model_output[:200]}...")
                        print(f"    Last 200 chars: ...{model_output[-200:]}")

                if seq_len > MAX_SEQ_LEN:
                    trajectory = trajectory[:MAX_SEQ_LEN]
                    seq_len = MAX_SEQ_LEN
                elif seq_len < MAX_SEQ_LEN:
                    padding = np.zeros(
                        (MAX_SEQ_LEN - seq_len, len(LAYERS_TO_COLLECT), collector.d_model),
                        dtype=np.float32
                    )
                    trajectory = np.vstack([trajectory, padding])

                f['trajectories'][i] = trajectory.astype(np.float16)
                f['sequence_lengths'][i] = seq_len
                f['is_correct'][i] = is_correct
                f['prompts'][i] = prompt
                f['model_outputs'][i] = model_output[:50000]  # Allow longer outputs
                f['ground_truth'][i] = str(answer)[:5000]

                if is_correct:
                    n_correct += 1
                else:
                    n_incorrect += 1

                pbar.set_postfix(correct=n_correct, incorrect=n_incorrect, out_len=len(model_output))

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
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

    del collector
    torch.cuda.empty_cache()

    file_size_gb = output_file.stat().st_size / 1e9
    print(f"\n  Completed: {n_correct} correct, {n_incorrect} incorrect")
    print(f"  File size: {file_size_gb:.2f} GB")


def main():
    if len(sys.argv) < 2:
        print("Usage: CUDA_VISIBLE_DEVICES=X python collect_single_logiqa.py <model_key>")
        print("  Models: olmo3_sft, olmo3_rl_zero, olmo3_think")
        sys.exit(1)

    model_key = sys.argv[1]

    print("=" * 80)
    print(f"Collecting LogiQA for: {model_key}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print("=" * 80)

    models = load_models_config()

    if model_key not in models:
        print(f"Model {model_key} not found in config")
        sys.exit(1)

    model_config = models[model_key]

    print("Preparing LogiQA data...")
    task_data = prepare_logiqa(n_samples=N_SAMPLES, split='test')
    print(f"Loaded {len(task_data)} samples\n")

    collect_logiqa(model_key, model_config, task_data)

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
