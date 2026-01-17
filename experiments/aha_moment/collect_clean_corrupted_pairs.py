#!/usr/bin/env python3
"""
Collect clean/corrupted trajectory pairs for Wynroe-style error detection.

This script:
1. Generates CoT traces from RL-Zero model on GSM8K
2. Parses <<expr=result>> annotations to find numbers
3. Creates corrupted versions (changes last number)
4. Runs both clean and corrupted through all 4 OLMo models
5. Collects trajectories at error token positions

Usage:
    python collect_clean_corrupted_pairs.py \
        --n_problems 200 \
        --models base,sft,rl_zero,think \
        --output data/wynroe_replication/

GPU Requirements:
    - 16GB+ VRAM (24GB recommended for safety)
    - Estimated time: 3-6 hours for 200 problems × 4 models
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Model configurations
MODEL_CONFIGS = {
    'base': 'allenai/OLMo-3-1B-0125',  # Using 1B for faster testing
    'base_7b': 'allenai/OLMo-3-7B-0125',
    'sft': 'allenai/OLMo-3-7B-Think-SFT',
    'rl_zero': 'allenai/OLMo-3-7B-RL-Zero-General',
    'think': 'allenai/OLMo-3-7B-Think',
}

# For generation, we use RL-Zero (produces <<expr=result>> annotations)
GENERATION_MODEL = 'rl_zero'


# ============================================================================
# GSM8K Data Loading
# ============================================================================

def load_gsm8k(n_problems: int, seed: int = 42) -> list[dict]:
    """Load GSM8K problems."""
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    # Shuffle and select
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))[:n_problems]

    problems = []
    for idx in indices:
        item = dataset[int(idx)]
        # Parse answer
        answer_text = item['answer']
        # Extract #### number
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer_text)
        if match:
            answer_num = match.group(1).replace(',', '')
        else:
            answer_num = None

        problems.append({
            'id': f'gsm8k_{idx}',
            'question': item['question'],
            'answer': answer_num,
            'full_solution': answer_text,
        })

    return problems


# ============================================================================
# CoT Generation and Annotation Parsing
# ============================================================================

def generate_cot(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 512,
    device: str = 'cuda',
) -> str:
    """Generate CoT response for a question."""
    # Format prompt (RL-Zero style)
    prompt = f"Question: {question}\nLet me solve this step by step.\n"

    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt
    response = generated[len(prompt):].strip()
    return response


def parse_annotations(text: str) -> list[dict]:
    """
    Parse <<expr=result>> annotations in CoT trace.

    Returns list of dicts with:
        - expr: the expression (e.g., "5*3")
        - result: the result (e.g., "15")
        - start: char position of start of annotation
        - end: char position of end of annotation
        - result_start: char position of result within annotation
    """
    annotations = []
    # Pattern: <<expr=result>>
    pattern = r'<<([^=]+)=([^>]+)>>'

    for match in re.finditer(pattern, text):
        expr = match.group(1)
        result = match.group(2)

        annotations.append({
            'expr': expr,
            'result': result,
            'start': match.start(),
            'end': match.end(),
            'result_start': match.start() + len('<<') + len(expr) + len('='),
            'full_match': match.group(0),
        })

    return annotations


def create_corrupted_trace(text: str, annotations: list[dict]) -> tuple[str, int, str]:
    """
    Create corrupted version by changing the last annotation's result.

    Returns:
        - corrupted_text: text with error introduced
        - error_char_pos: character position of error
        - original_value: the original correct value
    """
    if not annotations:
        return None, None, None

    # Get last annotation (most likely to cause error)
    last_ann = annotations[-1]

    # Parse result as number and corrupt it
    try:
        orig_num = float(last_ann['result'])
        # Add small offset to corrupt
        if orig_num == 0:
            corrupt_num = 1
        else:
            corrupt_num = orig_num + 1 if orig_num > 0 else orig_num - 1

        corrupt_str = str(int(corrupt_num)) if corrupt_num == int(corrupt_num) else str(corrupt_num)
    except ValueError:
        # Not a number, can't corrupt
        return None, None, None

    # Create corrupted text
    prefix = text[:last_ann['result_start']]
    suffix = text[last_ann['result_start'] + len(last_ann['result']):]

    # Also need to update the result after >> if it appears
    # Pattern: <<expr=result>>result
    suffix_pattern = r'^>>' + re.escape(last_ann['result'])
    suffix_match = re.match(suffix_pattern, suffix)
    if suffix_match:
        suffix = f'>>{corrupt_str}' + suffix[suffix_match.end():]

    corrupted_text = prefix + corrupt_str + suffix

    # Error is at the result position
    error_char_pos = last_ann['result_start']

    return corrupted_text, error_char_pos, last_ann['result']


# ============================================================================
# Activation Collection
# ============================================================================

class ActivationCollector:
    """Collect activations at specific token positions."""

    def __init__(self, model, layers_to_collect: list[int] = None):
        self.model = model
        self.activations = {}
        self.hooks = []

        # Determine layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            self.n_layers = len(model.transformer.h)
        else:
            raise ValueError("Cannot determine model architecture")

        if layers_to_collect is None:
            # Collect even layers (like Phase 2)
            self.layers_to_collect = list(range(0, self.n_layers, 2))
        else:
            self.layers_to_collect = layers_to_collect

    def _get_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.activations[layer_idx] = hidden.detach().cpu().float()
        return hook

    def register_hooks(self):
        self.activations = {}
        self.hooks = []

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            layers = self.model.transformer.h

        for layer_idx in self.layers_to_collect:
            hook = layers[layer_idx].register_forward_hook(self._get_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations_at_position(self, token_idx: int) -> np.ndarray:
        """
        Get activations at specific token position.
        Returns: (n_layers, hidden_dim)
        """
        acts = []
        for layer_idx in sorted(self.activations.keys()):
            # Shape: (batch=1, seq_len, hidden_dim)
            act = self.activations[layer_idx][0, token_idx, :].numpy()
            acts.append(act)
        return np.stack(acts, axis=0)  # (n_layers, hidden_dim)


def collect_trajectory_at_position(
    model,
    tokenizer,
    text: str,
    target_token_idx: int,
    collector: ActivationCollector,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Run forward pass and collect activations at target token.

    Args:
        model: The model
        tokenizer: Tokenizer
        text: Input text
        target_token_idx: Token index to collect (from alignment)
        collector: Activation collector
        device: Device

    Returns:
        Activations at target position: (n_layers, hidden_dim)
    """
    inputs = tokenizer(text, return_tensors='pt').to(device)

    collector.register_hooks()
    try:
        with torch.no_grad():
            model(**inputs)
        activations = collector.get_activations_at_position(target_token_idx)
    finally:
        collector.remove_hooks()

    return activations


def align_char_to_token(text: str, char_pos: int, tokenizer) -> int:
    """Convert character position to token index."""
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    for idx, (start, end) in enumerate(encoding.offset_mapping):
        if start <= char_pos < end:
            return idx

    return len(encoding.offset_mapping) - 1


# ============================================================================
# Main Collection Pipeline
# ============================================================================

def collect_wynroe_data(
    n_problems: int,
    model_names: list[str],
    output_dir: str,
    max_cot_tokens: int = 512,
    device: str = 'cuda',
    seed: int = 42,
):
    """
    Main collection pipeline for Wynroe-style error detection.

    Steps:
    1. Load GSM8K problems
    2. Generate CoT traces from RL-Zero
    3. Create clean/corrupted pairs
    4. Run through all models
    5. Save trajectories
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load problems
    print(f"\n=== Step 1: Loading {n_problems} GSM8K problems ===")
    problems = load_gsm8k(n_problems, seed)
    print(f"Loaded {len(problems)} problems")

    # Step 2: Generate CoT traces
    print(f"\n=== Step 2: Generating CoT traces with {GENERATION_MODEL} ===")

    gen_model_name = MODEL_CONFIGS[GENERATION_MODEL]
    print(f"Loading {gen_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    gen_model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate and parse CoT traces
    pairs = []
    for problem in tqdm(problems, desc="Generating CoT"):
        cot = generate_cot(gen_model, tokenizer, problem['question'], max_cot_tokens, device)
        annotations = parse_annotations(cot)

        if not annotations:
            continue

        corrupted, error_char, original = create_corrupted_trace(cot, annotations)
        if corrupted is None:
            continue

        # Align error position to token
        error_token_idx = align_char_to_token(corrupted, error_char, tokenizer)

        pairs.append({
            'problem': problem,
            'clean_trace': cot,
            'corrupted_trace': corrupted,
            'error_token_idx': error_token_idx,
            'error_char_pos': error_char,
            'original_value': original,
            'n_annotations': len(annotations),
        })

    print(f"Created {len(pairs)} clean/corrupted pairs")

    # Free generation model
    del gen_model
    torch.cuda.empty_cache()

    # Step 3: Collect trajectories from all models
    print(f"\n=== Step 3: Collecting trajectories from {len(model_names)} models ===")

    # Prepare output file
    output_file = Path(output_dir) / 'wynroe_trajectories.h5'

    with h5py.File(output_file, 'w') as f:
        # Metadata
        f.attrs['n_pairs'] = len(pairs)
        f.attrs['models'] = model_names
        f.attrs['generation_model'] = GENERATION_MODEL
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['seed'] = seed

        # Save pair metadata
        meta_group = f.create_group('metadata')
        for i, pair in enumerate(pairs):
            meta_group.create_dataset(f'pair_{i}', data=json.dumps({
                'problem_id': pair['problem']['id'],
                'question': pair['problem']['question'],
                'answer': pair['problem']['answer'],
                'error_token_idx': pair['error_token_idx'],
                'error_char_pos': pair['error_char_pos'],
                'original_value': pair['original_value'],
                'n_annotations': pair['n_annotations'],
            }))

        # Collect from each model
        for model_key in model_names:
            model_name = MODEL_CONFIGS.get(model_key) or MODEL_CONFIGS.get(f'{model_key}_7b')
            if model_name is None:
                print(f"Warning: Unknown model {model_key}, skipping")
                continue

            print(f"\nCollecting from: {model_name}")

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
            model.eval()

            collector = ActivationCollector(model)

            # Create groups for this model
            model_group = f.create_group(model_key)
            clean_group = model_group.create_group('clean')
            corrupt_group = model_group.create_group('corrupted')

            for i, pair in enumerate(tqdm(pairs, desc=f"  {model_key}")):
                try:
                    # Collect clean trajectory at error position
                    clean_acts = collect_trajectory_at_position(
                        model, tokenizer, pair['clean_trace'],
                        pair['error_token_idx'], collector, device
                    )

                    # Collect corrupted trajectory at error position
                    corrupt_acts = collect_trajectory_at_position(
                        model, tokenizer, pair['corrupted_trace'],
                        pair['error_token_idx'], collector, device
                    )

                    clean_group.create_dataset(
                        f'pair_{i}',
                        data=clean_acts.astype(np.float16),
                        compression='gzip',
                    )
                    corrupt_group.create_dataset(
                        f'pair_{i}',
                        data=corrupt_acts.astype(np.float16),
                        compression='gzip',
                    )

                except Exception as e:
                    print(f"  Error on pair {i}: {e}")
                    continue

            # Free model
            del model
            torch.cuda.empty_cache()

    print(f"\n=== Collection Complete ===")
    print(f"Output: {output_file}")
    print(f"Pairs: {len(pairs)}")
    print(f"Models: {model_names}")

    # Save summary
    summary_file = Path(output_dir) / 'collection_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'n_problems': n_problems,
            'n_pairs': len(pairs),
            'models': model_names,
            'generation_model': GENERATION_MODEL,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    return output_file


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect clean/corrupted pairs for Wynroe-style error detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--n_problems', '-n',
        type=int,
        default=200,
        help='Number of GSM8K problems to use'
    )
    parser.add_argument(
        '--models', '-m',
        type=str,
        default='base,sft,rl_zero,think',
        help='Comma-separated list of models to test'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='experiments/aha_moment/data/wynroe_replication/',
        help='Output directory'
    )
    parser.add_argument(
        '--max_cot_tokens',
        type=int,
        default=512,
        help='Max tokens for CoT generation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',')]

    collect_wynroe_data(
        n_problems=args.n_problems,
        model_names=model_names,
        output_dir=args.output,
        max_cot_tokens=args.max_cot_tokens,
        device=args.device,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
