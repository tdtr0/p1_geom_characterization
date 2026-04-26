#!/usr/bin/env python3
"""
Collect thinking traces with full trajectories from reasoning models.

This script:
1. Loads a thinking model (e.g., DeepSeek-R1-Distill)
2. Generates CoT responses to benchmark problems
3. Collects activation trajectories at all layers
4. Identifies pivot tokens (BUT, wait, actually, etc.)
5. Saves everything to HDF5

Usage:
    python collect_thinking_traces.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --benchmark decimal_comparison \
        --n_samples 100 \
        --output data/thinking_traces/
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Pivot Token Detection
# ============================================================================

PIVOT_PATTERNS = [
    (r'\bBUT\b', 'BUT'),
    (r'\bWait\b', 'Wait'),
    (r'\bwait\b', 'wait'),
    (r'\bactually\b', 'actually'),
    (r'\bActually\b', 'Actually'),
    (r'\bhowever\b', 'however'),
    (r'\bHowever\b', 'However'),
    (r'\blet me reconsider\b', 'reconsider'),
    (r'\bon second thought\b', 'second_thought'),
    (r'\bI was wrong\b', 'wrong'),
    (r'\bno,\s', 'no'),
    (r'\bNo,\s', 'No'),
    (r'\bhmm\b', 'hmm'),
    (r'\bHmm\b', 'Hmm'),
]


def find_pivot_tokens(text: str, tokenizer) -> list[dict]:
    """Find pivot tokens in generated text and return their token indices."""
    pivots = []

    for pattern, label in PIVOT_PATTERNS:
        for match in re.finditer(pattern, text):
            char_start = match.start()
            # Find which token this character position maps to
            # Tokenize prefix to get token index
            prefix = text[:char_start]
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            token_idx = len(prefix_tokens)

            pivots.append({
                'pattern': label,
                'char_start': char_start,
                'char_end': match.end(),
                'token_idx': token_idx,
                'matched_text': match.group(),
            })

    # Sort by position
    pivots.sort(key=lambda x: x['char_start'])
    return pivots


# ============================================================================
# Benchmark Problems
# ============================================================================

def get_decimal_comparison_problems(n: int = 100, seed: int = 42) -> list[dict]:
    """Generate decimal comparison problems where naive digit comparison fails."""
    np.random.seed(seed)
    problems = []

    # Cases where .XX looks bigger but isn't
    tricky_pairs = [
        (9.9, 9.11, '>'),      # .9 > .11
        (3.9, 3.14, '>'),      # .9 > .14
        (7.8, 7.12, '>'),      # .8 > .12
        (5.5, 5.123, '>'),     # .5 > .123
        (2.7, 2.15, '>'),      # .7 > .15
        (8.6, 8.234, '>'),     # .6 > .234
        (1.4, 1.39, '>'),      # .4 > .39
        (6.3, 6.29, '>'),      # .3 > .29
        (4.2, 4.19, '>'),      # .2 > .19
        (0.9, 0.89, '>'),      # .9 > .89
    ]

    for i in range(n):
        if i < len(tricky_pairs):
            a, b, relation = tricky_pairs[i]
        else:
            # Generate more
            base = np.random.randint(1, 10)
            # Make a decimal that looks smaller but is bigger
            dec_a = np.random.randint(5, 10) / 10  # .5 to .9
            dec_b = np.random.randint(10, 50) / 100  # .10 to .49
            a = base + dec_a
            b = base + dec_b
            relation = '>' if a > b else '<'

        question = f"Which is larger: {a} or {b}?"
        correct = f"{a}" if relation == '>' else f"{b}"
        naive_wrong = f"{b}" if relation == '>' else f"{a}"

        problems.append({
            'id': f'decimal_{i}',
            'question': question,
            'value_a': a,
            'value_b': b,
            'correct_answer': correct,
            'naive_wrong_answer': naive_wrong,
            'explanation': f"{a} {'>' if relation == '>' else '<'} {b} because {a-int(a):.10f} {'>' if relation == '>' else '<'} {b-int(b):.10f}"
        })

    return problems


def get_benchmark(name: str, n: int = 100, seed: int = 42) -> list[dict]:
    """Load benchmark problems by name."""
    if name == 'decimal_comparison':
        return get_decimal_comparison_problems(n, seed)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


# ============================================================================
# Activation Collection
# ============================================================================

class ActivationCollector:
    """Collect activations during model generation."""

    def __init__(self, model, layers_to_collect: Optional[list[int]] = None):
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
            # Collect all layers
            self.layers_to_collect = list(range(self.n_layers))
        else:
            self.layers_to_collect = layers_to_collect

    def _get_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Store: (batch, seq_len, hidden_dim)
            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(hidden.detach().cpu().float())

        return hook

    def register_hooks(self):
        """Register forward hooks on specified layers."""
        self.activations = {}
        self.hooks = []

        # Get layer modules
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            raise ValueError("Cannot find layers")

        for layer_idx in self.layers_to_collect:
            hook = layers[layer_idx].register_forward_hook(self._get_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_trajectory(self) -> np.ndarray:
        """
        Get collected trajectory as numpy array.
        Returns: (seq_len, n_layers, hidden_dim)
        """
        if not self.activations:
            return None

        # Concatenate activations from generation steps
        trajectories = []
        for layer_idx in sorted(self.activations.keys()):
            # Each entry is (batch=1, seq_len, hidden_dim) for each generation step
            layer_acts = torch.cat(self.activations[layer_idx], dim=1)  # (1, total_seq, hidden)
            trajectories.append(layer_acts.squeeze(0).numpy())  # (total_seq, hidden)

        # Stack: (n_layers, seq_len, hidden_dim) -> (seq_len, n_layers, hidden_dim)
        trajectory = np.stack(trajectories, axis=0)  # (n_layers, seq_len, hidden)
        trajectory = np.transpose(trajectory, (1, 0, 2))  # (seq_len, n_layers, hidden)

        return trajectory


# ============================================================================
# Main Collection
# ============================================================================

def collect_thinking_traces(
    model_name: str,
    benchmark_name: str,
    n_samples: int,
    output_dir: str,
    max_new_tokens: int = 1024,
    device: str = 'cuda',
    seed: int = 42,
    layers_to_collect: Optional[list[int]] = None,
):
    """
    Collect thinking traces with trajectories.

    Args:
        model_name: HuggingFace model name
        benchmark_name: Name of benchmark to use
        n_samples: Number of samples to collect
        output_dir: Where to save output
        max_new_tokens: Max tokens to generate
        device: 'cuda' or 'cpu'
        seed: Random seed
        layers_to_collect: Which layers to collect (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get problems
    print(f"Loading benchmark: {benchmark_name}")
    problems = get_benchmark(benchmark_name, n_samples, seed)

    # Setup activation collector
    collector = ActivationCollector(model, layers_to_collect)

    # Output file
    model_short = model_name.split('/')[-1].replace('-', '_').lower()
    output_file = Path(output_dir) / f"{model_short}_{benchmark_name}.h5"

    # Collect
    results = []

    with h5py.File(output_file, 'w') as f:
        # Metadata
        f.attrs['model'] = model_name
        f.attrs['benchmark'] = benchmark_name
        f.attrs['n_samples'] = n_samples
        f.attrs['max_new_tokens'] = max_new_tokens
        f.attrs['seed'] = seed
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['n_layers'] = collector.n_layers
        f.attrs['layers_collected'] = collector.layers_to_collect

        # Create groups
        traj_group = f.create_group('trajectories')
        meta_group = f.create_group('metadata')

        for i, problem in enumerate(tqdm(problems, desc="Collecting")):
            # Format prompt
            prompt = f"<think>\n{problem['question']}\nLet me think step by step.\n"

            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            input_len = inputs['input_ids'].shape[1]

            # Register hooks and generate
            collector.register_hooks()

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Greedy for reproducibility
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Get trajectory
                trajectory = collector.get_trajectory()

                # Decode output
                generated_ids = outputs[0][input_len:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Find pivot tokens
                pivots = find_pivot_tokens(generated_text, tokenizer)

                # Check correctness (simple heuristic)
                is_correct = problem['correct_answer'] in generated_text
                has_naive_wrong = problem['naive_wrong_answer'] in generated_text.split(problem['correct_answer'])[0] if problem['correct_answer'] in generated_text else True

                # Save trajectory
                traj_group.create_dataset(
                    f'sample_{i}',
                    data=trajectory.astype(np.float16),
                    compression='gzip',
                    compression_opts=4,
                )

                # Save metadata as JSON string
                meta = {
                    'problem_id': problem['id'],
                    'question': problem['question'],
                    'correct_answer': problem['correct_answer'],
                    'generated_text': generated_text,
                    'is_correct': is_correct,
                    'has_naive_wrong_first': has_naive_wrong,
                    'pivot_tokens': pivots,
                    'n_tokens_generated': len(generated_ids),
                    'trajectory_shape': list(trajectory.shape),
                }
                meta_group.create_dataset(f'sample_{i}', data=json.dumps(meta))

                results.append(meta)

            finally:
                collector.remove_hooks()
                collector.activations = {}

            # Progress update
            if (i + 1) % 10 == 0:
                n_correct = sum(1 for r in results if r['is_correct'])
                n_with_pivots = sum(1 for r in results if r['pivot_tokens'])
                print(f"  Progress: {i+1}/{n_samples}, "
                      f"Correct: {n_correct}/{len(results)} ({100*n_correct/len(results):.1f}%), "
                      f"With pivots: {n_with_pivots}/{len(results)}")

    # Summary
    n_correct = sum(1 for r in results if r['is_correct'])
    n_with_pivots = sum(1 for r in results if r['pivot_tokens'])

    print(f"\n=== Collection Complete ===")
    print(f"Output: {output_file}")
    print(f"Samples: {len(results)}")
    print(f"Correct: {n_correct} ({100*n_correct/len(results):.1f}%)")
    print(f"With pivot tokens: {n_with_pivots} ({100*n_with_pivots/len(results):.1f}%)")

    # Save summary
    summary_file = Path(output_dir) / f"{model_short}_{benchmark_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'model': model_name,
            'benchmark': benchmark_name,
            'n_samples': len(results),
            'n_correct': n_correct,
            'n_with_pivots': n_with_pivots,
            'results': results,
        }, f, indent=2)

    return output_file


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Collect thinking traces with trajectories')
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name (e.g., deepseek-ai/DeepSeek-R1-Distill-Llama-8B)')
    parser.add_argument('--benchmark', type=str, default='decimal_comparison',
                        choices=['decimal_comparison'],
                        help='Benchmark to use')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to collect')
    parser.add_argument('--output', type=str, default='data/thinking_traces/',
                        help='Output directory')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Max tokens to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--layers', type=str, default=None,
                        help='Layers to collect (e.g., "0,2,4,6" or "all")')

    args = parser.parse_args()

    # Parse layers
    if args.layers is None or args.layers == 'all':
        layers = None
    else:
        layers = [int(x) for x in args.layers.split(',')]

    collect_thinking_traces(
        model_name=args.model,
        benchmark_name=args.benchmark,
        n_samples=args.n_samples,
        output_dir=args.output,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        seed=args.seed,
        layers_to_collect=layers,
    )


if __name__ == '__main__':
    main()
