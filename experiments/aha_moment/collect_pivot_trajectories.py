#!/usr/bin/env python3
"""
Collect generation trajectories for pivot analysis.

This script generates text token-by-token and collects hidden states
at each generation step, enabling analysis of trajectory dynamics at
pivot points (self-correction moments).

Key difference from Phase 2 collection:
- Phase 2 only collected prompt trajectories (66 tokens)
- This script collects GENERATION trajectories (up to 512 tokens)

Usage:
    python collect_pivot_trajectories.py \
        --n_samples 200 \
        --model olmo3_think \
        --max_tokens 512 \
        --output experiments/aha_moment/data/pivot_collection/
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Model configurations
MODELS = {
    'olmo3_base': 'allenai/OLMo-3-1025-7B',
    'olmo3_sft': 'allenai/OLMo-3-7B-Think-SFT',
    'olmo3_rl_zero': 'allenai/OLMo-3-7B-RL-Zero-General',
    'olmo3_think': 'allenai/OLMo-3-7B-Think',
}

# Layers to collect (even layers for efficiency)
LAYERS_TO_COLLECT = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]


def load_gsm8k(n_samples: int, seed: int = 42) -> list[dict]:
    """Load GSM8K problems."""
    print(f"Loading GSM8K dataset (n={n_samples})...")
    dataset = load_dataset("gsm8k", "main", split="test")

    # Shuffle and select
    import random
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:n_samples]

    problems = []
    for idx in indices:
        item = dataset[idx]
        problems.append({
            'question': item['question'],
            'answer': item['answer'],
        })

    print(f"Loaded {len(problems)} problems")
    return problems


def create_prompt(question: str) -> str:
    """Create prompt for GSM8K problem."""
    return f"""Solve this math problem step by step.

Problem: {question}

Solution:"""


def collect_generation_trajectory(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    layers: list[int] = None,
) -> tuple[np.ndarray, str, list[int]]:
    """
    Generate text token-by-token and collect hidden states.

    Returns:
        trajectory: (n_tokens, n_layers, hidden_dim) array
        generated_text: The generated text
        token_ids: List of generated token IDs
    """
    if layers is None:
        layers = LAYERS_TO_COLLECT

    device = next(model.parameters()).device

    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    # Storage for trajectories
    trajectories = []
    generated_ids = []

    # Generate token by token
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass with hidden states
            outputs = model(
                input_ids,
                output_hidden_states=True,
                use_cache=False,  # Disable cache for clean hidden states
            )

            # Get hidden states at last token position for specified layers
            hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)

            # Extract activations at last token for each layer
            step_activations = []
            for layer_idx in layers:
                if layer_idx < len(hidden_states):
                    # Get last token's hidden state
                    h = hidden_states[layer_idx][0, -1, :].cpu().numpy()
                    step_activations.append(h)

            step_activations = np.stack(step_activations, axis=0)  # (n_layers, hidden)
            trajectories.append(step_activations)

            # Sample next token (greedy for reproducibility)
            logits = outputs.logits[0, -1, :]
            next_token = logits.argmax().unsqueeze(0).unsqueeze(0)
            generated_ids.append(next_token.item())

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Append token and continue
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Stack trajectories: (n_tokens, n_layers, hidden)
    trajectory = np.stack(trajectories, axis=0).astype(np.float16)

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return trajectory, generated_text, generated_ids


def main():
    parser = argparse.ArgumentParser(description="Collect generation trajectories for pivot analysis")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--model", type=str, default="olmo3_think",
                        choices=list(MODELS.keys()), help="Model to use")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max generation length")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load problems
    problems = load_gsm8k(args.n_samples, args.seed)

    # Load model
    model_path = MODELS[args.model]
    print(f"\nLoading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Prepare HDF5 file
    h5_path = output_dir / "pivot_trajectories.h5"

    # Collect trajectories
    print(f"\nCollecting trajectories for {len(problems)} problems...")

    all_trajectories = []
    all_texts = []
    all_prompts = []
    all_token_ids = []
    all_answers = []

    for i, problem in enumerate(tqdm(problems, desc="Generating")):
        prompt = create_prompt(problem['question'])

        try:
            trajectory, generated_text, token_ids = collect_generation_trajectory(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                layers=LAYERS_TO_COLLECT,
            )

            all_trajectories.append(trajectory)
            all_texts.append(generated_text)
            all_prompts.append(prompt)
            all_token_ids.append(token_ids)
            all_answers.append(problem['answer'])

        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            # Store empty data
            all_trajectories.append(np.zeros((1, len(LAYERS_TO_COLLECT), 4096), dtype=np.float16))
            all_texts.append("")
            all_prompts.append(prompt)
            all_token_ids.append([])
            all_answers.append(problem['answer'])

        # Clear cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # Pad trajectories to same length
    max_len = max(t.shape[0] for t in all_trajectories)
    n_layers = len(LAYERS_TO_COLLECT)
    hidden_dim = all_trajectories[0].shape[-1]

    padded_trajectories = np.zeros(
        (len(all_trajectories), max_len, n_layers, hidden_dim),
        dtype=np.float16
    )
    sequence_lengths = []

    for i, traj in enumerate(all_trajectories):
        seq_len = traj.shape[0]
        padded_trajectories[i, :seq_len, :, :] = traj
        sequence_lengths.append(seq_len)

    # Save to HDF5
    print(f"\nSaving to {h5_path}...")
    with h5py.File(h5_path, 'w') as f:
        # Trajectories
        f.create_dataset(
            'trajectories',
            data=padded_trajectories,
            compression='gzip',
            compression_opts=4,
        )

        # Sequence lengths
        f.create_dataset('sequence_lengths', data=np.array(sequence_lengths))

        # Generated texts
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('generated_texts', data=np.array(all_texts, dtype=object), dtype=dt)
        f.create_dataset('prompts', data=np.array(all_prompts, dtype=object), dtype=dt)
        f.create_dataset('ground_truth', data=np.array(all_answers, dtype=object), dtype=dt)

        # Metadata
        f.attrs['model'] = args.model
        f.attrs['model_path'] = model_path
        f.attrs['n_samples'] = len(problems)
        f.attrs['max_tokens'] = args.max_tokens
        f.attrs['layers'] = LAYERS_TO_COLLECT
        f.attrs['seed'] = args.seed
        f.attrs['timestamp'] = datetime.now().isoformat()

    # Save summary
    summary = {
        'model': args.model,
        'model_path': model_path,
        'n_samples': len(problems),
        'max_tokens': args.max_tokens,
        'layers': LAYERS_TO_COLLECT,
        'trajectory_shape': list(padded_trajectories.shape),
        'avg_sequence_length': float(np.mean(sequence_lengths)),
        'max_sequence_length': int(max(sequence_lengths)),
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
    }

    summary_path = output_dir / "collection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Collection Complete ===")
    print(f"Output: {h5_path}")
    print(f"Samples: {len(problems)}")
    print(f"Trajectory shape: {padded_trajectories.shape}")
    print(f"Avg sequence length: {np.mean(sequence_lengths):.1f}")
    print(f"Max sequence length: {max(sequence_lengths)}")


if __name__ == "__main__":
    main()
