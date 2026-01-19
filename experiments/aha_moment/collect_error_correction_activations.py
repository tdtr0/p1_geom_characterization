#!/usr/bin/env python3
"""
Collect activation trajectories during error correction experiment.

This script:
1. Loads test cases from Experiment C (corrupted math problems)
2. Feeds corrupted prefix to each model
3. Collects hidden state activations during generation
4. Saves trajectories for geometric analysis

Usage:
    python collect_error_correction_activations.py --model base --gpu 0
    python collect_error_correction_activations.py --model think --gpu 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configurations
MODEL_CONFIGS = {
    'base': 'allenai/OLMo-3-1025-7B',
    'rl_zero': 'allenai/OLMo-3-7B-RL-Zero-General',
    'think': 'allenai/OLMo-3-7B-Think',
}

# Layers to collect (even layers for efficiency)
LAYERS_TO_COLLECT = list(range(0, 32, 2))  # [0, 2, 4, ..., 30]

def load_test_cases(results_dir: Path) -> list:
    """Load test cases from Experiment C results."""
    details_file = results_dir / "correction_details.json"
    with open(details_file) as f:
        data = json.load(f)
    return data["test_cases"]

def create_corrupted_prompt(test_case: dict) -> str:
    """Create the prompt with corrupted prefix for the model."""
    question = test_case["question"]
    corrupted_trace = test_case["corrupted_trace"]

    # Format as chat-style prompt
    prompt = f"""Solve this math problem step by step.

Question: {question}

Solution: {corrupted_trace}"""

    return prompt

def collect_activations_during_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    layers: list = None,
) -> dict:
    """
    Generate continuation and collect hidden states at each token.

    Returns:
        dict with:
            - 'prompt_activations': activations at prompt tokens (n_prompt, n_layers, d_model)
            - 'gen_activations': activations at generated tokens (n_gen, n_layers, d_model)
            - 'prompt_tokens': token ids for prompt
            - 'gen_tokens': token ids for generated text
            - 'generated_text': the full generated text
    """
    if layers is None:
        layers = LAYERS_TO_COLLECT

    device = model.device

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids[0].tolist()

    # Get activations for prompt (single forward pass)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        prompt_hidden = outputs.hidden_states  # tuple of (1, seq_len, d_model)

        # Extract selected layers
        prompt_activations = []
        for layer_idx in layers:
            layer_act = prompt_hidden[layer_idx][0].cpu().numpy()  # (seq_len, d_model)
            prompt_activations.append(layer_act)
        prompt_activations = np.stack(prompt_activations, axis=1)  # (seq_len, n_layers, d_model)

    # Generate with hidden states collection
    gen_activations = []
    gen_tokens = []

    current_ids = inputs.input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(current_ids, output_hidden_states=True)

            # Get next token
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Collect activations for the last (new) position
            layer_acts = []
            for layer_idx in layers:
                layer_act = outputs.hidden_states[layer_idx][0, -1, :].cpu().numpy()  # (d_model,)
                layer_acts.append(layer_act)
            gen_activations.append(np.stack(layer_acts))  # (n_layers, d_model)
            gen_tokens.append(next_token.item())

            # Append token
            current_ids = torch.cat([current_ids, next_token], dim=1)

    # Convert to arrays
    if gen_activations:
        gen_activations = np.stack(gen_activations)  # (n_gen, n_layers, d_model)
    else:
        gen_activations = np.zeros((0, len(layers), model.config.hidden_size))

    # Decode generated text
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    return {
        'prompt_activations': prompt_activations.astype(np.float16),
        'gen_activations': gen_activations.astype(np.float16),
        'prompt_tokens': prompt_ids,
        'gen_tokens': gen_tokens,
        'generated_text': generated_text,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output-dir", type=str, default="results/error_correction_activations")
    args = parser.parse_args()

    # Setup
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results" / "error_correction"
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test cases
    print("Loading test cases...")
    test_cases = load_test_cases(results_dir)
    if args.max_samples:
        test_cases = test_cases[:args.max_samples]
    print(f"Loaded {len(test_cases)} test cases")

    # Load model
    model_path = MODEL_CONFIGS[args.model]
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Output file
    output_file = output_dir / f"{args.model}_activations.h5"
    print(f"Output file: {output_file}")

    # Collect activations
    with h5py.File(output_file, 'w') as f:
        # Store metadata
        f.attrs['model'] = args.model
        f.attrs['model_path'] = model_path
        f.attrs['layers'] = LAYERS_TO_COLLECT
        f.attrs['n_samples'] = len(test_cases)

        for i, test_case in enumerate(tqdm(test_cases, desc=f"Collecting {args.model}")):
            prompt = create_corrupted_prompt(test_case)

            try:
                result = collect_activations_during_generation(
                    model, tokenizer, prompt,
                    max_new_tokens=150,
                    layers=LAYERS_TO_COLLECT,
                )

                # Create group for this sample
                grp = f.create_group(f"sample_{i:04d}")
                grp.attrs['problem_id'] = test_case['problem_id']
                grp.attrs['ground_truth'] = test_case['ground_truth']
                grp.attrs['corrupted_value'] = test_case['corrupted_value']
                grp.attrs['generated_text'] = result['generated_text']

                # Store activations
                grp.create_dataset('prompt_activations', data=result['prompt_activations'],
                                   compression='gzip', compression_opts=4)
                grp.create_dataset('gen_activations', data=result['gen_activations'],
                                   compression='gzip', compression_opts=4)
                grp.create_dataset('prompt_tokens', data=result['prompt_tokens'])
                grp.create_dataset('gen_tokens', data=result['gen_tokens'])

            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue

            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

    print(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    main()
