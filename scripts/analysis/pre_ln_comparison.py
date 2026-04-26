#!/usr/bin/env python3
"""
Pre-LayerNorm vs Post-LayerNorm Comparison

Tests the hypothesis that pre-LayerNorm activations have higher cosine
similarity between consecutive layers, enabling meaningful Jacobian analysis.

Usage:
    python scripts/analysis/pre_ln_comparison.py --model olmo3_base --n-samples 50

Expected output:
    - cos(X_l, X_{l+1}) for pre-LN and post-LN
    - If pre-LN cos >> 0.1, the LayerNorm hypothesis is confirmed
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from task_data import prepare_gsm8k


# Model configurations
MODELS = {
    'olmo3_base': 'allenai/OLMo-3-1025-7B',
    'olmo3_sft': 'allenai/OLMo-3-7B-Think-SFT',
    'olmo3_rl_zero': 'allenai/OLMo-3-7B-RL-Zero-General',
    'olmo3_think': 'allenai/OLMo-3-7B-Think',
}

# Layers to collect (even layers for consistency with Phase 2)
LAYERS = list(range(0, 32, 2))  # [0, 2, 4, ..., 30]


def load_model(model_name: str):
    """Load model with HuggingFace transformers."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cuda' if torch.cuda.is_available() else 'cpu',
    )
    model.eval()

    print(f"✓ Loaded: {model.config.num_hidden_layers} layers, d_model={model.config.hidden_size}")
    return model, tokenizer


def collect_activations_hf(model, tokenizer, prompts: list, hook_point: str, layers: list):
    """Collect mean-pooled activations using HuggingFace hooks.

    hook_point:
        'pre': Input to each layer (before LayerNorm)
        'post': Output of each layer (after LayerNorm + attention + MLP)

    Returns:
        np.ndarray of shape (n_samples, n_layers, d_model)
    """
    n_samples = len(prompts)
    n_layers = len(layers)
    d_model = model.config.hidden_size
    device = next(model.parameters()).device

    activations = np.zeros((n_samples, n_layers, d_model), dtype=np.float32)

    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts, desc=f"Collecting {hook_point}")):
            # Tokenize
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            # Storage for this sample
            layer_acts = {}

            if hook_point == 'pre':
                # Hook the INPUT to each layer (before LayerNorm)
                def make_pre_hook(layer_idx):
                    def hook(module, input, output):
                        # input[0] is the hidden states BEFORE this layer processes them
                        if isinstance(input, tuple) and len(input) > 0:
                            layer_acts[layer_idx] = input[0].detach()
                    return hook

                handles = []
                for j, layer_idx in enumerate(layers):
                    layer = model.model.layers[layer_idx]
                    handle = layer.register_forward_hook(make_pre_hook(j))
                    handles.append(handle)
            else:  # 'post'
                # Hook the OUTPUT of each layer (after full layer computation)
                def make_post_hook(layer_idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            layer_acts[layer_idx] = output[0].detach()
                        else:
                            layer_acts[layer_idx] = output.detach()
                    return hook

                handles = []
                for j, layer_idx in enumerate(layers):
                    layer = model.model.layers[layer_idx]
                    handle = layer.register_forward_hook(make_post_hook(j))
                    handles.append(handle)

            try:
                # Forward pass
                _ = model(**tokens)

                # Extract and mean-pool activations
                for j in range(n_layers):
                    if j in layer_acts:
                        act = layer_acts[j][0].mean(dim=0).cpu().numpy()
                        activations[i, j] = act.astype(np.float32)
            finally:
                # Remove hooks
                for handle in handles:
                    handle.remove()

    return activations


def compute_layer_cosines(activations: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between consecutive layers.

    Args:
        activations: (n_samples, n_layers, d_model)

    Returns:
        (n_layers - 1,) array of mean cosine similarities
    """
    n_samples, n_layers, d_model = activations.shape
    cosines = np.zeros((n_samples, n_layers - 1))

    for i in range(n_layers - 1):
        X_l = activations[:, i, :]       # (n_samples, d_model)
        X_l1 = activations[:, i + 1, :]  # (n_samples, d_model)

        # Normalize
        X_l_norm = X_l / (np.linalg.norm(X_l, axis=1, keepdims=True) + 1e-8)
        X_l1_norm = X_l1 / (np.linalg.norm(X_l1, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity per sample
        cos_sim = (X_l_norm * X_l1_norm).sum(axis=1)
        cosines[:, i] = cos_sim

    return cosines.mean(axis=0)  # Mean over samples


def compute_activation_norms(activations: np.ndarray) -> np.ndarray:
    """Compute mean activation norms per layer.

    Args:
        activations: (n_samples, n_layers, d_model)

    Returns:
        (n_layers,) array of mean norms
    """
    return np.linalg.norm(activations, axis=2).mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description='Pre-LN vs Post-LN Comparison')
    parser.add_argument('--model', type=str, default='olmo3_base',
                        choices=list(MODELS.keys()),
                        help='Model to test')
    parser.add_argument('--n-samples', type=int, default=50,
                        help='Number of samples to collect')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare data
    print(f"\nPreparing GSM8K data (n={args.n_samples})...")
    data = prepare_gsm8k(n_shot=0, n_samples=args.n_samples, seed=args.seed)
    # prepare_gsm8k returns List[Tuple[prompt, answer, metadata]]
    prompts = [item[0] for item in data]

    # Load model
    model_name = MODELS[args.model]
    model, tokenizer = load_model(model_name)

    # Collect activations for both hook points
    print("\n" + "="*60)
    print("Collecting POST-LayerNorm activations (current baseline)...")
    print("="*60)
    post_acts = collect_activations_hf(model, tokenizer, prompts, 'post', LAYERS)

    print("\n" + "="*60)
    print("Collecting PRE-LayerNorm activations (test hypothesis)...")
    print("="*60)
    pre_acts = collect_activations_hf(model, tokenizer, prompts, 'pre', LAYERS)

    # Compute cosine similarities
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    post_cosines = compute_layer_cosines(post_acts)
    pre_cosines = compute_layer_cosines(pre_acts)

    print("\nCosine similarity between consecutive layers:")
    print(f"{'Layer pair':<15} {'Post-LN':<12} {'Pre-LN':<12} {'Ratio':<10}")
    print("-" * 50)

    for i, (layer, next_layer) in enumerate(zip(LAYERS[:-1], LAYERS[1:])):
        ratio = pre_cosines[i] / (post_cosines[i] + 1e-8)
        print(f"{layer:>2} → {next_layer:<10} {post_cosines[i]:<12.4f} {pre_cosines[i]:<12.4f} {ratio:<10.2f}x")

    print("-" * 50)
    print(f"{'Mean':<15} {post_cosines.mean():<12.4f} {pre_cosines.mean():<12.4f} {pre_cosines.mean() / (post_cosines.mean() + 1e-8):<10.2f}x")

    # Compute norms
    post_norms = compute_activation_norms(post_acts)
    pre_norms = compute_activation_norms(pre_acts)

    print("\n\nActivation norms per layer:")
    print(f"{'Layer':<10} {'Post-LN':<12} {'Pre-LN':<12}")
    print("-" * 35)
    for i, layer in enumerate(LAYERS):
        print(f"{layer:<10} {post_norms[i]:<12.2f} {pre_norms[i]:<12.2f}")
    print("-" * 35)
    print(f"{'Mean':<10} {post_norms.mean():<12.2f} {pre_norms.mean():<12.2f}")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    mean_post = post_cosines.mean()
    mean_pre = pre_cosines.mean()

    if mean_pre > 0.5 and mean_post < 0.2:
        print("\n✓ HYPOTHESIS CONFIRMED: Pre-LN activations have high cosine")
        print("  similarity between layers. LayerNorm is causing the")
        print("  orthogonality problem. Dynamical measures may work on pre-LN.")
    elif mean_pre > mean_post * 2:
        print("\n◐ PARTIAL SUPPORT: Pre-LN has notably higher cosine similarity")
        print(f"  ({mean_pre:.3f} vs {mean_post:.3f}). Worth investigating further.")
    else:
        print("\n✗ HYPOTHESIS NOT SUPPORTED: Pre-LN cosine similarity is")
        print("  not substantially higher. The orthogonality may be architectural,")
        print("  not just a LayerNorm effect.")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model': args.model,
        'n_samples': args.n_samples,
        'layers': LAYERS,
        'post_cosines': post_cosines.tolist(),
        'pre_cosines': pre_cosines.tolist(),
        'post_norms': post_norms.tolist(),
        'pre_norms': pre_norms.tolist(),
    }

    import json
    output_file = output_dir / f'pre_ln_comparison_{args.model}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Memory cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
