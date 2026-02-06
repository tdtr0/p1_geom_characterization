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


def load_model(model_name: str, hook_point: str = 'post'):
    """Load model with TransformerLens."""
    from transformer_lens import HookedTransformer

    print(f"Loading model: {model_name} (hook_point={hook_point})")
    model = HookedTransformer.from_pretrained(
        model_name,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float16,
        fold_ln=False,  # Keep LayerNorm separate
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    return model


def get_hook_names(hook_point: str, layers: list) -> list:
    """Get hook names for specified hook point."""
    suffix_map = {
        'pre': 'hook_resid_pre',
        'mid': 'hook_resid_mid',
        'post': 'hook_resid_post',
    }
    return [f"blocks.{layer}.{suffix_map[hook_point]}" for layer in layers]


def collect_activations_simple(model, prompts: list, hook_point: str, layers: list):
    """Collect mean-pooled activations for comparison.

    Returns:
        np.ndarray of shape (n_samples, n_layers, d_model)
    """
    hook_names = get_hook_names(hook_point, layers)
    n_samples = len(prompts)
    n_layers = len(layers)
    d_model = model.cfg.d_model

    activations = np.zeros((n_samples, n_layers, d_model), dtype=np.float32)

    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts, desc=f"Collecting {hook_point}")):
            tokens = model.to_tokens(prompt, prepend_bos=True)

            # Truncate if too long
            if tokens.shape[1] > 512:
                tokens = tokens[:, :512]

            _, cache = model.run_with_cache(
                tokens,
                names_filter=hook_names,
                remove_batch_dim=False
            )

            for j, hook_name in enumerate(hook_names):
                # Mean pool over sequence dimension
                act = cache[hook_name][0].mean(dim=0).cpu().numpy()
                activations[i, j] = act.astype(np.float32)

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
    model = load_model(model_name)

    # Collect activations for both hook points
    print("\n" + "="*60)
    print("Collecting POST-LayerNorm activations (current baseline)...")
    print("="*60)
    post_acts = collect_activations_simple(model, prompts, 'post', LAYERS)

    print("\n" + "="*60)
    print("Collecting PRE-LayerNorm activations (test hypothesis)...")
    print("="*60)
    pre_acts = collect_activations_simple(model, prompts, 'pre', LAYERS)

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
