#!/usr/bin/env python3
"""
Proper Wynroe Replication: Activation Patching on MATH Dataset

This script replicates Wynroe et al.'s methodology:
1. Take clean/corrupted solution pairs
2. For each layer, patch corrupted activations with clean activations
3. Measure logit-diff recovery for correction tokens ("Wait", "But", "Actually")
4. Find the critical layer where patching matters most

Key differences from our previous Experiment A:
- Uses activation PATCHING (causal), not probing (correlational)
- Uses MATH dataset (harder), not GSM8K (too easy)
- Measures logit-diff recovery, not Cohen's d

Reference: https://www.lesswrong.com/posts/BnzDiSYKfrHaFPefi/finding-an-error-detection-feature-in-deepseek-r1

Usage:
    python replicate_wynroe_patching.py --model think --gpu 0 --n_samples 50
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import h5py
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configurations
MODEL_CONFIGS = {
    'base': 'allenai/OLMo-3-1025-7B',
    'rl_zero': 'allenai/OLMo-3-7B-RL-Zero-General',
    'think': 'allenai/OLMo-3-7B-Think',
    # DeepSeek-R1 distilled model (what Wynroe used)
    'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
}

# Architecture-specific layer access paths
# Qwen-based models use same path as OLMo
def get_layers(model, model_key: str):
    """Get the transformer layers from the model, handling architecture differences."""
    if model_key == 'deepseek':
        # Qwen architecture: model.model.layers
        return model.model.layers
    else:
        # OLMo architecture: model.model.layers
        return model.model.layers

# Correction tokens to measure logit-diff for
CORRECTION_TOKENS = ["Wait", "But", "Actually", "Hmm", "wait", "but", "actually"]

# Default layers to patch (will be dynamically adjusted for each model)
LAYERS_TO_PATCH_DEFAULT = list(range(0, 32, 2))  # [0, 2, 4, ..., 30]


def get_layers_to_patch(model, model_key: str) -> List[int]:
    """Get layer indices to patch based on model architecture."""
    model_layers = get_layers(model, model_key)
    n_layers = len(model_layers)
    # Patch every 2nd layer
    return list(range(0, n_layers, 2))


def load_math_dataset(n_samples: int = 100, subset: str = "algebra") -> List[Dict]:
    """
    Load MATH dataset problems.

    Args:
        n_samples: Number of problems to load
        subset: MATH subset (algebra, geometry, number_theory, etc.)

    Returns:
        List of problems with 'problem' and 'solution' fields
    """
    print(f"Loading MATH dataset ({subset})...")

    try:
        # Use EleutherAI's version (hendrycks/competition_math has DMCA issues)
        dataset = load_dataset("EleutherAI/hendrycks_math", subset, split="train", trust_remote_code=True)
        problems = [
            {"problem": item["problem"], "solution": item["solution"]}
            for item in dataset
        ]
        print(f"Loaded {len(problems)} problems from MATH/{subset}")
        # Shuffle and take n_samples
        import random
        random.shuffle(problems)
        return problems[:n_samples]
    except Exception as e:
        print(f"Warning: Could not load MATH/{subset}: {e}")

        # Try another subset
        for alt_subset in ["algebra", "number_theory", "counting_and_probability"]:
            if alt_subset != subset:
                try:
                    dataset = load_dataset("EleutherAI/hendrycks_math", alt_subset, split="train", trust_remote_code=True)
                    problems = [{"problem": item["problem"], "solution": item["solution"]} for item in dataset]
                    print(f"Loaded {len(problems)} problems from MATH/{alt_subset}")
                    import random
                    random.shuffle(problems)
                    return problems[:n_samples]
                except:
                    continue

        print("Falling back to GSM8K with harder problems...")
        # Fallback to GSM8K
        dataset = load_dataset("gsm8k", "main", split="train")
        problems = [
            {"problem": item["question"], "solution": item["answer"]}
            for item in dataset
        ]
        # Filter for longer solutions (proxy for harder problems)
        problems = [p for p in problems if len(p["solution"]) > 300]
        return problems[:n_samples]


def generate_solution(model, tokenizer, problem: str, max_tokens: int = 512) -> str:
    """Generate a solution trace from the model."""
    prompt = f"Solve this math problem step by step:\n\n{problem}\n\nSolution:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the solution part
    solution = full_output[len(prompt):].strip()

    return solution


def inject_arithmetic_error(solution: str) -> Tuple[str, str, int]:
    """
    Inject an arithmetic error into the solution.

    Returns:
        (clean_solution, corrupted_solution, error_position)
    """
    # Find arithmetic expressions like "= 123" or "= $456"
    pattern = r'=\s*\$?(\d+(?:\.\d+)?)'
    matches = list(re.finditer(pattern, solution))

    if not matches:
        return None, None, -1

    # Pick a random match (prefer later ones as they're often final calculations)
    match = matches[-1] if len(matches) > 1 else matches[0]
    original_num = match.group(1)

    # Corrupt by adding 1
    try:
        if '.' in original_num:
            corrupted_num = str(float(original_num) + 1)
        else:
            corrupted_num = str(int(original_num) + 1)
    except ValueError:
        return None, None, -1

    # Create corrupted solution
    start, end = match.start(1), match.end(1)
    corrupted_solution = solution[:start] + corrupted_num + solution[end:]
    error_position = start  # Character position of error

    return solution, corrupted_solution, error_position


def get_correction_token_logits(model, tokenizer, prompt: str) -> Dict[str, float]:
    """
    Get logits for correction tokens at the next position.

    Returns:
        Dict mapping token -> logit value
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]  # (vocab_size,)

    # Get logits for correction tokens
    correction_logits = {}
    for token in CORRECTION_TOKENS:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            token_id = token_ids[0]  # Take first token if multi-token
            correction_logits[token] = next_token_logits[token_id].item()

    return correction_logits


def activation_patch_forward(
    model,
    tokenizer,
    corrupted_prompt: str,
    clean_activations: Dict[int, torch.Tensor],
    patch_layer: int,
    model_key: str = "think",
) -> Dict[str, float]:
    """
    Run forward pass with patching: replace corrupted activations with clean ones at specified layer.

    Args:
        model: The model
        tokenizer: The tokenizer
        corrupted_prompt: The prompt with error
        clean_activations: Dict mapping layer_idx -> clean hidden states
        patch_layer: Which layer to patch
        model_key: Model key for architecture-specific layer access

    Returns:
        Dict of correction token logits after patching
    """
    inputs = tokenizer(corrupted_prompt, return_tensors="pt").to(model.device)
    model_layers = get_layers(model, model_key)

    # Store hooks to remove later
    hooks = []

    def make_patch_hook(layer_idx: int):
        def hook(module, input, output):
            # output is (hidden_states, ...) or just hidden_states
            # CRITICAL: Must clone before modifying - in-place ops don't propagate correctly
            if isinstance(output, tuple):
                hidden_states = output[0].clone()  # Clone to ensure modification propagates
                # Replace with clean activations
                patched = clean_activations[layer_idx].to(hidden_states.device)
                # Match sequence length
                min_len = min(hidden_states.shape[1], patched.shape[1])
                hidden_states[:, :min_len, :] = patched[:, :min_len, :]
                return (hidden_states,) + output[1:]
            else:
                new_output = output.clone()  # Clone to ensure modification propagates
                patched = clean_activations[layer_idx].to(new_output.device)
                min_len = min(new_output.shape[1], patched.shape[1])
                new_output[:, :min_len, :] = patched[:, :min_len, :]
                return new_output
        return hook

    # Register hook at the specified layer
    layer = model_layers[patch_layer]
    hook = layer.register_forward_hook(make_patch_hook(patch_layer))
    hooks.append(hook)

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

        # Get correction token logits
        correction_logits = {}
        for token in CORRECTION_TOKENS:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if token_ids:
                token_id = token_ids[0]
                correction_logits[token] = next_token_logits[token_id].item()

        return correction_logits

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()


def collect_clean_activations(
    model,
    tokenizer,
    prompt: str,
    layers: List[int],
    model_key: str = "think",
) -> Dict[int, torch.Tensor]:
    """
    Collect activations for clean prompt at specified layers.

    Returns:
        Dict mapping layer_idx -> hidden states tensor
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    activations = {}
    model_layers = get_layers(model, model_key)

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[layer_idx] = output[0].detach().clone()
            else:
                activations[layer_idx] = output.detach().clone()
        return hook

    hooks = []
    for layer_idx in layers:
        layer = model_layers[layer_idx]
        hook = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    try:
        with torch.no_grad():
            model(**inputs)
        return activations
    finally:
        for hook in hooks:
            hook.remove()


def compute_logit_diff(logits_a: Dict[str, float], logits_b: Dict[str, float]) -> float:
    """
    Compute average logit difference for correction tokens.

    logit_diff = mean(logits_a[token] - logits_b[token]) for all tokens
    """
    diffs = []
    for token in CORRECTION_TOKENS:
        if token in logits_a and token in logits_b:
            diffs.append(logits_a[token] - logits_b[token])
    return np.mean(diffs) if diffs else 0.0


def run_patching_experiment(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    layers_to_patch: List[int],
    model_key: str = "think",
) -> Dict[str, Any]:
    """
    Run full patching experiment for one clean/corrupted pair.

    Returns:
        Dict with recovery % at each layer
    """
    # Step 1: Get baseline logits (no patching)
    clean_logits = get_correction_token_logits(model, tokenizer, clean_prompt)
    corrupted_logits = get_correction_token_logits(model, tokenizer, corrupted_prompt)

    # Baseline logit-diff (clean - corrupted)
    baseline_diff = compute_logit_diff(clean_logits, corrupted_logits)

    # Wynroe filtered for logit-diff > 3 (kept only 44% of pairs)
    # This ensures we only keep pairs where the model shows STRONG response to the error
    if abs(baseline_diff) < 3.0:
        # No meaningful difference - skip this pair (model doesn't notice error)
        return None

    # Step 2: Collect clean activations
    clean_activations = collect_clean_activations(model, tokenizer, clean_prompt, layers_to_patch, model_key)

    # Step 3: Patch at each layer and measure recovery
    results = {
        "baseline_diff": baseline_diff,
        "clean_logits": clean_logits,
        "corrupted_logits": corrupted_logits,
        "layer_results": {},
    }

    for layer in layers_to_patch:
        patched_logits = activation_patch_forward(
            model, tokenizer, corrupted_prompt, clean_activations, layer, model_key
        )

        # Compute how much of the baseline diff we recovered
        patched_diff = compute_logit_diff(patched_logits, corrupted_logits)
        recovery_pct = (patched_diff / baseline_diff) * 100 if baseline_diff != 0 else 0

        results["layer_results"][layer] = {
            "patched_logits": patched_logits,
            "patched_diff": patched_diff,
            "recovery_pct": recovery_pct,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="think", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to test: think, base, rl_zero, or deepseek (Wynroe's exact model)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=50, help="Number of problems to test")
    parser.add_argument("--output-dir", type=str, default="results/wynroe_patching")
    args = parser.parse_args()

    # Setup
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset - need 3-4x target because:
    # 1. ~50% fail error injection
    # 2. ~56% filtered out (logit-diff < 3, like Wynroe)
    problems = load_math_dataset(n_samples=args.n_samples * 5)
    print(f"Loaded {len(problems)} problems (target: {args.n_samples} after filtering)")

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

    # Determine layers to patch based on model architecture
    layers_to_patch = get_layers_to_patch(model, args.model)
    print(f"Layers to patch: {layers_to_patch} ({len(layers_to_patch)} layers)")

    # Run experiment
    all_results = []
    successful_pairs = 0
    filtered_low_diff = 0  # Track how many filtered for low logit-diff
    failed_injection = 0   # Track how many failed error injection

    for i, problem in enumerate(tqdm(problems, desc="Running patching experiment")):
        if successful_pairs >= args.n_samples:
            break

        # Step 1: Generate solution
        solution = generate_solution(model, tokenizer, problem["problem"])

        # Step 2: Inject error
        clean, corrupted, error_pos = inject_arithmetic_error(solution)
        if clean is None:
            failed_injection += 1
            continue

        # Step 3: Create prompts (prefix up to error)
        prefix = f"Solve this math problem step by step:\n\n{problem['problem']}\n\nSolution:"
        clean_prompt = prefix + " " + clean[:error_pos + 20]  # Include some context after error
        corrupted_prompt = prefix + " " + corrupted[:error_pos + 20]

        # Step 4: Run patching experiment
        try:
            result = run_patching_experiment(
                model, tokenizer, clean_prompt, corrupted_prompt, layers_to_patch, args.model
            )

            if result is not None:
                result["problem_idx"] = i
                result["problem"] = problem["problem"][:200]  # Truncate for storage
                all_results.append(result)
                successful_pairs += 1
            else:
                filtered_low_diff += 1  # Filtered for logit-diff < 3
        except Exception as e:
            print(f"Error on problem {i}: {e}")
            continue

        # Clear cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()

    total_attempted = i + 1
    print(f"\n" + "=" * 60)
    print("FILTERING SUMMARY (Wynroe kept 44%)")
    print("=" * 60)
    print(f"Total attempted:     {total_attempted}")
    print(f"Failed injection:    {failed_injection} ({100*failed_injection/max(1,total_attempted):.1f}%)")
    print(f"Filtered (diff<3):   {filtered_low_diff} ({100*filtered_low_diff/max(1,total_attempted):.1f}%)")
    print(f"Successful pairs:    {len(all_results)} ({100*len(all_results)/max(1,total_attempted):.1f}%)")
    print(f"(Wynroe retention:   44%)")

    # Aggregate results
    layer_recovery = {layer: [] for layer in layers_to_patch}
    for result in all_results:
        for layer, layer_result in result["layer_results"].items():
            layer_recovery[layer].append(layer_result["recovery_pct"])

    # Compute summary statistics
    summary = {
        "model": args.model,
        "n_pairs": len(all_results),
        "layers": layers_to_patch,
        "layer_stats": {},
    }

    print("\n" + "=" * 60)
    print("LAYER PROFILE: Recovery % by Layer")
    print("=" * 60)

    best_layer = None
    best_recovery = -float("inf")

    for layer in layers_to_patch:
        recoveries = layer_recovery[layer]
        if recoveries:
            mean_recovery = np.mean(recoveries)
            std_recovery = np.std(recoveries)
            summary["layer_stats"][layer] = {
                "mean_recovery": float(mean_recovery),
                "std_recovery": float(std_recovery),
                "n_samples": len(recoveries),
            }

            # Track best layer
            if mean_recovery > best_recovery:
                best_recovery = mean_recovery
                best_layer = layer

            # Visual bar
            bar_len = int(mean_recovery / 5)  # Scale: 5% per char
            bar = "█" * max(0, bar_len)
            print(f"Layer {layer:2d}: {mean_recovery:6.1f}% ± {std_recovery:.1f}  {bar}")

    summary["best_layer"] = best_layer
    summary["best_recovery"] = best_recovery

    print(f"\nBest layer: {best_layer} ({best_recovery:.1f}% recovery)")

    # Expected result comparison
    print("\n" + "=" * 60)
    print("COMPARISON TO WYNROE (DeepSeek-R1)")
    print("=" * 60)
    print("Wynroe found: Layer 20 is critical (sharp spike)")
    print(f"We found: Layer {best_layer} is best ({best_recovery:.1f}% recovery)")

    if best_layer and best_recovery > 50:
        print("✅ Results suggest localized error-detection circuitry")
    else:
        print("⚠️ Results suggest distributed processing (no clear critical layer)")

    # Save results
    output_file = output_dir / f"{args.model}_patching_results.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_file}")

    # Save detailed results
    detailed_file = output_dir / f"{args.model}_patching_detailed.json"
    # Convert numpy/tensor values to Python types for JSON
    for result in all_results:
        result["layer_results"] = {
            str(k): {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                     for kk, vv in v.items() if kk != "patched_logits"}
            for k, v in result["layer_results"].items()
        }
        result["baseline_diff"] = float(result["baseline_diff"])
        del result["clean_logits"]
        del result["corrupted_logits"]

    with open(detailed_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detailed results to {detailed_file}")


if __name__ == "__main__":
    main()
