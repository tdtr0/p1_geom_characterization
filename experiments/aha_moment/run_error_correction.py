#!/usr/bin/env python3
"""
Experiment C: Active Error Correction

Test whether different OLMo models can ACTIVELY correct arithmetic errors
when given a corrupted solution prefix.

Key question: Do think-trained models (rl_zero, think) correct errors
that base models simply propagate?

Methodology:
1. Generate CoT solutions with rl_zero (produces calculations we can corrupt)
2. Corrupt the last calculation (e.g., $430 + $320 = $751 instead of $750)
3. Truncate before final answer
4. Let each model continue and check if it outputs correct or corrupted answer

Usage:
    python run_error_correction.py \
        --n_problems 100 \
        --models base,sft,rl_zero,think \
        --output experiments/aha_moment/results/error_correction/

GPU Requirements: ~1-2 hours for 100 problems × 4 models
"""

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Model configurations (7B versions)
MODEL_CONFIGS = {
    'base': 'allenai/OLMo-3-1025-7B',  # Base model (no thinking training)
    'sft': 'allenai/OLMo-3-7B-Think-SFT',
    'rl_zero': 'allenai/OLMo-3-7B-RL-Zero-General',
    'think': 'allenai/OLMo-3-7B-Think',
}

# Generation model (produces structured calculations)
GENERATION_MODEL = 'rl_zero'


def load_gsm8k(n_problems: int, seed: int = 42) -> list[dict]:
    """Load GSM8K problems with answers."""
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))[:n_problems]

    problems = []
    for idx in indices:
        item = dataset[int(idx)]
        # Extract #### answer
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', item['answer'])
        if match:
            answer_num = float(match.group(1).replace(',', ''))
            problems.append({
                'id': f'gsm8k_{idx}',
                'question': item['question'],
                'answer': answer_num,
            })

    return problems


def generate_cot(model, tokenizer, question: str, max_new_tokens: int = 512, device: str = 'cuda') -> str:
    """Generate CoT response."""
    prompt = f"Question: {question}\nLet me solve this step by step.\n"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()


def parse_calculations(text: str) -> list[dict]:
    """Find calculation patterns like: $430 + $320 = $750 or 4 + 2 = 6"""
    pattern = r'([\$]?[\d,]+(?:\.\d+)?\s*[\+\-\*\/×÷]\s*[\$]?[\d,]+(?:\.\d+)?)\s*=\s*([\$]?[\d,]+(?:\.\d+)?)'
    annotations = []

    for match in re.finditer(pattern, text):
        expr = match.group(1).strip()
        result = match.group(2).strip()
        result_num = result.replace('$', '').replace(',', '')

        try:
            float(result_num)
            annotations.append({
                'expr': expr,
                'result': result,
                'result_num': result_num,
                'start': match.start(),
                'end': match.end(),
            })
        except ValueError:
            continue

    return annotations


def corrupt_trace(text: str, annotations: list[dict]) -> tuple[str, float, float]:
    """
    Corrupt the last calculation by adding 1 to the result.

    Returns: (corrupted_text, original_num, corrupted_num)
    """
    if not annotations:
        return None, None, None

    last = annotations[-1]
    try:
        orig_num = float(last['result_num'])
        # Corrupt by adding 1
        corrupt_num = orig_num + 1 if orig_num >= 0 else orig_num - 1

        # Format corrupted result (preserve $ if present)
        if last['result'].startswith('$'):
            corrupt_str = f"${int(corrupt_num)}" if corrupt_num == int(corrupt_num) else f"${corrupt_num}"
        else:
            corrupt_str = str(int(corrupt_num)) if corrupt_num == int(corrupt_num) else str(corrupt_num)

        # Find where result starts in the match
        eq_pos = text[last['start']:last['end']].find('=')
        result_start = last['start'] + eq_pos + 1
        # Skip whitespace
        while result_start < last['end'] and text[result_start] in ' \t':
            result_start += 1

        # Replace
        prefix = text[:result_start]
        suffix = text[result_start + len(last['result']):]
        corrupted = prefix + corrupt_str + suffix

        return corrupted, orig_num, corrupt_num

    except Exception:
        return None, None, None


def truncate_before_answer(text: str, question: str) -> str:
    """
    Truncate trace before the final answer, leaving the model to complete.

    We want to stop just before the model would output the final number,
    so we can see if it corrects or propagates.
    """
    full_text = f"Question: {question}\nLet me solve this step by step.\n{text}"

    # Find patterns that indicate final answer is coming
    patterns = [
        r'####\s*$',  # GSM8K format (truncate before number)
        r'the answer is\s*$',
        r'Therefore,?\s*(the answer is)?\s*$',
        r'So,?\s*(the answer is)?\s*$',
        r'Thus,?\s*(the answer is)?\s*$',
        r'final answer:?\s*$',
    ]

    # Also check for #### followed by number - truncate before the number
    match = re.search(r'####\s*\d', full_text)
    if match:
        # Truncate just before the number
        return full_text[:match.end()-1].rstrip()

    # Check other patterns
    for pattern in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            return full_text[:match.end()].rstrip()

    # Fallback: truncate after last sentence before answer-like content
    # Find last calculation
    calcs = parse_calculations(text)
    if calcs:
        last_calc = calcs[-1]
        # Find sentence end after this calculation
        after_calc = text[last_calc['end']:]
        sentence_end = re.search(r'[.!?]\s', after_calc)
        if sentence_end:
            truncate_pos = last_calc['end'] + sentence_end.end()
            return f"Question: {question}\nLet me solve this step by step.\n{text[:truncate_pos]}"

    # Last resort: just use full text (model will continue from wherever)
    return full_text


def extract_final_answer(text: str) -> float:
    """Extract the final numerical answer from generated text."""
    # Look for #### pattern first
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return float(match.group(1).replace(',', ''))

    # Look for "answer is X" pattern
    match = re.search(r'(?:answer is|answer:)\s*\$?(-?\d+(?:,\d+)*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))

    # Look for last number in text
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return float(numbers[-1].replace(',', ''))

    return None


def run_experiment(
    n_problems: int,
    model_names: list[str],
    output_dir: str,
    max_cot_tokens: int = 512,
    max_continuation_tokens: int = 100,
    device: str = 'cuda',
    seed: int = 42,
):
    """Run the active error correction experiment."""
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load problems
    print(f"\n=== Step 1: Loading {n_problems} GSM8K problems ===")
    problems = load_gsm8k(n_problems, seed)
    print(f"Loaded {len(problems)} problems")

    # Step 2: Generate CoT traces and create corrupted prefixes
    print(f"\n=== Step 2: Generating CoT traces with {GENERATION_MODEL} ===")

    gen_model_name = MODEL_CONFIGS[GENERATION_MODEL]
    print(f"Loading {gen_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    gen_model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate and corrupt
    test_cases = []
    for problem in tqdm(problems, desc="Generating CoT"):
        cot = generate_cot(gen_model, tokenizer, problem['question'], max_cot_tokens, device)
        annotations = parse_calculations(cot)

        if not annotations:
            continue

        corrupted, orig_num, corrupt_num = corrupt_trace(cot, annotations)
        if corrupted is None:
            continue

        # Create truncated prefix
        prefix = truncate_before_answer(corrupted, problem['question'])

        test_cases.append({
            'problem_id': problem['id'],
            'question': problem['question'],
            'ground_truth': problem['answer'],
            'original_value': orig_num,
            'corrupted_value': corrupt_num,
            'clean_trace': cot,
            'corrupted_trace': corrupted,
            'prefix': prefix,
        })

    print(f"Created {len(test_cases)} test cases with corrupted prefixes")

    # Free generation model
    del gen_model
    torch.cuda.empty_cache()

    # Step 3: Run continuation with each model
    print(f"\n=== Step 3: Testing {len(model_names)} models ===")

    all_results = {}

    for model_key in model_names:
        model_name = MODEL_CONFIGS.get(model_key)
        if model_name is None:
            print(f"Warning: Unknown model {model_key}, skipping")
            continue

        print(f"\n--- Testing: {model_key} ({model_name}) ---")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        results = []
        for case in tqdm(test_cases, desc=f"  {model_key}"):
            try:
                # Tokenize prefix
                inputs = tokenizer(case['prefix'], return_tensors='pt').to(device)

                # Generate continuation
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_continuation_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                continuation = full_output[len(case['prefix']):].strip()

                # Extract predicted answer
                predicted = extract_final_answer(continuation)

                # Determine outcome
                correct = False
                if predicted is not None:
                    # Check if predicted matches ground truth (within tolerance)
                    correct = abs(predicted - case['ground_truth']) < 0.01

                # Check for explicit correction phrases
                explicit_correction = any(phrase in continuation.lower() for phrase in [
                    'wait', 'actually', 'no,', 'correction', 'mistake', 'error',
                    'let me recalculate', 'that\'s wrong', 'should be',
                ])

                results.append({
                    'problem_id': case['problem_id'],
                    'ground_truth': case['ground_truth'],
                    'corrupted_value': case['corrupted_value'],
                    'predicted': predicted,
                    'correct': correct,
                    'explicit_correction': explicit_correction,
                    'continuation': continuation[:500],  # Truncate for storage
                })

            except Exception as e:
                print(f"  Error on {case['problem_id']}: {e}")
                results.append({
                    'problem_id': case['problem_id'],
                    'ground_truth': case['ground_truth'],
                    'corrupted_value': case['corrupted_value'],
                    'predicted': None,
                    'correct': False,
                    'explicit_correction': False,
                    'error': str(e),
                })

        all_results[model_key] = results

        # Print summary for this model
        n_correct = sum(1 for r in results if r['correct'])
        n_explicit = sum(1 for r in results if r['explicit_correction'])
        print(f"  Correction rate: {n_correct}/{len(results)} ({100*n_correct/len(results):.1f}%)")
        print(f"  Explicit corrections: {n_explicit}/{len(results)} ({100*n_explicit/len(results):.1f}%)")

        # Free model
        del model
        torch.cuda.empty_cache()

    # Step 4: Save results
    print(f"\n=== Step 4: Saving results ===")

    # Summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_test_cases': len(test_cases),
        'models': {},
    }

    for model_key, results in all_results.items():
        n_total = len(results)
        n_correct = sum(1 for r in results if r['correct'])
        n_explicit = sum(1 for r in results if r['explicit_correction'])
        n_propagated = sum(1 for r in results if r['predicted'] == r['corrupted_value'])

        summary['models'][model_key] = {
            'n_total': n_total,
            'n_correct': n_correct,
            'n_explicit_correction': n_explicit,
            'n_propagated_error': n_propagated,
            'correction_rate': n_correct / n_total if n_total > 0 else 0,
            'explicit_rate': n_explicit / n_total if n_total > 0 else 0,
            'propagation_rate': n_propagated / n_total if n_total > 0 else 0,
        }

    # Save summary
    summary_file = Path(output_dir) / 'correction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    details_file = Path(output_dir) / 'correction_details.json'
    with open(details_file, 'w') as f:
        json.dump({
            'test_cases': [{k: v for k, v in c.items() if k != 'prefix'} for c in test_cases],
            'results': all_results,
        }, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS: Active Error Correction")
    print("=" * 60)

    print(f"\nTest cases: {len(test_cases)}")
    print("\n| Model | Correction Rate | Explicit Corrections | Error Propagation |")
    print("|-------|-----------------|---------------------|-------------------|")

    for model_key in model_names:
        if model_key in summary['models']:
            s = summary['models'][model_key]
            print(f"| {model_key:5} | {s['correction_rate']:14.1%} | {s['explicit_rate']:19.1%} | {s['propagation_rate']:17.1%} |")

    print(f"\nResults saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Experiment C: Active Error Correction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--n_problems', '-n',
        type=int,
        default=100,
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
        default='experiments/aha_moment/results/error_correction/',
        help='Output directory'
    )
    parser.add_argument(
        '--max_cot_tokens',
        type=int,
        default=512,
        help='Max tokens for initial CoT generation'
    )
    parser.add_argument(
        '--max_continuation_tokens',
        type=int,
        default=100,
        help='Max tokens for model continuation'
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

    run_experiment(
        n_problems=args.n_problems,
        model_names=model_names,
        output_dir=args.output,
        max_cot_tokens=args.max_cot_tokens,
        max_continuation_tokens=args.max_continuation_tokens,
        device=args.device,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
