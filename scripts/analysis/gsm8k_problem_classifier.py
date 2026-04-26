#!/usr/bin/env python3
"""
GSM8K Problem Type Classifier

Classifies GSM8K problems by type to enable controlled subset analysis.
Works with existing HDF5 trajectory files that contain question text.

Problem types:
- rate_distance: Speed, rate, distance, time problems
- arithmetic: Pure multi-step arithmetic
- fraction: Fractions, ratios, percentages
- algebra: Algebraic equations, solving for unknowns
- comparison: Comparing quantities
- other: Uncategorized

Usage:
    python scripts/analysis/gsm8k_problem_classifier.py --data-dir data/trajectories

Output:
    - results/gsm8k_problem_types.json: Classification results
    - results/gsm8k_subset_indices.json: Indices for each problem type
"""

import os
import sys
import re
import json
import h5py
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


# Problem type patterns (order matters - first match wins)
PROBLEM_PATTERNS = {
    'rate_distance': [
        r'\b(per hour|miles|kilometers|km|speed|rate|faster|slower)\b',
        r'\b(travel|drove|walk|run|bike|flew)\b.*\b(hour|minute|day)\b',
        r'\b(mile|kilometer)s?\s+(per|each|a)\s+(hour|minute|day)\b',
    ],
    'fraction': [
        r'\b(fraction|ratio|percent|%)\b',
        r'\b\d+/\d+\b',  # Actual fractions like 1/2, 3/4
        r'\b(half|third|quarter|fifth|sixth)\b',
        r'\b(proportion|split|share|divide.*equally)\b',
    ],
    'algebra': [
        r'\b(equation|solve for|find.*value|let\s+\w\s*=)\b',
        r'\b(if\s+\w\s+is|when\s+\w\s+=|where\s+\w\s+represents)\b',
        r'\b(unknown|variable)\b',
    ],
    'comparison': [
        r'\b(more than|less than|difference|how (much|many) more)\b',
        r'\b(compare|comparison|ratio.*between)\b',
        r'\b(greater|smaller|larger|fewer)\b.*\b(than)\b',
    ],
    'counting': [
        r'\b(total|altogether|in all|combined)\b.*\b(how many|count)\b',
        r'\bhow many\b.*\b(total|altogether|in all)\b',
        r'\b(each|every)\b.*\b(gets|receives|has)\b',
    ],
    'time': [
        r'\b(hour|minute|second|day|week|month|year)s?\b.*\b(ago|later|before|after)\b',
        r'\b(schedule|start|finish|begin|end|duration)\b',
        r'\b(clock|time|when)\b',
    ],
    'money': [
        r'\$\d+|\d+\s*dollars?|\d+\s*cents?',
        r'\b(price|cost|pay|spend|earn|save|budget)\b',
        r'\b(buy|sell|purchase|sale|discount)\b',
    ],
    'arithmetic': [
        # Multi-step arithmetic (catch-all for number-heavy problems)
        r'^\s*\S+\s+\S+.*\d+.*\d+.*\d+',  # At least 3 numbers
    ],
}


def classify_problem(question: str) -> Tuple[str, float]:
    """Classify a single problem by type.

    Returns:
        (problem_type, confidence) where confidence is 1.0 for pattern match
    """
    question_lower = question.lower()

    # Check each pattern category
    for problem_type, patterns in PROBLEM_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                return problem_type, 1.0

    return 'other', 0.5


def extract_problem_features(question: str) -> Dict:
    """Extract numerical features from a problem for analysis."""
    # Count numbers
    numbers = re.findall(r'-?\d+\.?\d*', question)

    # Count operations mentioned
    operations = {
        'addition': len(re.findall(r'\b(add|plus|more|total|sum|increase|gain)\b', question, re.I)),
        'subtraction': len(re.findall(r'\b(subtract|minus|less|decrease|lose|remain|left)\b', question, re.I)),
        'multiplication': len(re.findall(r'\b(times|multiply|each|per|every)\b', question, re.I)),
        'division': len(re.findall(r'\b(divide|split|share|half|average)\b', question, re.I)),
    }

    # Estimate complexity
    n_sentences = len(re.split(r'[.!?]', question))
    n_words = len(question.split())

    return {
        'n_numbers': len(numbers),
        'numbers': numbers[:10],  # First 10 numbers
        'operations': operations,
        'n_sentences': n_sentences,
        'n_words': n_words,
    }


def load_prompts_from_hdf5(filepath: str) -> List[str]:
    """Load prompts from HDF5 trajectory file."""
    with h5py.File(filepath, 'r') as f:
        if 'prompts' in f:
            # String dataset
            prompts = [p.decode('utf-8') if isinstance(p, bytes) else p
                       for p in f['prompts'][:]]
            return prompts
        else:
            print(f"Warning: No 'prompts' dataset in {filepath}")
            return []


def classify_dataset(prompts: List[str]) -> Dict:
    """Classify all prompts and return structured results."""
    results = {
        'classifications': [],
        'type_indices': {},
        'type_counts': Counter(),
    }

    for i, prompt in enumerate(prompts):
        # Extract just the question (before any model-specific formatting)
        # GSM8K prompts typically start with "Question:" or just the question text
        question = prompt
        if 'Question:' in prompt:
            question = prompt.split('Question:')[-1].split('Answer:')[0].strip()

        problem_type, confidence = classify_problem(question)
        features = extract_problem_features(question)

        results['classifications'].append({
            'index': i,
            'type': problem_type,
            'confidence': confidence,
            'features': features,
            'question_preview': question[:200],
        })

        results['type_counts'][problem_type] += 1

        if problem_type not in results['type_indices']:
            results['type_indices'][problem_type] = []
        results['type_indices'][problem_type].append(i)

    return results


def print_summary(results: Dict):
    """Print classification summary."""
    print("\n" + "="*60)
    print("PROBLEM TYPE DISTRIBUTION")
    print("="*60)

    total = sum(results['type_counts'].values())
    for problem_type, count in sorted(results['type_counts'].items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = '█' * int(pct / 2)
        print(f"  {problem_type:<15} {count:>4} ({pct:>5.1f}%) {bar}")

    print(f"\n  {'Total':<15} {total:>4}")

    # Show examples for each type
    print("\n" + "="*60)
    print("EXAMPLES BY TYPE (first 2 per type)")
    print("="*60)

    for problem_type in sorted(results['type_indices'].keys()):
        indices = results['type_indices'][problem_type][:2]
        print(f"\n{problem_type.upper()}:")
        for idx in indices:
            clf = results['classifications'][idx]
            preview = clf['question_preview'][:100]
            if len(clf['question_preview']) > 100:
                preview += "..."
            print(f"  [{idx}] {preview}")


def main():
    parser = argparse.ArgumentParser(description='GSM8K Problem Classifier')
    parser.add_argument('--data-dir', type=str, default='data/trajectories',
                        help='Directory containing HDF5 files')
    parser.add_argument('--model', type=str, default='olmo3_base',
                        help='Model subdirectory to use')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--hdf5-file', type=str, default=None,
                        help='Direct path to HDF5 file (overrides --data-dir/--model)')
    args = parser.parse_args()

    # Find HDF5 file
    if args.hdf5_file:
        hdf5_path = Path(args.hdf5_file)
    else:
        hdf5_path = Path(args.data_dir) / args.model / 'gsm8k_trajectories.h5'

    if not hdf5_path.exists():
        print(f"Error: HDF5 file not found: {hdf5_path}")
        print("\nAvailable files:")
        data_dir = Path(args.data_dir)
        if data_dir.exists():
            for f in data_dir.rglob('*.h5'):
                print(f"  {f}")

        # Try to load from existing analysis data instead
        print("\nAttempting to load from task_data module instead...")
        from task_data import prepare_gsm8k
        data = prepare_gsm8k(n_shot=0, n_samples=500, seed=42)
        prompts = [item['prompt'] for item in data]
        print(f"Loaded {len(prompts)} prompts from prepare_gsm8k()")
    else:
        print(f"Loading prompts from: {hdf5_path}")
        prompts = load_prompts_from_hdf5(str(hdf5_path))
        print(f"Loaded {len(prompts)} prompts")

    # Classify
    print("\nClassifying problems...")
    results = classify_dataset(prompts)

    # Print summary
    print_summary(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full classifications
    output_file = output_dir / 'gsm8k_problem_types.json'
    with open(output_file, 'w') as f:
        json.dump({
            'type_counts': dict(results['type_counts']),
            'classifications': results['classifications'],
        }, f, indent=2)
    print(f"\nFull classifications saved to: {output_file}")

    # Save indices by type (for easy loading in analysis scripts)
    indices_file = output_dir / 'gsm8k_subset_indices.json'
    with open(indices_file, 'w') as f:
        json.dump(results['type_indices'], f, indent=2)
    print(f"Subset indices saved to: {indices_file}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR CONTROLLED ANALYSIS")
    print("="*60)

    largest_type = max(results['type_counts'].items(), key=lambda x: x[1])
    print(f"\nLargest homogeneous subset: {largest_type[0]} ({largest_type[1]} samples)")
    print(f"Recommendation: Run analysis on this subset first.")

    if 'money' in results['type_indices'] and len(results['type_indices']['money']) >= 50:
        print(f"\nMoney problems: {len(results['type_indices']['money'])} samples")
        print("  Good candidate - concrete domain, consistent structure.")

    if 'rate_distance' in results['type_indices'] and len(results['type_indices']['rate_distance']) >= 30:
        print(f"\nRate/distance problems: {len(results['type_indices']['rate_distance'])} samples")
        print("  Good candidate - requires d=rt formula, clear structure.")


if __name__ == '__main__':
    main()
