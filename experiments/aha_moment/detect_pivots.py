#!/usr/bin/env python3
"""
Detect pivot tokens (self-correction points) in Phase 2 trajectory data.

This script reads Phase 2 HDF5 files from B2 storage, detects pivot tokens
in the model_outputs text, and aligns them to trajectory token positions.

Usage:
    python detect_pivots.py \
        --input data/phase2/*.h5 \
        --method regex \
        --output data/pivot_labels.json

Methods:
    - regex: Fast pattern matching (default)
    - zero-shot: Use BART-large-mnli for semantic filtering (slower, more accurate)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm

# Try to import transformers for zero-shot classification (optional)
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ============================================================================
# Pivot Patterns
# ============================================================================

# Strong pivot indicators (high precision)
STRONG_PIVOT_PATTERNS = [
    (r'\bBUT WAIT\b', 'BUT_WAIT'),
    (r'\bWait,\s*(no|that|I|let)', 'Wait_correction'),
    (r'\bActually,\s*(no|that|I|let)', 'Actually_correction'),
    (r'\bI was wrong\b', 'I_was_wrong'),
    (r'\blet me reconsider\b', 'reconsider'),
    (r'\bno,\s*that\'s (not|wrong)', 'no_thats_wrong'),
    (r'\bHmm,\s*(wait|but|actually|no)', 'Hmm_correction'),
]

# Weak pivot indicators (need filtering)
WEAK_PIVOT_PATTERNS = [
    (r'\bBUT\b', 'BUT'),
    (r'\bWait\b', 'Wait'),
    (r'\bwait\b', 'wait'),
    (r'\bactually\b', 'actually'),
    (r'\bActually\b', 'Actually'),
    (r'\bhowever\b', 'however'),
    (r'\bHowever\b', 'However'),
    (r'\bhmm\b', 'hmm'),
    (r'\bHmm\b', 'Hmm'),
]


def detect_pivots_regex(text: str, use_strong_only: bool = False) -> list[dict]:
    """
    Detect pivot positions using regex patterns.

    Args:
        text: The model output text
        use_strong_only: If True, only use high-precision patterns

    Returns:
        List of pivot dictionaries with char_pos and pattern
    """
    pivots = []
    patterns = STRONG_PIVOT_PATTERNS if use_strong_only else STRONG_PIVOT_PATTERNS + WEAK_PIVOT_PATTERNS

    for pattern, label in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            pivots.append({
                'char_pos': match.start(),
                'char_end': match.end(),
                'pattern': label,
                'matched_text': match.group(),
            })

    # Sort by position and deduplicate overlapping matches
    pivots.sort(key=lambda x: x['char_pos'])

    # Remove overlapping matches (keep first/strongest)
    deduplicated = []
    last_end = -1
    for p in pivots:
        if p['char_pos'] >= last_end:
            deduplicated.append(p)
            last_end = p['char_end']

    return deduplicated


def detect_pivots_zero_shot(text: str, candidates: list[dict]) -> list[dict]:
    """
    Filter pivot candidates using zero-shot classification.

    Args:
        text: The model output text
        candidates: List of candidate pivots from regex detection

    Returns:
        Filtered list of pivots
    """
    try:
        from transformers import pipeline
    except ImportError:
        print("Warning: transformers not installed, falling back to regex-only")
        return candidates

    # Load classifier (cached after first call)
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,  # CPU
    )

    filtered = []
    for p in tqdm(candidates, desc="Zero-shot filtering", leave=False):
        # Get context window around pivot
        start = max(0, p['char_pos'] - 50)
        end = min(len(text), p['char_end'] + 50)
        window = text[start:end]

        result = classifier(
            window,
            candidate_labels=["self-correction or reconsideration", "normal continuation"],
        )

        if result['labels'][0] == "self-correction or reconsideration":
            p['zero_shot_score'] = result['scores'][0]
            filtered.append(p)

    return filtered


def align_char_to_token(text: str, char_pos: int, tokenizer) -> int:
    """
    Convert character position to token index.

    Args:
        text: Full text
        char_pos: Character position in text
        tokenizer: HuggingFace tokenizer

    Returns:
        Token index corresponding to the character position
    """
    # Tokenize with offset mapping
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    # Find which token contains this character
    for idx, (start, end) in enumerate(encoding.offset_mapping):
        if start <= char_pos < end:
            return idx

    # If not found, return closest token
    return len(encoding.offset_mapping) - 1


def detect_pivots_in_sample(
    text: str,
    tokenizer,
    method: str = 'regex',
    use_strong_only: bool = False,
) -> list[dict]:
    """
    Detect and align pivots for a single sample.

    Args:
        text: Model output text
        tokenizer: HuggingFace tokenizer for alignment
        method: Detection method ('regex' or 'zero-shot')
        use_strong_only: Only use high-precision regex patterns

    Returns:
        List of pivot dictionaries with token_idx
    """
    # Step 1: Regex detection
    candidates = detect_pivots_regex(text, use_strong_only=use_strong_only)

    # Step 2: Optional zero-shot filtering
    if method == 'zero-shot' and candidates:
        candidates = detect_pivots_zero_shot(text, candidates)

    # Step 3: Align to token positions
    for p in candidates:
        p['token_idx'] = align_char_to_token(text, p['char_pos'], tokenizer)

    return candidates


# ============================================================================
# Phase 2 HDF5 Processing
# ============================================================================

def process_phase2_file(
    input_path: str,
    tokenizer,
    method: str = 'regex',
    use_strong_only: bool = False,
) -> dict:
    """
    Process a Phase 2 HDF5 file and detect pivots.

    Args:
        input_path: Path to HDF5 file
        tokenizer: HuggingFace tokenizer
        method: Detection method
        use_strong_only: Only use high-precision patterns

    Returns:
        Dictionary with pivot info for each sample
    """
    results = {
        'file': str(input_path),
        'samples': [],
        'stats': {
            'n_samples': 0,
            'n_with_pivots': 0,
            'total_pivots': 0,
            'pivots_by_pattern': {},
        }
    }

    with h5py.File(input_path, 'r') as f:
        # Get model outputs
        if 'model_outputs' not in f:
            print(f"Warning: {input_path} has no model_outputs dataset")
            return results

        model_outputs = f['model_outputs'][:]

        # Get correctness if available
        correctness = None
        if 'correctness' in f:
            correctness = f['correctness'][:]

        n_samples = len(model_outputs)
        results['stats']['n_samples'] = n_samples

        for i in tqdm(range(n_samples), desc=f"Processing {Path(input_path).name}"):
            # Decode text if bytes
            text = model_outputs[i]
            if isinstance(text, bytes):
                text = text.decode('utf-8')

            # Skip empty outputs
            if not text or len(text.strip()) < 10:
                results['samples'].append({
                    'sample_idx': i,
                    'pivots': [],
                    'is_correct': bool(correctness[i]) if correctness is not None else None,
                })
                continue

            # Detect pivots
            pivots = detect_pivots_in_sample(
                text, tokenizer, method=method, use_strong_only=use_strong_only
            )

            # Track stats
            if pivots:
                results['stats']['n_with_pivots'] += 1
                results['stats']['total_pivots'] += len(pivots)
                for p in pivots:
                    pattern = p['pattern']
                    results['stats']['pivots_by_pattern'][pattern] = \
                        results['stats']['pivots_by_pattern'].get(pattern, 0) + 1

            results['samples'].append({
                'sample_idx': i,
                'pivots': pivots,
                'is_correct': bool(correctness[i]) if correctness is not None else None,
                'text_length': len(text),
            })

    return results


def load_tokenizer(model_name: str = 'allenai/OLMo-3-1B-0125'):
    """Load tokenizer for OLMo models."""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library required. Install with: pip install transformers")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Detect pivot tokens in Phase 2 trajectory data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        nargs='+',
        required=True,
        help='Input HDF5 file(s) from Phase 2 collection'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/pivot_labels.json',
        help='Output JSON file with pivot labels'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['regex', 'zero-shot'],
        default='regex',
        help='Detection method (default: regex)'
    )
    parser.add_argument(
        '--strong-only',
        action='store_true',
        help='Only use high-precision pivot patterns'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='allenai/OLMo-3-1B-0125',
        help='Tokenizer to use for alignment (default: OLMo-3-1B)'
    )

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # Process all input files
    all_results = {}

    for input_path in args.input:
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"Warning: {input_path} does not exist, skipping")
            continue

        result = process_phase2_file(
            str(input_path),
            tokenizer,
            method=args.method,
            use_strong_only=args.strong_only,
        )

        # Use relative key
        key = input_path.name
        all_results[key] = result

        # Print stats
        stats = result['stats']
        print(f"\n{key}:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  With pivots: {stats['n_with_pivots']} ({100*stats['n_with_pivots']/max(1,stats['n_samples']):.1f}%)")
        print(f"  Total pivots: {stats['total_pivots']}")
        if stats['pivots_by_pattern']:
            print(f"  By pattern: {dict(sorted(stats['pivots_by_pattern'].items(), key=lambda x: -x[1]))}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Summary
    total_samples = sum(r['stats']['n_samples'] for r in all_results.values())
    total_with_pivots = sum(r['stats']['n_with_pivots'] for r in all_results.values())
    total_pivots = sum(r['stats']['total_pivots'] for r in all_results.values())

    print(f"\n=== Summary ===")
    print(f"Files processed: {len(all_results)}")
    print(f"Total samples: {total_samples}")
    print(f"Samples with pivots: {total_with_pivots} ({100*total_with_pivots/max(1,total_samples):.1f}%)")
    print(f"Total pivots detected: {total_pivots}")


if __name__ == '__main__':
    main()
