#!/usr/bin/env python3
"""Extract Q&A pairs from trajectory HDF5 files for verification.

Usage:
    # Print first 5 samples
    python extract_qa.py data/trajectories/olmo3_sft/gsm8k_trajectories.h5 -n 5

    # Save to file
    python extract_qa.py data/trajectories/olmo3_sft/gsm8k_trajectories.h5 -o output.md

    # Only correct samples
    python extract_qa.py data/trajectories/olmo3_sft/gsm8k_trajectories.h5 --correct-only

    # Only incorrect samples
    python extract_qa.py data/trajectories/olmo3_sft/gsm8k_trajectories.h5 --incorrect-only

    # Specific sample indices
    python extract_qa.py data/trajectories/olmo3_sft/gsm8k_trajectories.h5 --indices 0,5,10

    # Summary only (no full outputs)
    python extract_qa.py data/trajectories/olmo3_sft/gsm8k_trajectories.h5 --summary
"""

import argparse
import h5py
import sys
from pathlib import Path


def decode_bytes(value):
    """Decode bytes to string if needed."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def truncate(text, max_len=2000):
    """Truncate text with indicator."""
    if len(text) > max_len:
        return text[:max_len] + f"\n... [TRUNCATED - {len(text)} chars total]"
    return text


def extract_qa(filepath, n_samples=None, correct_only=False, incorrect_only=False,
               indices=None, summary_only=False, max_output_len=2000, output_file=None):
    """Extract Q&A pairs from trajectory file."""

    try:
        f = h5py.File(filepath, 'r')
    except Exception as e:
        print(f"ERROR: Cannot open file: {e}")
        return None

    # Get metadata
    total_samples = len(f['is_correct'])
    correct_count = sum(f['is_correct'][:])
    incorrect_count = total_samples - correct_count

    # Determine which samples to show
    if indices:
        sample_indices = [int(i) for i in indices.split(',')]
    else:
        sample_indices = list(range(total_samples))

    # Filter by correctness
    if correct_only:
        sample_indices = [i for i in sample_indices if f['is_correct'][i]]
    elif incorrect_only:
        sample_indices = [i for i in sample_indices if not f['is_correct'][i]]

    # Limit number of samples
    if n_samples and n_samples < len(sample_indices):
        sample_indices = sample_indices[:n_samples]

    # Build output
    lines = []

    # Header
    fname = Path(filepath).name
    lines.append(f"# Q&A Extraction: {fname}")
    lines.append("")
    lines.append(f"**File**: `{filepath}`")
    lines.append(f"**Total samples**: {total_samples}")
    lines.append(f"**Correct**: {correct_count} ({100*correct_count/total_samples:.1f}%)")
    lines.append(f"**Incorrect**: {incorrect_count} ({100*incorrect_count/total_samples:.1f}%)")
    lines.append("")

    if 'trajectories' in f:
        traj_shape = f['trajectories'].shape
        lines.append(f"**Trajectory shape**: {traj_shape}")
        lines.append("")

    if summary_only:
        lines.append("---")
        lines.append("")
        lines.append("## Sample Correctness Overview")
        lines.append("")
        lines.append("| Index | Correct | Output Length |")
        lines.append("|-------|---------|---------------|")
        for i in sample_indices[:50]:  # Limit to 50 for summary
            is_correct = "✅" if f['is_correct'][i] else "❌"
            output_len = len(decode_bytes(f['model_outputs'][i]))
            lines.append(f"| {i} | {is_correct} | {output_len} |")
        if len(sample_indices) > 50:
            lines.append(f"| ... | ... | ... |")
            lines.append(f"| ({len(sample_indices)} total) | | |")
    else:
        lines.append("---")
        lines.append("")

        for idx in sample_indices:
            is_correct = f['is_correct'][idx]
            prompt = decode_bytes(f['prompts'][idx])
            output = decode_bytes(f['model_outputs'][idx])
            ground_truth = decode_bytes(f['ground_truth'][idx])

            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"

            lines.append(f"## Sample {idx} - {status}")
            lines.append("")
            lines.append("### Prompt")
            lines.append("```")
            lines.append(truncate(prompt, max_output_len))
            lines.append("```")
            lines.append("")
            lines.append("### Model Output")
            lines.append("```")
            lines.append(truncate(output, max_output_len))
            lines.append("```")
            lines.append("")
            lines.append("### Ground Truth")
            lines.append("```")
            lines.append(truncate(ground_truth, 1000))
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    f.close()

    result = "\n".join(lines)

    # Output
    if output_file:
        with open(output_file, 'w') as out:
            out.write(result)
        print(f"Saved to: {output_file}")
    else:
        print(result)

    return result


def main():
    parser = argparse.ArgumentParser(description='Extract Q&A pairs from trajectory HDF5 files')
    parser.add_argument('filepath', help='Path to HDF5 trajectory file')
    parser.add_argument('-n', '--n-samples', type=int, help='Number of samples to show')
    parser.add_argument('-o', '--output', help='Output file (default: print to stdout)')
    parser.add_argument('--correct-only', action='store_true', help='Only show correct samples')
    parser.add_argument('--incorrect-only', action='store_true', help='Only show incorrect samples')
    parser.add_argument('--indices', help='Comma-separated list of sample indices')
    parser.add_argument('--summary', action='store_true', help='Show summary only, no full outputs')
    parser.add_argument('--max-len', type=int, default=2000, help='Max length for outputs (default: 2000)')

    args = parser.parse_args()

    if args.correct_only and args.incorrect_only:
        print("ERROR: Cannot use both --correct-only and --incorrect-only")
        sys.exit(1)

    extract_qa(
        args.filepath,
        n_samples=args.n_samples,
        correct_only=args.correct_only,
        incorrect_only=args.incorrect_only,
        indices=args.indices,
        summary_only=args.summary,
        max_output_len=args.max_len,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
