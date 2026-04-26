#!/usr/bin/env python3
"""Fix correctness labels for models that don't use #### format."""

import h5py
import re
import argparse
from pathlib import Path


def extract_gsm8k_answer_robust(text: str) -> str:
    """Extract answer using multiple patterns, handling think model output."""
    if not text:
        return ""

    # Pattern 1: Standard GSM8K format #### <number>
    match = re.search(r"####\s*(\-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Pattern 2: "the answer is/should be <number>"
    match = re.search(
        r"(?:the\s+)?answer\s+(?:is|should\s+be|would\s+be)[:\s]*\$?(\-?\d+(?:,\d{3})*(?:\.\d+)?)",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).replace(",", "")

    # Pattern 3: "= <number>" at end of text (last occurrence)
    matches = list(re.finditer(r"=\s*\$?(\-?\d+(?:,\d{3})*(?:\.\d+)?)", text))
    if matches:
        return matches[-1].group(1).replace(",", "")

    # Pattern 4: Final number in the response (last number mentioned)
    matches = list(re.finditer(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\b", text))
    if matches:
        # Get last number, but filter out small numbers that might be part of reasoning
        for m in reversed(matches):
            num = m.group(1).replace(",", "")
            if len(num) > 1 or int(float(num)) > 5:  # Skip very small numbers
                return num

    return ""


def extract_ground_truth_answer(text: str) -> str:
    """Extract answer from ground truth (always has #### format)."""
    match = re.search(r"####\s*(\-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    return ""


def fix_labels(h5_path: str, dry_run: bool = True):
    """Fix correctness labels in HDF5 file."""
    print(f"\n{'='*60}")
    print(f"Processing: {h5_path}")
    print(f"{'='*60}")

    mode = "r" if dry_run else "r+"
    with h5py.File(h5_path, mode) as f:
        n_samples = len(f["is_correct"])
        original_correct = sum(f["is_correct"][:])

        new_labels = []
        changes = []

        for i in range(n_samples):
            model_output = f["model_outputs"][i]
            ground_truth = f["ground_truth"][i]

            # Decode if bytes
            if isinstance(model_output, bytes):
                model_output = model_output.decode()
            if isinstance(ground_truth, bytes):
                ground_truth = ground_truth.decode()

            # Extract answers
            pred_answer = extract_gsm8k_answer_robust(model_output)
            true_answer = extract_ground_truth_answer(ground_truth)

            # Compare
            is_correct = pred_answer == true_answer if pred_answer and true_answer else False
            old_correct = bool(f["is_correct"][i])

            new_labels.append(is_correct)

            if is_correct != old_correct:
                changes.append({
                    "idx": i,
                    "old": old_correct,
                    "new": is_correct,
                    "pred": pred_answer,
                    "true": true_answer
                })

        new_correct = sum(new_labels)

        print(f"Original: {original_correct}/500 correct ({100*original_correct/500:.1f}%)")
        print(f"Fixed:    {new_correct}/500 correct ({100*new_correct/500:.1f}%)")
        print(f"Changes:  {len(changes)} labels modified")

        if changes:
            print(f"\nSample changes:")
            for c in changes[:5]:
                print(f"  Sample {c['idx']}: {c['old']} -> {c['new']} (pred={c['pred']}, true={c['true']})")

        if not dry_run and changes:
            print("\nWriting fixed labels...")
            f["is_correct"][:] = new_labels
            print("Done!")
        elif dry_run:
            print("\n[DRY RUN - no changes written]")

    return new_correct, len(changes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/trajectories_8shot")
    parser.add_argument("--model", default=None, help="Specific model to fix (default: all)")
    parser.add_argument("--apply", action="store_true", help="Actually apply changes (default: dry run)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.model:
        models = [args.model]
    else:
        models = [d.name for d in data_dir.iterdir() if d.is_dir()]

    print(f"Mode: {'APPLY CHANGES' if args.apply else 'DRY RUN'}")

    results = {}
    for model in sorted(models):
        h5_path = data_dir / model / "gsm8k_trajectories_8shot.h5"
        if h5_path.exists():
            new_correct, n_changes = fix_labels(str(h5_path), dry_run=not args.apply)
            results[model] = {"correct": new_correct, "changes": n_changes}

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, r in results.items():
        print(f"  {model}: {r['correct']}/500 ({100*r['correct']/500:.1f}%) - {r['changes']} changes")


if __name__ == "__main__":
    main()
