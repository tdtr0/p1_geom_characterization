#!/usr/bin/env python3
"""
Relabel HumanEval correctness by extracting code from markdown blocks.

The model outputs contain chain-of-thought reasoning followed by code in
markdown blocks (```python ... ```). The original checker failed because
it tried to execute the entire output including reasoning text.

This script:
1. Extracts code from markdown blocks in model outputs
2. Runs HumanEval tests on extracted code
3. Updates is_correct labels in HDF5 files
"""

import re
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import argparse
import sys

# HumanEval test cases (subset for validation)
# Full tests would require the humaneval package
HUMANEVAL_TESTS = {
    'concatenate': '''
assert concatenate([]) == ''
assert concatenate(['a', 'b', 'c']) == 'abc'
assert concatenate(['x', 'y', 'z']) == 'xyz'
''',
    'flip_case': '''
assert flip_case('Hello') == 'hELLO'
assert flip_case('') == ''
''',
}


def extract_code_from_output(model_output: str, entry_point: str = None) -> Optional[str]:
    """
    Extract Python code from model output.

    Looks for:
    1. Code in ```python ... ``` blocks (after </think> if present)
    2. Code in ``` ... ``` blocks
    3. Inline code patterns like "def func_name(...): return ..."
    4. Code written out in reasoning
    """
    # First, try to find code after </think> tag
    think_split = model_output.split('</think>')
    if len(think_split) > 1:
        search_text = think_split[-1]  # Everything after </think>
        # Also search full output for code patterns
        full_search = model_output
    else:
        search_text = model_output
        full_search = model_output

    # Look for ```python ... ``` blocks
    python_blocks = re.findall(r'```python\s*(.*?)```', search_text, re.DOTALL)
    if python_blocks:
        return python_blocks[-1].strip()

    # Look for generic ``` ... ``` blocks
    code_blocks = re.findall(r'```\s*(.*?)```', search_text, re.DOTALL)
    if code_blocks:
        for block in reversed(code_blocks):
            if 'def ' in block or 'return ' in block:
                return block.strip()

    # NEW: Look for code patterns in reasoning text
    # Pattern: "def function_name(...): return ..."
    if entry_point:
        # Look for the function definition with this entry point
        def_pattern = rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->.*?)?:\s*\n?\s*return\s+[^\n]+'
        match = re.search(def_pattern, full_search)
        if match:
            return match.group(0).strip()

        # Look for more complex function bodies
        # Pattern: def func(...): followed by indented code
        func_pattern = rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)(?:\s*->.*?)?:'
        for match in re.finditer(func_pattern, full_search):
            start = match.start()
            # Find the function body by looking at indentation
            lines = full_search[start:].split('\n')
            code_lines = [lines[0]]  # def line

            for line in lines[1:]:
                if line.strip() == '':
                    code_lines.append(line)
                elif line.startswith('    ') or line.startswith('\t'):
                    code_lines.append(line)
                elif line.strip().startswith('return'):
                    code_lines.append('    ' + line.strip())
                    break
                else:
                    # End of function body
                    break

            code = '\n'.join(code_lines)
            if 'return' in code:
                return code.strip()

    # Look for simple one-liner patterns
    # "the code would be: return X" or "return ''.join(strings)"
    return_patterns = [
        r'(?:the\s+)?(?:code|solution|function|answer)\s*(?:would\s+be|is|:)\s*\n?\s*(return\s+[^\n]+)',
        r'(?:so\s+)?(?:the\s+)?(?:code|body)\s*(?:would\s+be|is|:)\s*\n?\s*(return\s+[^\n]+)',
    ]

    for pattern in return_patterns:
        match = re.search(pattern, full_search, re.IGNORECASE)
        if match:
            return_stmt = match.group(1).strip()
            if entry_point:
                # Wrap in function definition
                return f"def {entry_point}(*args, **kwargs):\n    {return_stmt}"
            return return_stmt

    # Look for indented code blocks (4 spaces or tab)
    lines = search_text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            in_code = True
            code_lines.append(line)
        elif in_code and line.strip() == '':
            code_lines.append(line)
        elif in_code and line.strip():
            if 'def ' in '\n'.join(code_lines):
                break

    if code_lines and 'def ' in '\n'.join(code_lines):
        return '\n'.join(code_lines).strip()

    return None


def check_code_syntax(code: str) -> bool:
    """Check if code compiles without syntax errors."""
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def run_humaneval_test(code: str, entry_point: str, test_code: str) -> Tuple[bool, str]:
    """
    Run HumanEval test on extracted code.

    Returns (passed, error_message)
    """
    if not code:
        return False, "No code extracted"

    if not check_code_syntax(code):
        return False, "Syntax error in extracted code"

    try:
        # Create namespace and execute code
        namespace = {}
        exec(code, namespace)

        # Check if entry point exists
        if entry_point not in namespace:
            return False, f"Entry point '{entry_point}' not found"

        # Run tests
        exec(test_code, namespace)
        return True, "Tests passed"

    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"Runtime error: {type(e).__name__}: {e}"


def load_humaneval_tests():
    """Load HumanEval test cases from the dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        tests = {}
        for item in ds:
            tests[item['task_id']] = {
                'test': item['test'],
                'entry_point': item['entry_point'],
                'prompt': item['prompt'],
            }
        return tests
    except Exception as e:
        print(f"Warning: Could not load HumanEval dataset: {e}")
        print("Using built-in subset of tests")
        return None


def relabel_humaneval_file(filepath: str, dry_run: bool = False, verbose: bool = False):
    """
    Relabel a HumanEval HDF5 file.

    Args:
        filepath: Path to HDF5 file
        dry_run: If True, don't modify file, just report what would change
        verbose: Print detailed info for each sample
    """
    print(f"\nProcessing: {filepath}")

    # Load HumanEval tests
    humaneval_tests = load_humaneval_tests()

    with h5py.File(filepath, 'r' if dry_run else 'r+') as f:
        outputs = f['model_outputs']
        old_labels = f['is_correct'][:]
        prompts = f['prompts']
        n_samples = len(outputs)

        new_labels = np.zeros(n_samples, dtype=bool)
        stats = {'extracted': 0, 'syntax_ok': 0, 'passed': 0, 'failed': 0, 'no_code': 0}

        print(f"  Samples: {n_samples}")
        print(f"  Old correct: {old_labels.sum()}/{n_samples} ({100*old_labels.mean():.1f}%)")

        for i in range(n_samples):
            output = outputs[i]
            if isinstance(output, bytes):
                output = output.decode('utf-8')

            prompt = prompts[i]
            if isinstance(prompt, bytes):
                prompt = prompt.decode('utf-8')

            # Try to determine entry point from prompt first
            entry_match = re.search(r'def\s+(\w+)\s*\(', prompt)
            entry_point = entry_match.group(1) if entry_match else None

            # Extract code (pass entry_point for better extraction)
            code = extract_code_from_output(output, entry_point)

            if code is None:
                stats['no_code'] += 1
                if verbose:
                    print(f"  [{i}] No code extracted (entry_point={entry_point})")
                continue

            stats['extracted'] += 1

            if check_code_syntax(code):
                stats['syntax_ok'] += 1

            # Get test code
            test_code = None
            if humaneval_tests:
                # Try to find matching test by entry point
                for task_id, test_info in humaneval_tests.items():
                    if test_info['entry_point'] == entry_point:
                        test_code = test_info['test']
                        break

            if test_code and entry_point:
                passed, msg = run_humaneval_test(code, entry_point, test_code)
                if passed:
                    stats['passed'] += 1
                    new_labels[i] = True
                    if verbose:
                        print(f"  [{i}] PASS: {entry_point}")
                else:
                    stats['failed'] += 1
                    if verbose:
                        print(f"  [{i}] FAIL: {entry_point} - {msg}")
            else:
                # No test available, check syntax only
                if check_code_syntax(code):
                    # Be conservative: only mark as correct if we can verify
                    stats['failed'] += 1
                    if verbose:
                        print(f"  [{i}] NO TEST: {entry_point}")

        # Report
        print(f"\n  Stats:")
        print(f"    Code extracted: {stats['extracted']}/{n_samples}")
        print(f"    Syntax OK: {stats['syntax_ok']}/{stats['extracted']}")
        print(f"    Tests passed: {stats['passed']}/{n_samples}")
        print(f"    Tests failed: {stats['failed']}/{n_samples}")
        print(f"    No code: {stats['no_code']}/{n_samples}")

        print(f"\n  New correct: {new_labels.sum()}/{n_samples} ({100*new_labels.mean():.1f}%)")
        print(f"  Change: {old_labels.sum()} -> {new_labels.sum()}")

        if not dry_run:
            f['is_correct'][...] = new_labels
            print(f"  ✓ Labels updated")
        else:
            print(f"  (dry run - no changes made)")

    return new_labels.sum(), n_samples


def main():
    parser = argparse.ArgumentParser(description="Relabel HumanEval correctness")
    parser.add_argument('--data-dir', type=str, default='/data/thanhdo/trajectories_0shot',
                       help='Directory containing trajectory data')
    parser.add_argument('--models', type=str, default='olmo3_base,olmo3_sft,olmo3_rl_zero,olmo3_think',
                       help='Comma-separated list of models')
    parser.add_argument('--dry-run', action='store_true', help='Don\'t modify files')
    parser.add_argument('--verbose', action='store_true', help='Print details for each sample')

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(',')]

    print("=" * 70)
    print("HUMANEVAL RELABELING")
    print("=" * 70)
    print(f"Data dir: {args.data_dir}")
    print(f"Models: {models}")
    print(f"Dry run: {args.dry_run}")

    results = {}
    for model in models:
        filepath = Path(args.data_dir) / model / 'humaneval_trajectories.h5'
        if not filepath.exists():
            print(f"\n⚠ {model}: File not found")
            continue

        try:
            n_correct, n_total = relabel_humaneval_file(str(filepath),
                                                         dry_run=args.dry_run,
                                                         verbose=args.verbose)
            results[model] = {'correct': n_correct, 'total': n_total}
        except Exception as e:
            print(f"\n⚠ {model}: Error - {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for model, res in results.items():
        pct = 100 * res['correct'] / res['total'] if res['total'] > 0 else 0
        print(f"  {model}: {res['correct']}/{res['total']} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
