#!/usr/bin/env python3
"""Test LogiQA collection with 3 samples and verify outputs.

This script:
1. Collects 3 samples from LogiQA
2. Saves to a test HDF5 file
3. Reads it back and prints full outputs for verification

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./src python scripts/collection/test_logiqa_collection.py olmo3_sft
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import numpy as np
import h5py
from pathlib import Path
import yaml

from task_data import prepare_logiqa

# Test configuration
N_TEST_SAMPLES = 3
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 2048  # Sufficient for most outputs; repetition_penalty prevents loops
LAYERS_TO_COLLECT = list(range(0, 32, 2))
OUTPUT_DIR = Path("data/trajectories")
USE_FLASH_ATTN = True


def load_models_config():
    config_path = Path("configs/models.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get('primary', {})


def extract_logiqa_answer(model_output: str) -> str:
    import re
    patterns = [
        r'(?:answer|choice|option)\s*(?:is|:)?\s*[:\s]*([A-D])',
        r'\b([A-D])\s*(?:is correct|is the answer)',
        r'(?:^|\n)\s*([A-D])\s*[.)\s]',
        r'\\boxed\{([A-D])\}',
        r'\*\*([A-D])\*\*',
    ]
    for pattern in patterns:
        match = re.search(pattern, model_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    matches = re.findall(r'\b([A-D])\b', model_output)
    if matches:
        return matches[-1].upper()
    return ""


def check_logiqa_correct(model_output: str, ground_truth: str) -> bool:
    try:
        gt_idx = int(ground_truth)
        gt_letter = chr(65 + gt_idx)
    except (ValueError, TypeError):
        gt_letter = str(ground_truth).upper()
    model_answer = extract_logiqa_answer(model_output)
    return model_answer == gt_letter


class TestCollector:
    def __init__(self, model_name: str, layers_to_collect: list):
        self.model_name = model_name
        self.layers_to_collect = layers_to_collect

        print(f"Loading model: {model_name}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            print(f"  GPU 0: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

            try:
                if USE_FLASH_ATTN:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                    ).cuda()
                    print("  Using Flash Attention 2")
                else:
                    raise ImportError("Disabled")
            except (ImportError, ValueError) as e:
                print(f"  Flash Attention not available: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                ).cuda()
        else:
            raise RuntimeError("CUDA required")

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size

        print(f"  Loaded: {self.n_layers} layers, d_model={self.d_model}")

    def generate_with_trajectory(self, prompt: str, max_new_tokens: int, max_seq_len: int) -> tuple:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[1]

        layer_outputs = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                layer_outputs[layer_idx] = hidden.detach().cpu()
            return hook

        handles = []
        for layer_idx in self.layers_to_collect:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            with torch.no_grad():
                _ = self.model(**inputs)

                prompt_trajectory = []
                for layer_idx in self.layers_to_collect:
                    act = layer_outputs[layer_idx][0].numpy().astype(np.float16)
                    prompt_trajectory.append(act)

                trajectory = np.stack(prompt_trajectory, axis=0)
                trajectory = np.transpose(trajectory, (1, 0, 2))

                output_ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.2,  # Prevent repetition loops
                )

                generated_text = self.tokenizer.decode(
                    output_ids[0][prompt_len:],
                    skip_special_tokens=True
                )

                # Count tokens generated
                n_tokens_generated = output_ids.shape[1] - prompt_len

        finally:
            for handle in handles:
                handle.remove()

        return generated_text, trajectory, n_tokens_generated

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()


def run_test(model_key: str):
    print("=" * 80)
    print(f"TEST COLLECTION: {model_key}")
    print(f"Samples: {N_TEST_SAMPLES}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print("=" * 80)

    models = load_models_config()
    if model_key not in models:
        print(f"Model {model_key} not found")
        return False

    model_config = models[model_key]

    # Prepare data
    print("\nPreparing LogiQA data...")
    task_data = prepare_logiqa(n_samples=N_TEST_SAMPLES, split='test')
    print(f"Loaded {len(task_data)} samples")

    # Test output file
    output_dir = OUTPUT_DIR / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    test_file = output_dir / "logiqa_TEST.h5"

    # Remove old test file
    if test_file.exists():
        test_file.unlink()

    # Initialize collector
    collector = TestCollector(
        model_name=model_config['model_name'],
        layers_to_collect=LAYERS_TO_COLLECT
    )

    # Collect samples
    results = []

    with h5py.File(test_file, 'w') as f:
        f.create_dataset(
            'trajectories',
            shape=(N_TEST_SAMPLES, MAX_SEQ_LEN, len(LAYERS_TO_COLLECT), collector.d_model),
            dtype='float16',
            compression='gzip',
            compression_opts=4
        )
        f.create_dataset('sequence_lengths', shape=(N_TEST_SAMPLES,), dtype='int32')
        f.create_dataset('is_correct', shape=(N_TEST_SAMPLES,), dtype='bool')
        f.create_dataset('prompts', shape=(N_TEST_SAMPLES,), dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset('model_outputs', shape=(N_TEST_SAMPLES,), dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset('ground_truth', shape=(N_TEST_SAMPLES,), dtype=h5py.string_dtype(encoding='utf-8'))

        for i in range(N_TEST_SAMPLES):
            prompt, answer, metadata = task_data[i]

            print(f"\n{'='*60}")
            print(f"SAMPLE {i}")
            print(f"{'='*60}")

            model_output, trajectory, n_tokens = collector.generate_with_trajectory(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                max_seq_len=MAX_SEQ_LEN
            )

            is_correct = check_logiqa_correct(model_output, answer)
            extracted_answer = extract_logiqa_answer(model_output)
            seq_len = trajectory.shape[0]

            # Pad trajectory
            if seq_len > MAX_SEQ_LEN:
                trajectory = trajectory[:MAX_SEQ_LEN]
                seq_len = MAX_SEQ_LEN
            elif seq_len < MAX_SEQ_LEN:
                padding = np.zeros(
                    (MAX_SEQ_LEN - seq_len, len(LAYERS_TO_COLLECT), collector.d_model),
                    dtype=np.float32
                )
                trajectory = np.vstack([trajectory, padding])

            # Store
            f['trajectories'][i] = trajectory.astype(np.float16)
            f['sequence_lengths'][i] = seq_len
            f['is_correct'][i] = is_correct
            f['prompts'][i] = prompt
            f['model_outputs'][i] = model_output[:50000]
            f['ground_truth'][i] = str(answer)

            # Print results
            try:
                gt_letter = chr(65 + int(answer))
            except:
                gt_letter = answer

            print(f"\nGround truth: {gt_letter}")
            print(f"Extracted answer: {extracted_answer}")
            print(f"Correct: {is_correct}")
            print(f"Tokens generated: {n_tokens}")
            print(f"Output chars: {len(model_output)}")
            print(f"Trajectory shape: {trajectory.shape}")
            print(f"\n--- PROMPT ---")
            print(prompt[:500])
            print(f"\n--- MODEL OUTPUT (first 1000 chars) ---")
            print(model_output[:1000])
            print(f"\n--- MODEL OUTPUT (last 500 chars) ---")
            print(model_output[-500:])

            results.append({
                'correct': is_correct,
                'n_tokens': n_tokens,
                'output_len': len(model_output),
                'extracted': extracted_answer,
                'ground_truth': gt_letter,
            })

    # Cleanup
    del collector
    torch.cuda.empty_cache()

    # Verify file can be read back
    print(f"\n{'='*80}")
    print("VERIFICATION: Reading back HDF5 file")
    print(f"{'='*80}")

    try:
        with h5py.File(test_file, 'r') as f:
            print(f"File: {test_file}")
            print(f"Keys: {list(f.keys())}")
            print(f"Trajectories shape: {f['trajectories'].shape}")
            print(f"is_correct: {list(f['is_correct'][:])}")

            for i in range(N_TEST_SAMPLES):
                print(f"\nSample {i}:")
                print(f"  seq_len: {f['sequence_lengths'][i]}")
                print(f"  is_correct: {f['is_correct'][i]}")
                print(f"  output_len: {len(f['model_outputs'][i].decode())}")
                print(f"  traj non-zero: {np.count_nonzero(f['trajectories'][i])}")

        print("\n✓ HDF5 file verified successfully!")

    except Exception as e:
        print(f"\n✗ ERROR reading HDF5: {e}")
        return False

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    n_correct = sum(1 for r in results if r['correct'])
    print(f"Correct: {n_correct}/{N_TEST_SAMPLES}")
    print(f"Avg tokens: {sum(r['n_tokens'] for r in results) / N_TEST_SAMPLES:.0f}")
    print(f"Avg output chars: {sum(r['output_len'] for r in results) / N_TEST_SAMPLES:.0f}")

    for i, r in enumerate(results):
        status = "✓" if r['correct'] else "✗"
        print(f"  Sample {i}: {status} (extracted={r['extracted']}, gt={r['ground_truth']}, tokens={r['n_tokens']})")

    # Check for truncation
    max_tokens_hit = any(r['n_tokens'] >= MAX_NEW_TOKENS - 10 for r in results)
    if max_tokens_hit:
        print(f"\n⚠️  WARNING: Some outputs may be truncated (hit ~{MAX_NEW_TOKENS} tokens)")
    else:
        print(f"\n✓ No truncation detected (all outputs < {MAX_NEW_TOKENS} tokens)")

    print(f"\nTest file saved to: {test_file}")
    print("If everything looks good, run the full collection.")

    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: CUDA_VISIBLE_DEVICES=X python test_logiqa_collection.py <model_key>")
        sys.exit(1)

    model_key = sys.argv[1]
    success = run_test(model_key)
    sys.exit(0 if success else 1)
