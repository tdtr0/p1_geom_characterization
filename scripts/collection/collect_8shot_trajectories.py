#!/usr/bin/env python3
"""
Phase 2b: Collect 8-shot trajectories for GSM8K and LogiQA

This script collects activation trajectories using standard 8-shot CoT prompting
to match OLMo benchmark evaluation setup (lm-evaluation-harness format).

Key differences from 0-shot collection:
- Uses 8 exemplars with chain-of-thought reasoning
- Base models: raw text format
- Instruct models: multi-turn chat template
- GSM8K: "#### <number>" format
- LogiQA: "Answer: A/B/C/D" format

Usage:
    python collect_8shot_trajectories.py --model olmo3_base --task gsm8k
    python collect_8shot_trajectories.py --model olmo3_base --task logiqa
    python collect_8shot_trajectories.py --all --task logiqa  # Run all models
"""

import os
import sys
import re
import h5py
import torch
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
from tqdm import tqdm
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from task_data import prepare_gsm8k, prepare_logiqa

# Configuration
OUTPUT_DIR = Path("data/trajectories_8shot")
CHECKPOINT_DIR = Path("data/checkpoints")
N_SAMPLES = 500
# CRITICAL: 8-shot prompts are ~1600-2000 tokens, must accommodate full prompt!
# OLMo-3 context is 4096. Using 2048 for trajectories + 512 for generation = safe.
MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 512
USE_FLASH_ATTN = True

# Even layers only: [0, 2, 4, ..., 30] = 16 layers
LAYERS_TO_COLLECT = list(range(0, 32, 2))

# Model type mapping for chat template
MODEL_TYPES = {
    'olmo3_base': 'base',
    'olmo3_sft': 'instruct',
    'olmo3_rl_zero': 'base',  # RL-Zero uses base format
    'olmo3_think': 'instruct',
}


def load_models_config():
    """Load model configurations"""
    config_path = Path("configs/models.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get('primary', {})


def load_checkpoint(model_key, task_name, n_shot):
    """Load checkpoint if exists"""
    checkpoint_file = CHECKPOINT_DIR / f"labeled_{model_key}_{task_name}_{n_shot}shot.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed_samples": 0, "n_correct": 0, "n_incorrect": 0}


def save_checkpoint(model_key, task_name, n_shot, completed_samples, n_correct, n_incorrect):
    """Save checkpoint"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"labeled_{model_key}_{task_name}_{n_shot}shot.json"
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "completed_samples": completed_samples,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect
        }, f)


def extract_gsm8k_answer(text: str) -> str:
    """Extract numerical answer from GSM8K response.

    GSM8K uses "#### <number>" format.
    """
    # Try to find #### pattern first (standard format)
    match = re.search(r'####\s*(-?[\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()

    # Try "the answer is X" pattern
    match = re.search(r'(?:the answer is|answer:?)\s*(-?[\d,\.]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '').strip()

    # Try to find the last number in the text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()

    return ""


def check_gsm8k_correct(model_output: str, ground_truth: str) -> bool:
    """Check if GSM8K answer is correct."""
    # Extract ground truth answer
    gt_match = re.search(r'####\s*(-?[\d,\.]+)', ground_truth)
    if not gt_match:
        return False
    gt_answer = gt_match.group(1).replace(',', '').strip()

    # Extract model answer
    model_answer = extract_gsm8k_answer(model_output)

    # Compare (handle floating point)
    try:
        gt_val = float(gt_answer)
        model_val = float(model_answer)
        return abs(gt_val - model_val) < 1e-6
    except (ValueError, TypeError):
        return gt_answer == model_answer


def extract_logiqa_answer(text: str) -> str:
    """Extract A/B/C/D answer from LogiQA response."""
    # Try "Answer: X" pattern first
    match = re.search(r'(?:Answer|ANSWER|answer)[:\s]+([A-Da-d])', text)
    if match:
        return match.group(1).upper()

    # Try standalone letter at end
    match = re.search(r'\b([A-Da-d])\s*[.\):]?\s*$', text)
    if match:
        return match.group(1).upper()

    # Try "The answer is X" pattern
    match = re.search(r'(?:the answer is|I (?:choose|select|pick))\s*([A-Da-d])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for letter followed by period or parenthesis in last line
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1]
        match = re.search(r'\b([A-Da-d])[.\):]', last_line)
        if match:
            return match.group(1).upper()

    # Fallback: first A/B/C/D found
    match = re.search(r'\b([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()

    return ""


def check_logiqa_correct(model_output: str, ground_truth: str) -> bool:
    """Check if LogiQA answer is correct."""
    # ground_truth is the index (0, 1, 2, 3) or letter (A, B, C, D)
    if isinstance(ground_truth, int) or ground_truth.isdigit():
        gt_answer = chr(65 + int(ground_truth))  # 0->A, 1->B, etc.
    else:
        gt_answer = ground_truth.upper().strip()

    model_answer = extract_logiqa_answer(model_output)
    return model_answer == gt_answer


class TrajectoryCollector:
    """Collects trajectories during generation."""

    def __init__(self, model_name: str, layers_to_collect: list):
        self.model_name = model_name
        self.layers_to_collect = layers_to_collect

        print(f"Loading model: {model_name}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            for i in range(n_gpus):
                free, total = torch.cuda.mem_get_info(i)
                print(f"  GPU {i}: {free/1e9:.1f} GB free")

            if n_gpus == 1:
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
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                    ).cuda()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size

        print(f"  Loaded: {self.n_layers} layers, d_model={self.d_model}")

    def generate_with_trajectory(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        max_seq_len: int = 512
    ) -> tuple:
        """Generate response and collect trajectory."""
        # Tokenize - for 8-shot, prompt is long, must keep the END (actual question)
        # Set truncation_side='left' so if truncation happens, we keep the question
        self.tokenizer.truncation_side = 'left'
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_len = inputs['input_ids'].shape[1]

        # Set up hooks
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
                # Collect trajectory for prompt
                _ = self.model(**inputs)

                prompt_trajectory = []
                for layer_idx in self.layers_to_collect:
                    act = layer_outputs[layer_idx][0].numpy().astype(np.float16)
                    prompt_trajectory.append(act)

                trajectory = np.stack(prompt_trajectory, axis=0)
                trajectory = np.transpose(trajectory, (1, 0, 2))

                # Generate response with stop sequences
                output_ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[
                        self.tokenizer.eos_token_id,
                        self.tokenizer.encode("Question:", add_special_tokens=False)[0] if "Question:" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id,
                    ],
                )

                generated_text = self.tokenizer.decode(
                    output_ids[0][prompt_len:],
                    skip_special_tokens=True
                )
        finally:
            for handle in handles:
                handle.remove()

        return generated_text, trajectory

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect_for_model(
    model_key: str,
    model_config: dict,
    task_name: str = 'gsm8k',
    n_samples: int = 500
):
    """Collect 8-shot trajectories for a model."""

    model_type = MODEL_TYPES.get(model_key, 'base')

    # Load checkpoint
    checkpoint = load_checkpoint(model_key, task_name, 8)
    start_idx = checkpoint["completed_samples"]
    n_correct = checkpoint["n_correct"]
    n_incorrect = checkpoint["n_incorrect"]

    if start_idx >= n_samples:
        print(f"  Already completed ({n_samples} samples)")
        print(f"  Correct: {n_correct}, Incorrect: {n_incorrect}")
        return

    # Output file
    output_file = OUTPUT_DIR / model_key / f"{task_name}_trajectories_8shot.h5"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize collector
    collector = TrajectoryCollector(
        model_name=model_config['model_name'],
        layers_to_collect=LAYERS_TO_COLLECT,
    )

    # Prepare task data with 8-shot
    if task_name == 'gsm8k':
        task_data = prepare_gsm8k(
            n_samples=n_samples,
            split='test',
            seed=42,  # Same seed as 0-shot for same problems
            n_shot=8,
            model_type=model_type,
            tokenizer=collector.tokenizer if model_type == 'instruct' else None
        )
        check_correct = check_gsm8k_correct
    elif task_name == 'logiqa':
        task_data = prepare_logiqa(
            n_samples=n_samples,
            split='test',
            seed=42,  # Same seed as 0-shot for same problems
            n_shot=8,
            model_type=model_type,
            tokenizer=collector.tokenizer if model_type == 'instruct' else None
        )
        check_correct = check_logiqa_correct
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Create or open HDF5 file
    if start_idx == 0:
        with h5py.File(output_file, 'w') as f:
            f.create_dataset(
                'trajectories',
                shape=(n_samples, MAX_SEQ_LEN, len(LAYERS_TO_COLLECT), collector.d_model),
                dtype='float16',
                compression='gzip',
                compression_opts=4
            )
            f.create_dataset('sequence_lengths', shape=(n_samples,), dtype='int32')
            f.create_dataset('is_correct', shape=(n_samples,), dtype='bool')
            f.create_dataset('prompts', shape=(n_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
            f.create_dataset('model_outputs', shape=(n_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
            f.create_dataset('ground_truth', shape=(n_samples,), dtype=h5py.string_dtype(encoding='utf-8'))

            f.attrs['model'] = model_key
            f.attrs['task'] = task_name
            f.attrs['n_shot'] = 8
            f.attrs['model_type'] = model_type
            f.attrs['n_samples'] = n_samples
            f.attrs['max_seq_len'] = MAX_SEQ_LEN
            f.attrs['layers'] = LAYERS_TO_COLLECT
            f.attrs['d_model'] = collector.d_model
            f.attrs['collection_date'] = datetime.now().isoformat()

    # Collect trajectories
    print(f"  Collecting {n_samples - start_idx} samples (from {start_idx})...")
    print(f"  Current: {n_correct} correct, {n_incorrect} incorrect")

    with h5py.File(output_file, 'a') as f:
        pbar = tqdm(range(start_idx, n_samples), desc=f"  {model_key}/{task_name}-8shot")

        for i in pbar:
            prompt, answer, metadata = task_data[i]

            try:
                model_output, trajectory = collector.generate_with_trajectory(
                    prompt=prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    max_seq_len=MAX_SEQ_LEN
                )

                is_correct = check_correct(model_output, answer)
                seq_len = trajectory.shape[0]

                # Pad or truncate
                if seq_len > MAX_SEQ_LEN:
                    trajectory = trajectory[:MAX_SEQ_LEN]
                    seq_len = MAX_SEQ_LEN
                elif seq_len < MAX_SEQ_LEN:
                    padding = np.zeros(
                        (MAX_SEQ_LEN - seq_len, len(LAYERS_TO_COLLECT), collector.d_model),
                        dtype=np.float16
                    )
                    trajectory = np.vstack([trajectory, padding])

                # Store
                f['trajectories'][i] = trajectory.astype(np.float16)
                f['sequence_lengths'][i] = seq_len
                f['is_correct'][i] = is_correct
                f['prompts'][i] = prompt[:50000]  # Truncate very long prompts
                f['model_outputs'][i] = model_output[:10000]
                f['ground_truth'][i] = str(answer)[:5000]

                if is_correct:
                    n_correct += 1
                else:
                    n_incorrect += 1

                pbar.set_postfix({'correct': n_correct, 'incorrect': n_incorrect})

            except Exception as e:
                print(f"\n  Error on sample {i}: {e}")
                f['trajectories'][i] = np.zeros(
                    (MAX_SEQ_LEN, len(LAYERS_TO_COLLECT), collector.d_model),
                    dtype=np.float16
                )
                f['sequence_lengths'][i] = 0
                f['is_correct'][i] = False
                f['prompts'][i] = prompt[:50000]
                f['model_outputs'][i] = f"ERROR: {str(e)}"
                f['ground_truth'][i] = str(answer)[:5000]
                n_incorrect += 1

            # Checkpoint every 25 samples
            if (i + 1) % 25 == 0:
                save_checkpoint(model_key, task_name, 8, i + 1, n_correct, n_incorrect)

    # Final checkpoint
    save_checkpoint(model_key, task_name, 8, n_samples, n_correct, n_incorrect)

    # Clean up
    del collector
    torch.cuda.empty_cache()

    file_size_gb = output_file.stat().st_size / 1e9
    print(f"  Completed: {n_correct} correct, {n_incorrect} incorrect")
    print(f"  Accuracy: {100*n_correct/n_samples:.1f}%")
    print(f"  File size: {file_size_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Collect 8-shot trajectories for GSM8K or LogiQA")
    parser.add_argument('--model', type=str, help='Specific model to run')
    parser.add_argument('--task', type=str, default='gsm8k', choices=['gsm8k', 'logiqa'],
                       help='Task to collect (gsm8k or logiqa)')
    parser.add_argument('--all', action='store_true', help='Run all models')
    parser.add_argument('--n-samples', type=int, default=500, help='Number of samples')

    args = parser.parse_args()

    print("=" * 80)
    print(f"PHASE 2b: 8-Shot Trajectory Collection ({args.task.upper()})")
    print("=" * 80)
    print()
    print(f"Task: {args.task}")
    print(f"Layers: Even only [0, 2, ..., 30] = {len(LAYERS_TO_COLLECT)} layers")
    print(f"Samples: {args.n_samples}")
    print(f"Format: 8-shot CoT prompting")
    print()

    models = load_models_config()

    if args.model:
        model_keys = [args.model]
    elif args.all:
        model_keys = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']
    else:
        print("Specify --model <name> or --all")
        return

    for model_key in model_keys:
        if model_key not in models:
            print(f"Model {model_key} not in config, skipping")
            continue

        model_config = models[model_key]
        print(f"\n{'=' * 80}")
        print(f"Model: {model_key} ({model_config['model_name']})")
        print(f"Type: {MODEL_TYPES.get(model_key, 'base')}")
        print(f"{'=' * 80}")

        try:
            collect_for_model(
                model_key=model_key,
                model_config=model_config,
                task_name=args.task,
                n_samples=args.n_samples
            )
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
