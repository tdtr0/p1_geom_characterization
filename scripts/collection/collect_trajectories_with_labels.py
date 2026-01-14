#!/usr/bin/env python3
"""
Phase 2: Collect trajectories WITH correctness labels

This script:
1. Generates model answers for each problem
2. Checks correctness against ground truth
3. Collects activation trajectories during the forward pass
4. Stores everything with is_correct labels for H1/H2 tests

Correctness checking:
- GSM8K: Extract #### number, exact match
- HumanEval: Run tests, pass@1 (simplified: check if code compiles and produces expected output)
- LogiQA: Extract A/B/C/D, exact match
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
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_data import prepare_gsm8k, prepare_humaneval, prepare_logiqa

# Configuration
OUTPUT_DIR = Path("data/trajectories")
CHECKPOINT_DIR = Path("data/checkpoints")
N_SAMPLES = 500  # Full 500 for correct/incorrect split
BATCH_SIZE = 1  # Process one at a time for trajectory collection
MAX_SEQ_LEN = 512  # Maximum sequence length for trajectories
MAX_NEW_TOKENS = 512  # For generation (need long outputs for CoT)

# Performance optimizations
TORCH_COMPILE = False  # Set True to use torch.compile (slower startup, faster inference)
USE_FLASH_ATTN = True  # Use Flash Attention 2 if available

# Even layers only: [0, 2, 4, ..., 30] = 16 layers
LAYERS_TO_COLLECT = list(range(0, 32, 2))


def load_models_config():
    """Load model configurations - flattens 'primary' section"""
    config_path = Path("configs/models.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Models are nested under 'primary' key
    return config.get('primary', {})


def load_checkpoint(model_key, task_name):
    """Load checkpoint if exists"""
    checkpoint_file = CHECKPOINT_DIR / f"labeled_{model_key}_{task_name}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed_samples": 0, "n_correct": 0, "n_incorrect": 0}


def save_checkpoint(model_key, task_name, completed_samples, n_correct, n_incorrect):
    """Save checkpoint"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"labeled_{model_key}_{task_name}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "completed_samples": completed_samples,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect
        }, f)


# ============================================================================
# Correctness Checking Functions
# ============================================================================

def extract_gsm8k_answer(text: str) -> str:
    """Extract numerical answer from GSM8K response.

    GSM8K answers have format: "#### 42" at the end.
    Model responses should ideally follow this, but we also try other patterns.
    """
    # Try to find #### pattern first
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
    """Check if GSM8K answer is correct.

    ground_truth format: Full solution ending with "#### <number>"
    """
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
    """Extract A/B/C/D from LogiQA response."""
    # Look for explicit "Answer: X" or "The answer is X"
    match = re.search(r'(?:answer(?:\s+is)?:?\s*)([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for standalone letter at end
    match = re.search(r'\b([A-D])\b[^A-Da-z]*$', text)
    if match:
        return match.group(1).upper()

    # Look for first standalone A/B/C/D
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1).upper()

    return ""


def check_logiqa_correct(model_output: str, ground_truth: str) -> bool:
    """Check if LogiQA answer is correct.

    ground_truth: integer index (0-3) mapping to A-D
    """
    # Convert ground truth index to letter
    try:
        gt_idx = int(ground_truth)
        gt_letter = chr(65 + gt_idx)  # 0->A, 1->B, etc.
    except (ValueError, TypeError):
        # Already a letter
        gt_letter = str(ground_truth).upper()

    model_answer = extract_logiqa_answer(model_output)
    return model_answer == gt_letter


def check_humaneval_correct(model_output: str, test_code: str, entry_point: str) -> bool:
    """Check if HumanEval code is correct by running tests.

    WARNING: This is a simplified check that attempts to execute code.
    For production use, this should be run in a sandboxed environment.
    """
    try:
        # Try to compile the code first
        compile(model_output, '<string>', 'exec')
    except SyntaxError:
        return False
    
    # Attempt to execute the code and tests in a namespace
    # This is NOT safe for untrusted code - use sandbox in production
    try:
        namespace = {}
        # Execute the model's code
        exec(model_output, namespace)
        
        # Check if the entry point exists
        if entry_point not in namespace:
            return False
        
        # Execute the test code
        exec(test_code, namespace)
        
        # If we get here without exception, tests passed
        return True
    except Exception:
        # Any exception during execution = incorrect
        return False


def check_correctness(model_output: str, ground_truth: str, task_name: str, metadata: dict = None) -> bool:
    """Unified correctness checker for all tasks."""
    if task_name == "gsm8k":
        return check_gsm8k_correct(model_output, ground_truth)
    elif task_name == "logiqa":
        return check_logiqa_correct(model_output, ground_truth)
    elif task_name == "humaneval":
        # HumanEval needs test code and entry point from metadata
        test_code = metadata.get('test', '') if metadata else ''
        entry_point = metadata.get('entry_point', '') if metadata else ''
        return check_humaneval_correct(model_output, test_code, entry_point)
    else:
        raise ValueError(f"Unknown task: {task_name}")


# ============================================================================
# Model Loading and Generation
# ============================================================================

class TrajectoryCollector:
    """Collects trajectories during generation and provides generation capability."""

    def __init__(self, model_name: str, layers_to_collect: list):
        self.model_name = model_name
        self.layers_to_collect = layers_to_collect

        print(f"Loading model: {model_name}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check available GPU memory
        if torch.cuda.is_available():
            free_mem = []
            n_gpus = torch.cuda.device_count()
            for i in range(n_gpus):
                free, total = torch.cuda.mem_get_info(i)
                free_mem.append(free / 1e9)
                print(f"  GPU {i}: {free/1e9:.1f} GB free")

            total_free = sum(free_mem)
            print(f"  Total GPU memory: {total_free:.1f} GB free")

            # If only 1 GPU visible (parallel mode), use it directly
            # Otherwise use device_map="auto" to split across GPUs
            if n_gpus == 1:
                print(f"  Loading model on single GPU (CUDA:0)")
                # Try to use Flash Attention 2 for faster inference
                try:
                    if USE_FLASH_ATTN:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            attn_implementation="flash_attention_2",
                        ).cuda()
                        print("  Using Flash Attention 2")
                    else:
                        raise ImportError("Flash Attention disabled")
                except (ImportError, ValueError) as e:
                    print(f"  Flash Attention not available: {e}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                    ).cuda()
            elif total_free >= 14:  # 7B model needs ~14GB
                print("  Using device_map='auto' to split model across GPUs")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory={i: f"{int(free_mem[i] * 0.9)}GB" for i in range(len(free_mem))},
                )
            else:
                print("  WARNING: Insufficient GPU memory, using CPU offload")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    offload_folder="offload",
                )
        else:
            print("  WARNING: No CUDA available, using CPU (will be slow)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )

        self.model.eval()
        print(f"  Model device map: {getattr(self.model, 'hf_device_map', 'N/A')}")

        # Get device of first layer for input placement
        self.device = next(self.model.parameters()).device

        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size

        print(f"  Loaded: {self.n_layers} layers, d_model={self.d_model}")
        print(f"  Device: {self.device}")

    def generate_with_trajectory(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        max_seq_len: int = 512
    ) -> tuple:
        """Generate response and collect trajectory for the prompt.

        Returns:
            (generated_text, trajectory)
            trajectory shape: (seq_len, n_layers, d_model)
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_len = inputs['input_ids'].shape[1]

        # Set up hooks to collect activations during generation
        layer_outputs = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                layer_outputs[layer_idx] = hidden.detach().cpu()
            return hook

        # Register hooks for target layers
        handles = []
        for layer_idx in self.layers_to_collect:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            with torch.no_grad():
                # First, collect trajectory for the prompt
                _ = self.model(**inputs)

                # Store prompt trajectory
                prompt_trajectory = []
                for layer_idx in self.layers_to_collect:
                    # Shape: (1, seq_len, d_model) -> (seq_len, d_model)
                    act = layer_outputs[layer_idx][0].numpy().astype(np.float16)
                    prompt_trajectory.append(act)

                # Stack: (n_layers, seq_len, d_model) -> transpose to (seq_len, n_layers, d_model)
                trajectory = np.stack(prompt_trajectory, axis=0)  # (n_layers, seq_len, d_model)
                trajectory = np.transpose(trajectory, (1, 0, 2))  # (seq_len, n_layers, d_model)

                # Now generate response
                output_ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                # Decode generated text (excluding prompt)
                generated_text = self.tokenizer.decode(
                    output_ids[0][prompt_len:],
                    skip_special_tokens=True
                )

        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()

        return generated_text, trajectory

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# Main Collection Pipeline
# ============================================================================

def collect_for_model_task(
    model_key: str,
    model_config: dict,
    task_name: str,
    task_data: list,
    n_samples: int = 500
):
    """Collect trajectories with correctness labels for a model/task pair."""

    # Load checkpoint
    checkpoint = load_checkpoint(model_key, task_name)
    start_idx = checkpoint["completed_samples"]
    n_correct = checkpoint["n_correct"]
    n_incorrect = checkpoint["n_incorrect"]

    if start_idx >= n_samples:
        print(f"  Already completed ({n_samples} samples)")
        print(f"  Correct: {n_correct}, Incorrect: {n_incorrect}")
        return

    # Output file
    output_file = OUTPUT_DIR / model_key / f"{task_name}_trajectories.h5"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize collector (device is determined by model loading)
    collector = TrajectoryCollector(
        model_name=model_config['model_name'],
        layers_to_collect=LAYERS_TO_COLLECT,
    )

    # Create or open HDF5 file
    if start_idx == 0:
        with h5py.File(output_file, 'w') as f:
            # Pre-allocate datasets
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

            # Metadata
            f.attrs['model'] = model_key
            f.attrs['task'] = task_name
            f.attrs['n_samples'] = n_samples
            f.attrs['max_seq_len'] = MAX_SEQ_LEN
            f.attrs['layers'] = LAYERS_TO_COLLECT
            f.attrs['d_model'] = collector.d_model
            f.attrs['collection_date'] = datetime.now().isoformat()

    # Collect trajectories with labels
    print(f"  Collecting {n_samples - start_idx} samples (from {start_idx})...")
    print(f"  Current: {n_correct} correct, {n_incorrect} incorrect")

    with h5py.File(output_file, 'a') as f:
        pbar = tqdm(range(start_idx, n_samples), desc=f"  {model_key}/{task_name}")

        for i in pbar:
            prompt, answer, metadata = task_data[i]

            try:
                # Generate response and collect trajectory
                model_output, trajectory = collector.generate_with_trajectory(
                    prompt=prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    max_seq_len=MAX_SEQ_LEN
                )

                # Check correctness
                is_correct = check_correctness(model_output, answer, task_name, metadata)

                # Get sequence length
                seq_len = trajectory.shape[0]

                # Pad or truncate trajectory
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
                f['prompts'][i] = prompt
                f['model_outputs'][i] = model_output[:10000]  # Truncate very long outputs
                f['ground_truth'][i] = str(answer)[:5000]

                # Update counters
                if is_correct:
                    n_correct += 1
                else:
                    n_incorrect += 1

                # Update progress bar
                pbar.set_postfix({'correct': n_correct, 'incorrect': n_incorrect})

            except Exception as e:
                print(f"\n  Error on sample {i}: {e}")
                # Store placeholder for failed samples
                f['trajectories'][i] = np.zeros(
                    (MAX_SEQ_LEN, len(LAYERS_TO_COLLECT), collector.d_model),
                    dtype=np.float16
                )
                f['sequence_lengths'][i] = 0
                f['is_correct'][i] = False
                f['prompts'][i] = prompt
                f['model_outputs'][i] = f"ERROR: {str(e)}"
                f['ground_truth'][i] = str(answer)[:5000]
                n_incorrect += 1

            # Checkpoint every 25 samples
            if (i + 1) % 25 == 0:
                save_checkpoint(model_key, task_name, i + 1, n_correct, n_incorrect)

    # Final checkpoint
    save_checkpoint(model_key, task_name, n_samples, n_correct, n_incorrect)

    # Clean up
    del collector
    torch.cuda.empty_cache()

    # Report
    file_size_gb = output_file.stat().st_size / 1e9
    print(f"  Completed: {n_correct} correct, {n_incorrect} incorrect")
    print(f"  File size: {file_size_gb:.2f} GB")


def main():
    """Main collection pipeline"""
    print("=" * 80)
    print("PHASE 2: Trajectory Collection WITH Correctness Labels")
    print("=" * 80)
    print()
    print(f"Layers: Even only [0, 2, ..., 30] = {len(LAYERS_TO_COLLECT)} layers")
    print(f"Samples: {N_SAMPLES} per task")
    print(f"Max sequence length: {MAX_SEQ_LEN} tokens")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    # Load configs
    models = load_models_config()

    # Task preparation functions
    task_prep_fns = {
        'gsm8k': prepare_gsm8k,
        'humaneval': prepare_humaneval,
        'logiqa': prepare_logiqa
    }

    # Models to process
    model_keys = ['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']

    for model_key in model_keys:
        if model_key not in models:
            print(f"Model {model_key} not in config, skipping")
            continue

        model_config = models[model_key]
        print(f"\n{'=' * 80}")
        print(f"Model: {model_key} ({model_config['model_name']})")
        print(f"{'=' * 80}")

        for task_name, prep_fn in task_prep_fns.items():
            print(f"\nTask: {task_name}")

            # Prepare task data
            task_data = prep_fn(n_samples=N_SAMPLES, split='test')

            try:
                collect_for_model_task(
                    model_key=model_key,
                    model_config=model_config,
                    task_name=task_name,
                    task_data=task_data,
                    n_samples=N_SAMPLES
                )
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

        # Free GPU memory between models
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)

    # Summary
    print("\nSummary:")
    for model_key in model_keys:
        print(f"\n{model_key}:")
        for task_name in task_prep_fns.keys():
            checkpoint = load_checkpoint(model_key, task_name)
            print(f"  {task_name}: {checkpoint['n_correct']} correct, {checkpoint['n_incorrect']} incorrect")

    total_size = sum(
        f.stat().st_size
        for f in OUTPUT_DIR.rglob("*.h5")
    ) / 1e9
    print(f"\nTotal storage: {total_size:.2f} GB")


if __name__ == "__main__":
    main()
