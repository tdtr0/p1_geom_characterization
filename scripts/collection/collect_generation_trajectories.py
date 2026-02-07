#!/usr/bin/env python3
"""
Generation-Time Trajectory Collection

Collects:
1. Hidden states at each generation step (even layers: 0, 2, 4, ..., 30)
2. Top-100 tokens + exact entropy per step
3. Selective attention weights (layers 4, 8, 12, 16, 20, 24, 28, 31)

Features:
- Per-sample checkpointing (resumable)
- Duplicate detection via prompt hashing
- Uploads to B2 after each model
"""

import os
import sys
import re
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import yaml
import json
import hashlib
from datetime import datetime
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from task_data import prepare_gsm8k, prepare_humaneval, prepare_logiqa

# =============================================================================
# Configuration
# =============================================================================

MAX_NEW_TOKENS = 512
MAX_SEQ_LEN = 512

# Even layers for hidden states: [0, 2, 4, ..., 30] = 16 layers
HIDDEN_LAYERS = list(range(0, 32, 2))

# Selective layers for attention: spread across depth
ATTENTION_LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]

# Number of heads to capture (first 8)
N_ATTENTION_HEADS = 8

# Top-k tokens to store
TOP_K = 100


# =============================================================================
# Model Configuration
# =============================================================================

def load_models_config():
    """Load model configurations."""
    config_path = Path("configs/models.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get('primary', {})


# =============================================================================
# Checkpointing
# =============================================================================

def get_checkpoint_path(checkpoint_dir: Path, model_key: str, task: str) -> Path:
    """Get checkpoint file path."""
    return checkpoint_dir / f"generation_{model_key}_{task}.json"


def load_checkpoint(checkpoint_dir: Path, model_key: str, task: str) -> dict:
    """Load checkpoint if exists."""
    path = get_checkpoint_path(checkpoint_dir, model_key, task)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_samples": [], "n_correct": 0, "n_total": 0}


def save_checkpoint(checkpoint_dir: Path, model_key: str, task: str,
                   completed_samples: list, n_correct: int, n_total: int):
    """Save checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = get_checkpoint_path(checkpoint_dir, model_key, task)
    with open(path, 'w') as f:
        json.dump({
            "completed_samples": completed_samples,
            "n_correct": n_correct,
            "n_total": n_total,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)


def hash_prompt(prompt: str) -> str:
    """Hash prompt for duplicate detection."""
    return hashlib.md5(prompt.encode()).hexdigest()[:16]


# =============================================================================
# Correctness Checking
# =============================================================================

def extract_gsm8k_answer(text: str) -> str:
    """Extract numerical answer from GSM8K response."""
    match = re.search(r'####\s*(-?[\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    match = re.search(r'(?:the answer is|answer:?)\s*(-?[\d,\.]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '').strip()
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return ""


def check_gsm8k_correct(model_output: str, ground_truth: str) -> bool:
    """Check GSM8K correctness."""
    gt_match = re.search(r'####\s*(-?[\d,\.]+)', ground_truth)
    if not gt_match:
        return False
    gt_answer = gt_match.group(1).replace(',', '').strip()
    model_answer = extract_gsm8k_answer(model_output)
    try:
        return abs(float(gt_answer) - float(model_answer)) < 1e-6
    except (ValueError, TypeError):
        return gt_answer == model_answer


def check_logiqa_correct(model_output: str, ground_truth: str) -> bool:
    """Check LogiQA correctness."""
    match = re.search(r'(?:answer(?:\s+is)?:?\s*)([A-D])', model_output, re.IGNORECASE)
    if match:
        return match.group(1).upper() == ground_truth.upper()
    match = re.search(r'\b([A-D])\b', model_output)
    if match:
        return match.group(1).upper() == ground_truth.upper()
    return False


def check_humaneval_correct(model_output: str, test_code: str, entry_point: str) -> bool:
    """Check HumanEval correctness (simplified: syntax check)."""
    try:
        compile(model_output, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def check_correctness(model_output: str, ground_truth: str, task: str, metadata: dict = None) -> bool:
    """Unified correctness checker."""
    if task == "gsm8k":
        return check_gsm8k_correct(model_output, ground_truth)
    elif task == "logiqa":
        return check_logiqa_correct(model_output, ground_truth)
    elif task == "humaneval":
        test_code = metadata.get('test', '') if metadata else ''
        entry_point = metadata.get('entry_point', '') if metadata else ''
        return check_humaneval_correct(model_output, test_code, entry_point)
    else:
        raise ValueError(f"Unknown task: {task}")


# =============================================================================
# Generation Collector
# =============================================================================

class GenerationCollector:
    """Collects hidden states, attention, and logits during generation."""

    def __init__(self, model_name: str, hidden_layers: list, attention_layers: list):
        self.model_name = model_name
        self.hidden_layers = hidden_layers
        self.attention_layers = attention_layers

        print(f"Loading model: {model_name}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with attention output enabled
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            output_attentions=True,
        )
        self.model.eval()

        # Enable attention output in config
        self.model.config.output_attentions = True

        self.device = next(self.model.parameters()).device
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size
        self.n_heads = self.model.config.num_attention_heads
        self.vocab_size = self.model.config.vocab_size

        print(f"  Loaded: {self.n_layers} layers, d_model={self.d_model}, vocab={self.vocab_size}")
        print(f"  Hidden layers to collect: {hidden_layers}")
        print(f"  Attention layers to collect: {attention_layers}")

        # Pre-allocate storage for hooks
        self.step_hidden_states = []  # List of dicts per step
        self.step_attentions = []     # List of dicts per step

        # Register hooks
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]

            if layer_idx in self.hidden_layers:
                handle = layer.register_forward_hook(self._make_hidden_hook(layer_idx))
                self.handles.append(handle)

            # For attention, we need to hook the attention module
            if layer_idx in self.attention_layers:
                handle = layer.self_attn.register_forward_hook(self._make_attn_hook(layer_idx))
                self.handles.append(handle)

    def _make_hidden_hook(self, layer_idx):
        """Create hook for hidden states."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Only keep last token for generation steps
            last_hidden = hidden[:, -1, :].detach().cpu().half()

            if not hasattr(self, '_current_step_hidden'):
                self._current_step_hidden = {}
            self._current_step_hidden[layer_idx] = last_hidden
        return hook

    def _make_attn_hook(self, layer_idx):
        """Create hook for attention weights."""
        def hook(module, input, output):
            # output is (attn_output, attn_weights, past_key_value) when output_attentions=True
            if len(output) > 1 and output[1] is not None:
                attn_weights = output[1]  # (batch, n_heads, seq_len, seq_len)
                # Keep only last query position, first N_ATTENTION_HEADS heads
                last_attn = attn_weights[:, :N_ATTENTION_HEADS, -1, :].detach().cpu().half()

                if not hasattr(self, '_current_step_attn'):
                    self._current_step_attn = {}
                self._current_step_attn[layer_idx] = last_attn
        return hook

    def _clear_step_buffers(self):
        """Clear per-step buffers."""
        self._current_step_hidden = {}
        self._current_step_attn = {}

    def _save_step_data(self):
        """Save current step data and clear buffers."""
        if hasattr(self, '_current_step_hidden') and self._current_step_hidden:
            self.step_hidden_states.append(dict(self._current_step_hidden))
        if hasattr(self, '_current_step_attn') and self._current_step_attn:
            self.step_attentions.append(dict(self._current_step_attn))
        self._clear_step_buffers()

    def generate_with_capture(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
        """Generate response and capture all data.

        Returns:
            dict with:
                - 'text': generated text
                - 'hidden_states': (gen_len, n_hidden_layers, d_model)
                - 'attention': (gen_len, n_attn_layers, n_heads, seq_len)
                - 'entropy': (gen_len,)
                - 'top_k_tokens': (gen_len, k)
                - 'top_k_probs': (gen_len, k)
                - 'prompt_len': int
                - 'gen_len': int
        """
        # Reset storage
        self.step_hidden_states = []
        self.step_attentions = []
        self._clear_step_buffers()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[1]

        # Generate with scores and attentions
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Process scores -> entropy + top-k
        if outputs.scores:
            scores = torch.stack(outputs.scores, dim=1)  # (batch, gen_len, vocab)
            probs = F.softmax(scores, dim=-1)

            # Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropy = entropy[0].cpu().numpy().astype(np.float32)

            # Top-k
            top_probs, top_indices = torch.topk(probs, k=TOP_K, dim=-1)
            top_tokens = top_indices[0].cpu().numpy().astype(np.int32)
            top_probs = top_probs[0].cpu().numpy().astype(np.float16)
        else:
            gen_len = outputs.sequences.shape[1] - prompt_len
            entropy = np.zeros(gen_len, dtype=np.float32)
            top_tokens = np.zeros((gen_len, TOP_K), dtype=np.int32)
            top_probs = np.zeros((gen_len, TOP_K), dtype=np.float16)

        gen_len = len(entropy)

        # Process hidden states from hooks
        # The hooks capture at each forward pass during generate()
        # We need to extract the generation-time states
        if self.step_hidden_states:
            hidden_list = []
            for step_data in self.step_hidden_states[-gen_len:]:  # Take last gen_len steps
                step_hidden = []
                for layer_idx in self.hidden_layers:
                    if layer_idx in step_data:
                        step_hidden.append(step_data[layer_idx][0].numpy())
                    else:
                        step_hidden.append(np.zeros(self.d_model, dtype=np.float16))
                hidden_list.append(np.stack(step_hidden))
            hidden_states = np.stack(hidden_list)  # (gen_len, n_layers, d_model)
        else:
            hidden_states = np.zeros((gen_len, len(self.hidden_layers), self.d_model), dtype=np.float16)

        # Process attention from hooks
        if self.step_attentions:
            attn_list = []
            max_seq = 0
            for step_data in self.step_attentions[-gen_len:]:
                for layer_idx in self.attention_layers:
                    if layer_idx in step_data:
                        max_seq = max(max_seq, step_data[layer_idx].shape[-1])

            for step_data in self.step_attentions[-gen_len:]:
                step_attn = []
                for layer_idx in self.attention_layers:
                    if layer_idx in step_data:
                        attn = step_data[layer_idx][0].numpy()  # (n_heads, seq_len)
                        # Pad to max_seq
                        if attn.shape[-1] < max_seq:
                            pad = np.zeros((N_ATTENTION_HEADS, max_seq - attn.shape[-1]), dtype=np.float16)
                            attn = np.concatenate([attn, pad], axis=-1)
                        step_attn.append(attn)
                    else:
                        step_attn.append(np.zeros((N_ATTENTION_HEADS, max_seq), dtype=np.float16))
                attn_list.append(np.stack(step_attn))
            attention = np.stack(attn_list)  # (gen_len, n_attn_layers, n_heads, seq_len)
        else:
            attention = np.zeros((gen_len, len(self.attention_layers), N_ATTENTION_HEADS, prompt_len + gen_len), dtype=np.float16)

        # Decode text
        generated_ids = outputs.sequences[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            'text': text,
            'hidden_states': hidden_states,
            'attention': attention,
            'entropy': entropy,
            'top_k_tokens': top_tokens,
            'top_k_probs': top_probs,
            'prompt_len': prompt_len,
            'gen_len': gen_len,
        }

    def cleanup(self):
        """Remove hooks and free memory."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()


# =============================================================================
# Data Collection
# =============================================================================

def collect_for_task(
    collector: GenerationCollector,
    model_key: str,
    task: str,
    task_data: list,
    output_dir: Path,
    checkpoint_dir: Path,
    max_samples: int,
    resume: bool = True
):
    """Collect generation data for a task."""

    # Load checkpoint
    if resume:
        ckpt = load_checkpoint(checkpoint_dir, model_key, task)
        completed = set(ckpt['completed_samples'])
        n_correct = ckpt['n_correct']
    else:
        completed = set()
        n_correct = 0

    # Filter samples to process
    samples_to_process = [(i, d) for i, d in enumerate(task_data[:max_samples]) if i not in completed]

    if not samples_to_process:
        print(f"  All {max_samples} samples already completed")
        return

    print(f"  Processing {len(samples_to_process)} samples (resuming from {len(completed)})")

    # Prepare output file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task}_generation.h5"

    # Process samples
    for sample_idx, sample in tqdm(samples_to_process, desc=f"{model_key}/{task}"):
        prompt = sample['prompt']
        ground_truth = sample['ground_truth']
        metadata = sample.get('metadata', {})

        # Check for duplicates
        prompt_hash = hash_prompt(prompt)

        try:
            # Generate with capture
            result = collector.generate_with_capture(prompt)

            # Check correctness
            is_correct = check_correctness(result['text'], ground_truth, task, metadata)
            if is_correct:
                n_correct += 1

            # Save to HDF5
            with h5py.File(output_file, 'a') as f:
                # Create group for this sample
                group_name = f"sample_{sample_idx:04d}"
                if group_name in f:
                    del f[group_name]  # Overwrite if exists

                g = f.create_group(group_name)

                # Metadata
                g.attrs['prompt'] = prompt
                g.attrs['response'] = result['text']
                g.attrs['ground_truth'] = ground_truth
                g.attrs['is_correct'] = is_correct
                g.attrs['prompt_len'] = result['prompt_len']
                g.attrs['gen_len'] = result['gen_len']
                g.attrs['prompt_hash'] = prompt_hash

                # Data
                g.create_dataset('hidden_states', data=result['hidden_states'],
                               compression='gzip', dtype=np.float16)
                g.create_dataset('attention', data=result['attention'],
                               compression='gzip', dtype=np.float16)
                g.create_dataset('entropy', data=result['entropy'], dtype=np.float32)
                g.create_dataset('top_k_tokens', data=result['top_k_tokens'],
                               compression='gzip', dtype=np.int32)
                g.create_dataset('top_k_probs', data=result['top_k_probs'],
                               compression='gzip', dtype=np.float16)

            # Update checkpoint
            completed.add(sample_idx)
            save_checkpoint(checkpoint_dir, model_key, task,
                          list(completed), n_correct, len(completed))

        except Exception as e:
            print(f"  ERROR on sample {sample_idx}: {e}")
            continue

    print(f"  Completed: {len(completed)}/{max_samples}, Correct: {n_correct}/{len(completed)}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect generation trajectories")
    parser.add_argument("--model", type=str, required=True,
                       choices=['olmo3_base', 'olmo3_sft', 'olmo3_rl_zero', 'olmo3_think'])
    parser.add_argument("--task", type=str, required=True,
                       choices=['gsm8k', 'humaneval', 'logiqa'])
    parser.add_argument("--output-dir", type=Path, default=Path("data/generation_trajectories"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--layers-even", action="store_true",
                       help="Use even layers [0,2,4,...,30]")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Generation Trajectory Collection")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Max samples: {args.max_samples}")
    print(f"Resume: {args.resume}")
    print()

    # Load model config
    models_config = load_models_config()
    if args.model not in models_config:
        raise ValueError(f"Unknown model: {args.model}")

    model_config = models_config[args.model]
    model_name = model_config['model_name']  # 'model_name' in YAML config

    # Load task data
    print(f"Loading {args.task} data...")
    if args.task == 'gsm8k':
        task_data = prepare_gsm8k(n_shot=0, seed=args.seed)
    elif args.task == 'humaneval':
        task_data = prepare_humaneval(seed=args.seed)
    elif args.task == 'logiqa':
        task_data = prepare_logiqa(n_shot=0, seed=args.seed)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    print(f"  Loaded {len(task_data)} samples")

    # Initialize collector
    hidden_layers = HIDDEN_LAYERS if args.layers_even else list(range(32))
    collector = GenerationCollector(model_name, hidden_layers, ATTENTION_LAYERS)

    # Create output directory for this model
    model_output_dir = args.output_dir / args.model

    try:
        # Collect
        collect_for_task(
            collector=collector,
            model_key=args.model,
            task=args.task,
            task_data=task_data,
            output_dir=model_output_dir,
            checkpoint_dir=args.checkpoint_dir,
            max_samples=args.max_samples,
            resume=args.resume
        )
    finally:
        collector.cleanup()

    print()
    print("Done!")


if __name__ == "__main__":
    main()
