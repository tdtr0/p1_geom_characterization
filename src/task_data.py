"""
Task Data Preparation Module

Loads and formats datasets for activation collection.
Supports GSM8K, HumanEval, and LogiQA reasoning tasks.
"""

from datasets import load_dataset
from typing import List, Tuple, Optional
import random


def prepare_gsm8k(
    n_samples: int = 500,
    split: str = "test",
    seed: Optional[int] = 42
) -> List[Tuple[str, str, dict]]:
    """
    Prepare GSM8K math word problems.

    Returns:
        List of (prompt, answer, metadata) tuples
    """
    ds = load_dataset("gsm8k", "main", split=split)

    if seed is not None:
        random.seed(seed)
        indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
        ds = ds.select(indices)
    else:
        ds = ds.select(range(min(n_samples, len(ds))))

    prompts = []
    for idx, item in enumerate(ds):
        # Format as CoT prompt
        prompt = f"""Solve this math problem step by step.

Problem: {item['question']}

Solution:"""

        metadata = {
            "task": "gsm8k",
            "index": idx,
            "question": item['question'],
        }

        prompts.append((prompt, item['answer'], metadata))

    print(f"✓ Loaded {len(prompts)} GSM8K samples")
    return prompts


def prepare_humaneval(
    n_samples: int = 164,
    split: str = "test",
    seed: Optional[int] = 42
) -> List[Tuple[str, str, dict]]:
    """
    Prepare HumanEval coding problems.

    Returns:
        List of (prompt, solution, metadata) tuples
    """
    ds = load_dataset("openai_humaneval", split=split)

    if seed is not None:
        random.seed(seed)
        indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
        ds = ds.select(indices)
    else:
        ds = ds.select(range(min(n_samples, len(ds))))

    prompts = []
    for idx, item in enumerate(ds):
        prompt = f"""Complete the following Python function.

{item['prompt']}"""

        metadata = {
            "task": "humaneval",
            "index": idx,
            "task_id": item['task_id'],
        }

        prompts.append((prompt, item['canonical_solution'], metadata))

    print(f"✓ Loaded {len(prompts)} HumanEval samples")
    return prompts


def prepare_logiqa(
    n_samples: int = 500,
    split: str = "test",
    seed: Optional[int] = 42
) -> List[Tuple[str, str, dict]]:
    """
    Prepare LogiQA logical reasoning problems.

    Returns:
        List of (prompt, answer, metadata) tuples
    """
    ds = load_dataset("lucasmccabe/logiqa", split=split)

    if seed is not None:
        random.seed(seed)
        indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
        ds = ds.select(indices)
    else:
        ds = ds.select(range(min(n_samples, len(ds))))

    prompts = []
    for idx, item in enumerate(ds):
        # Format options as A, B, C, D
        options = "\n".join([
            f"{chr(65+i)}. {opt}"
            for i, opt in enumerate(item['options'])
        ])

        prompt = f"""Answer this logical reasoning question.

Context: {item['context']}

Question: {item['question']}

Options:
{options}

Think step by step, then give your answer (A, B, C, or D):"""

        metadata = {
            "task": "logiqa",
            "index": idx,
            "correct_answer": item['answer'],
        }

        prompts.append((prompt, item['answer'], metadata))

    print(f"✓ Loaded {len(prompts)} LogiQA samples")
    return prompts


def prepare_mmlu(
    n_samples: int = 500,
    subject: str = "all",
    split: str = "test",
    seed: Optional[int] = 42
) -> List[Tuple[str, str, dict]]:
    """
    Prepare MMLU (Massive Multitask Language Understanding) problems.

    Args:
        n_samples: Number of samples to load
        subject: Subject area or "all" for mixed subjects
        split: Dataset split
        seed: Random seed for sampling

    Returns:
        List of (prompt, answer, metadata) tuples
    """
    if subject == "all":
        # Load from multiple subjects for diversity
        subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics",
                   "clinical_knowledge", "college_biology", "college_chemistry",
                   "college_computer_science", "college_mathematics", "college_physics"]
        samples_per_subject = n_samples // len(subjects)

        all_prompts = []
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split=split)
                n_take = min(samples_per_subject, len(ds))

                if seed is not None:
                    random.seed(seed + hash(subj))
                    indices = random.sample(range(len(ds)), n_take)
                    ds = ds.select(indices)
                else:
                    ds = ds.select(range(n_take))

                all_prompts.extend(_format_mmlu_samples(ds, subj))
            except Exception as e:
                print(f"Warning: Could not load MMLU subject {subj}: {e}")

        prompts = all_prompts[:n_samples]
    else:
        ds = load_dataset("cais/mmlu", subject, split=split)

        if seed is not None:
            random.seed(seed)
            indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
            ds = ds.select(indices)
        else:
            ds = ds.select(range(min(n_samples, len(ds))))

        prompts = _format_mmlu_samples(ds, subject)

    print(f"✓ Loaded {len(prompts)} MMLU samples")
    return prompts


def _format_mmlu_samples(ds, subject: str) -> List[Tuple[str, str, dict]]:
    """Helper to format MMLU samples."""
    prompts = []
    for idx, item in enumerate(ds):
        choices = "\n".join([
            f"{chr(65+i)}. {choice}"
            for i, choice in enumerate(item['choices'])
        ])

        prompt = f"""Answer this question from {subject.replace('_', ' ')}.

Question: {item['question']}

Choices:
{choices}

Answer:"""

        metadata = {
            "task": "mmlu",
            "subject": subject,
            "index": idx,
            "correct_answer": item['answer'],
        }

        prompts.append((prompt, str(item['answer']), metadata))

    return prompts


# Task registry
TASKS = {
    "gsm8k": prepare_gsm8k,
    "humaneval": prepare_humaneval,
    "logiqa": prepare_logiqa,
    "mmlu": prepare_mmlu,
}


def get_task_data(
    task_name: str,
    n_samples: int = 500,
    **kwargs
) -> List[Tuple[str, str, dict]]:
    """
    Unified interface to load any task.

    Args:
        task_name: Name of task (gsm8k, humaneval, logiqa, mmlu)
        n_samples: Number of samples to load
        **kwargs: Additional task-specific arguments

    Returns:
        List of (prompt, answer, metadata) tuples
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")

    task_fn = TASKS[task_name]
    return task_fn(n_samples=n_samples, **kwargs)
