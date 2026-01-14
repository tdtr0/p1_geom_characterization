# Phase 1 Implementation Plan: Geometric Characterization

---

## 📊 PROGRESS TRACKER (Last Updated: 2026-01-10)

### Completed ✅

| Task | Status | Notes |
|------|--------|-------|
| Environment setup | ✅ Done | conda env `geometric_transfer` on eyecog |
| Model download/verification | ✅ Done | 4 OLMo models (Base, SFT, RL-Zero, Think) |
| Activation collection pipeline | ✅ Done | With checkpointing, error handling |
| Task data preparation | ✅ Done | GSM8K (500), HumanEval (164), LogiQA (500) |
| Data collection | ✅ Done | 12 files, 4 models × 3 tasks, 0% NaN |
| Core geometric measures | ✅ Done | Effective rank, spectral decay, subspace preservation |
| Analysis pipeline | ✅ Done | `scripts/run_analysis.py` |
| Statistical tests | ✅ Done | t-tests, Cohen's d, Mann-Whitney U |
| Curvature analysis | ✅ Done | Local manifold curvature estimation |

### Key Results 🔬

**Subspace Preservation vs Base Model:**
| Model | Training | GSM8K | HumanEval | LogiQA |
|-------|----------|-------|-----------|--------|
| RL-Zero | Pure RL (no SFT) | **98.6% ± 1.1%** | **97.4% ± 2.9%** | **96.0% ± 4.1%** |
| SFT | Supervised fine-tuning | 52.4% ± 12.6% | 50.3% ± 15.5% | 39.6% ± 17.7% |
| Think | SFT + DPO + RLVR | 50.4% ± 12.9% | 49.7% ± 16.0% | 39.0% ± 18.0% |

**Statistical Tests (RL-Zero vs SFT):**
| Task | Welch t | p-value | Cohen's d | Effect |
|------|---------|---------|-----------|--------|
| GSM8K | 20.74 | 6.5e-20 | 5.19 | Huge |
| HumanEval | 16.93 | 7.2e-18 | 4.23 | Huge |
| LogiQA | 17.62 | 9.1e-19 | 4.41 | Huge |

**Curvature Analysis (layer 15, higher = more curved):**
| Model | GSM8K | HumanEval |
|-------|-------|-----------|
| Base | 0.626 ± 0.027 | 0.582 ± 0.052 |
| RL-Zero | 0.624 ± 0.027 | 0.576 ± 0.053 |
| SFT | 0.644 ± 0.025 | 0.597 ± 0.045 |
| Think | 0.643 ± 0.026 | 0.597 ± 0.045 |

**Interpretation:**
- **RL-Zero preserves >96% of base geometry across all 3 tasks**
- **SFT preserves <53% on all tasks** - dramatic geometric reshaping
- Pure RL makes minimal geometric changes while achieving fine-tuning
- Adding DPO+RLVR to SFT doesn't further change geometry (Think ≈ SFT)
- Curvature: RL-Zero maintains base curvature, SFT slightly increases it
- All effects are huge (Cohen's d > 4.0) with p < 10^-17

### Storage Status for Phase 2

**Current Usage (Phase 1 - last_token only):**
- Total: 1.1 GB (4 models × 3 tasks)
- Per model: ~269 MB

**Phase 2 Requirements (full trajectories):**
- Estimated: ~112 GB compressed (gzip)
- Available space: 122 GB
- Margin: ~10 GB (tight)
- **Recommendation**: Clean up space or collect trajectories in batches

### Decision Point Assessment

Based on success criteria in Section 5.3:
- ✅ **RL-Zero shows higher preservation**: 96-98% vs 40-53%
- ✅ **Effect size**: Huge (Cohen's d > 4.0 for all tasks)
- ✅ **p-value**: p < 10^-17 for all tasks
- ✅ **Consistent across tasks**: GSM8K, HumanEval, LogiQA all show same pattern

**✅ PHASE 1 COMPLETE - Strong signal confirmed across all measures**

---

## What We're Actually Doing (And Why)

### The Core Question

We want to know: **Do RLVR and SFT produce measurably different geometric structures in activation space?**

This is a prerequisite question. Before we can ask whether geometry *predicts* transfer, we need to establish that different training paradigms produce different geometry at all. Phase 1 is purely descriptive—we're characterizing the activation manifolds of different models without yet connecting to transfer performance.

### What "Geometric Characterization" Means Operationally

When a transformer processes a prompt, each layer produces an activation vector (the residual stream state). For a model with L layers and hidden dimension d, processing a sequence of T tokens produces:

- **Per-token trajectory**: T sequences of L vectors, each in ℝ^d
- **Sequence-level representation**: L vectors in ℝ^d (after some aggregation like mean-pooling)

"Geometry" refers to the statistical structure of these vectors across many inputs:
- How they're distributed in the high-dimensional space (dimensionality)
- How concentrated vs. spread out they are (spectral properties)
- How smooth vs. curved the manifold they trace is (curvature)
- How similar the subspaces are between different models (preservation)

### Why These Specific Models

**OLMo 3 family** provides the cleanest controlled comparison available:

| Model | Training | Why It Matters |
|-------|----------|----------------|
| OLMo 3-Base 7B | Pretraining only | Baseline—no post-training signal |
| OLMo 3-Think-SFT 7B | Base + SFT | SFT on Dolci-Think dataset |
| OLMo 3-RL-Zero 7B | Base + RL (no SFT) | **Pure RL effect isolated** |
| OLMo 3-Think 7B | Base + SFT + DPO + RLVR | Full training pipeline |

This is rare: most RLHF models are SFT→RL, confounding the two signals. OLMo 3-RL-Zero goes directly Base→RL, letting us isolate RL's geometric effect.

**Experimental Design:**
- **SFT vs RL-Zero**: Does pure RL differ from pure SFT? → **YES, dramatically!**
- **SFT vs Think**: What does adding DPO+RLVR do? → Minimal additional geometric change
- **RL-Zero vs Think**: How does pure RL compare to full pipeline? → Very different

**DeepSeek family** (secondary) provides scale but less control:
- V3-Base, V3, R1-Zero, R1 span the full training trajectory
- Use distilled versions (8B, 14B) for compute feasibility
- Less controlled because distillation may introduce artifacts

---

## Week 1-2: Infrastructure Setup

### 2.1 Environment Configuration

```bash
# Create isolated environment
conda create -n geometric_transfer python=3.10
conda activate geometric_transfer

# Core dependencies
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0
pip install accelerate==0.29.0
pip install bitsandbytes==0.43.0  # For efficient loading

# Analysis tools
pip install numpy scipy scikit-learn
pip install h5py  # Activation storage
pip install signatory  # Path signatures (install separately, needs torch first)

# TransformerLens for activation extraction
pip install transformer_lens==1.14.0

# Monitoring
pip install wandb tqdm
```

**Why TransformerLens?** It provides clean hooks for extracting activations at arbitrary points in the forward pass without manual surgery on model code. Critical features:
- `run_with_cache()` extracts all intermediate activations
- Named hooks like `blocks.{layer}.hook_resid_post` for residual stream
- Works with HuggingFace models via `from_pretrained_no_processing`

### 2.2 Model Download and Verification

```python
# models_config.py
MODELS = {
    # Primary comparison (OLMo family)
    "olmo3_base": "allenai/OLMo-3-7B",
    "olmo3_instruct": "allenai/OLMo-3-7B-Instruct", 
    "olmo3_rl_zero": "allenai/OLMo-3-7B-RL-Zero",
    
    # Secondary comparison (DeepSeek distilled)
    "deepseek_r1_8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek_v3_8b": "deepseek-ai/DeepSeek-V3-0324",  # Check actual name
    
    # Tertiary validation
    "llama3_base": "meta-llama/Meta-Llama-3-8B",
    "llama3_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# Storage requirements (approximate)
# OLMo 7B: ~14GB per model in fp16
# DeepSeek 8B distill: ~16GB per model
# Total: ~100GB for all models
```

**Verification checklist:**
- [ ] All models load without errors
- [ ] Hidden dimension matches expected (check `model.config.hidden_size`)
- [ ] Number of layers matches expected (check `model.config.num_hidden_layers`)
- [ ] Generate a test completion to verify model works
- [ ] TransformerLens hooks attach correctly

### 2.3 Activation Collection Pipeline

This is the core infrastructure. We need to extract activations efficiently and store them in a format that supports downstream analysis.

```python
# activation_collector.py
import torch
import h5py
from transformer_lens import HookedTransformer
from typing import List, Dict, Tuple
import numpy as np

class ActivationCollector:
    """
    Collects and stores activations from transformer forward passes.
    
    Design decisions:
    - Store residual stream (not attention/MLP separately) because:
      1. It's the "main" representation at each layer
      2. Reduces storage by ~3x vs storing all components
      3. Sufficient for geometric analysis
    - Store both pre-MLP and post-MLP for each layer to capture full trajectory
    - Use float16 to halve storage (precision sufficient for SVD)
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            dtype=dtype
        )
        self.model.eval()
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        
    def get_hook_names(self) -> List[str]:
        """Return hook names for residual stream at each layer."""
        hooks = []
        for layer in range(self.n_layers):
            # Post-attention, pre-MLP
            hooks.append(f"blocks.{layer}.hook_resid_mid")
            # Post-MLP (full layer output)
            hooks.append(f"blocks.{layer}.hook_resid_post")
        return hooks
    
    def collect_activations(
        self,
        texts: List[str],
        aggregation: str = "last_token"  # or "mean", "all_tokens"
    ) -> Dict[str, np.ndarray]:
        """
        Collect activations for a batch of texts.
        
        Args:
            texts: List of input strings
            aggregation: How to aggregate across sequence positions
                - "last_token": Use only final token (for causal models)
                - "mean": Mean-pool across all tokens
                - "all_tokens": Store full sequence (expensive)
        
        Returns:
            Dict mapping hook names to activation arrays
            Shape depends on aggregation:
                - "last_token"/"mean": (n_texts, d_model)
                - "all_tokens": (n_texts, max_seq_len, d_model)
        """
        hook_names = self.get_hook_names()
        all_activations = {name: [] for name in hook_names}
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                tokens = self.model.to_tokens(text)
                
                # Forward with cache
                _, cache = self.model.run_with_cache(tokens, names_filter=hook_names)
                
                # Extract and aggregate
                for name in hook_names:
                    act = cache[name]  # Shape: (1, seq_len, d_model)
                    
                    if aggregation == "last_token":
                        act = act[0, -1, :]  # (d_model,)
                    elif aggregation == "mean":
                        act = act[0].mean(dim=0)  # (d_model,)
                    elif aggregation == "all_tokens":
                        act = act[0]  # (seq_len, d_model)
                    
                    all_activations[name].append(act.cpu().numpy().astype(np.float16))
        
        # Stack into arrays
        for name in hook_names:
            all_activations[name] = np.stack(all_activations[name])
        
        return all_activations
    
    def save_to_hdf5(
        self,
        activations: Dict[str, np.ndarray],
        filepath: str,
        metadata: Dict = None
    ):
        """Save activations with metadata for reproducibility."""
        with h5py.File(filepath, 'w') as f:
            # Store activations
            for name, arr in activations.items():
                f.create_dataset(name, data=arr, compression='gzip')
            
            # Store metadata
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, str):
                        meta_group.attrs[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        meta_group.create_dataset(key, data=value)
                    else:
                        meta_group.attrs[key] = value
```

### 2.4 Task Data Preparation

We need diverse prompts that elicit reasoning. The goal is NOT to evaluate model performance—it's to collect activations during reasoning-like processing.

```python
# task_data.py
from datasets import load_dataset
from typing import List, Tuple

def prepare_gsm8k(n_samples: int = 500) -> List[Tuple[str, str, bool]]:
    """
    Prepare GSM8K math problems.
    Returns: List of (prompt, answer, is_correct_format)
    """
    ds = load_dataset("gsm8k", "main", split="test")
    
    prompts = []
    for item in ds.select(range(min(n_samples, len(ds)))):
        # Format as CoT prompt
        prompt = f"""Solve this math problem step by step.

Problem: {item['question']}

Solution:"""
        prompts.append((prompt, item['answer'], True))
    
    return prompts

def prepare_humaneval(n_samples: int = 164) -> List[Tuple[str, str, bool]]:
    """Prepare HumanEval coding problems."""
    ds = load_dataset("openai_humaneval", split="test")
    
    prompts = []
    for item in ds.select(range(min(n_samples, len(ds)))):
        prompt = f"""Complete the following Python function.

{item['prompt']}"""
        prompts.append((prompt, item['canonical_solution'], True))
    
    return prompts

def prepare_logiqa(n_samples: int = 500) -> List[Tuple[str, str, bool]]:
    """Prepare LogiQA logical reasoning problems."""
    ds = load_dataset("lucasmccabe/logiqa", split="test")
    
    prompts = []
    for item in ds.select(range(min(n_samples, len(ds)))):
        options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(item['options'])])
        prompt = f"""Answer this logical reasoning question.

Context: {item['context']}

Question: {item['question']}

Options:
{options}

Think step by step, then give your answer."""
        prompts.append((prompt, item['answer'], True))
    
    return prompts

# Task registry
TASKS = {
    "gsm8k": prepare_gsm8k,
    "humaneval": prepare_humaneval,
    "logiqa": prepare_logiqa,
}
```

### 2.5 Pipeline Verification

Before collecting data, verify everything works on a small scale:

```python
# verify_pipeline.py
def run_verification():
    """End-to-end test on small sample."""
    
    # 1. Load model
    collector = ActivationCollector("allenai/OLMo-3-7B")
    print(f"Model loaded: {collector.n_layers} layers, d_model={collector.d_model}")
    
    # 2. Test prompts
    test_prompts = [
        "What is 2 + 2? Let me think step by step.",
        "Write a function to compute factorial.",
    ]
    
    # 3. Collect activations
    activations = collector.collect_activations(test_prompts, aggregation="last_token")
    
    # 4. Verify shapes
    for name, arr in activations.items():
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
        assert arr.shape == (2, collector.d_model), f"Unexpected shape: {arr.shape}"
    
    # 5. Test storage
    collector.save_to_hdf5(
        activations, 
        "test_activations.h5",
        metadata={"model": "olmo3_base", "n_samples": 2}
    )
    
    # 6. Verify loading
    with h5py.File("test_activations.h5", 'r') as f:
        loaded = f["blocks.0.hook_resid_post"][:]
        assert np.allclose(loaded, activations["blocks.0.hook_resid_post"])
    
    print("✓ Pipeline verification passed")
```

---

## Week 3-4: Data Collection

### 3.1 Collection Strategy

**Storage estimates:**
- Per model: 500 samples × 32 layers × 2 hooks × 4096 dims × 2 bytes (fp16) = ~250 MB per task
- 3 tasks × 3 OLMo models = ~2.3 GB for primary comparison
- Including DeepSeek and validation models: ~8-10 GB total

This is manageable. We can store everything without aggressive compression.

**Collection order:**
1. OLMo family first (primary comparison)
2. Validate geometric measures show signal
3. Only then collect DeepSeek/Llama (avoid wasted compute)

### 3.2 Collection Script

```python
# collect_activations.py
import os
import json
from datetime import datetime
from tqdm import tqdm

def collect_all_activations(
    output_dir: str,
    models: List[str] = ["olmo3_base", "olmo3_instruct", "olmo3_rl_zero"],
    tasks: List[str] = ["gsm8k", "humaneval", "logiqa"],
    samples_per_task: int = 500,
    batch_size: int = 4,  # Adjust based on GPU memory
):
    """
    Main collection loop.
    
    Organization:
    output_dir/
        olmo3_base/
            gsm8k.h5
            humaneval.h5
            logiqa.h5
        olmo3_instruct/
            ...
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Processing: {model_name}")
        print(f"{'='*50}")
        
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Load model once per model
        collector = ActivationCollector(MODELS[model_name])
        
        for task_name in tasks:
            print(f"\n  Task: {task_name}")
            output_path = os.path.join(model_dir, f"{task_name}.h5")
            
            # Skip if already collected
            if os.path.exists(output_path):
                print(f"    Already exists, skipping")
                continue
            
            # Load task data
            task_fn = TASKS[task_name]
            prompts_data = task_fn(samples_per_task)
            prompts = [p[0] for p in prompts_data]
            
            # Collect in batches
            all_activations = None
            for i in tqdm(range(0, len(prompts), batch_size)):
                batch = prompts[i:i+batch_size]
                batch_acts = collector.collect_activations(batch, aggregation="last_token")
                
                if all_activations is None:
                    all_activations = batch_acts
                else:
                    for name in all_activations:
                        all_activations[name] = np.concatenate([
                            all_activations[name], 
                            batch_acts[name]
                        ])
            
            # Save with metadata
            metadata = {
                "model": model_name,
                "task": task_name,
                "n_samples": len(prompts),
                "collection_date": datetime.now().isoformat(),
                "aggregation": "last_token",
            }
            collector.save_to_hdf5(all_activations, output_path, metadata)
            print(f"    Saved to {output_path}")
        
        # Free GPU memory before loading next model
        del collector
        torch.cuda.empty_cache()
```

### 3.3 Full Trajectory Collection (Subset)

For path signature analysis, we need full token-by-token trajectories. This is expensive, so we collect for a smaller subset.

```python
def collect_trajectory_subset(
    output_dir: str,
    models: List[str],
    task: str = "gsm8k",
    n_samples: int = 100,
):
    """
    Collect full trajectories (all tokens) for path signature analysis.
    
    Storage: 100 samples × ~200 tokens × 32 layers × 4096 dims × 2 bytes = ~5 GB per model
    """
    for model_name in models:
        print(f"Collecting trajectories for {model_name}")
        
        collector = ActivationCollector(MODELS[model_name])
        prompts_data = TASKS[task](n_samples)
        
        # Collect with full token storage
        # This is slow and memory-intensive
        activations = collector.collect_activations(
            [p[0] for p in prompts_data],
            aggregation="all_tokens"
        )
        
        output_path = os.path.join(output_dir, model_name, f"{task}_trajectories.h5")
        collector.save_to_hdf5(activations, output_path, {"type": "full_trajectory"})
```

### 3.4 Quality Checks

After collection, verify data integrity:

```python
def verify_collected_data(data_dir: str):
    """Run quality checks on collected activations."""
    
    issues = []
    
    for model_dir in os.listdir(data_dir):
        model_path = os.path.join(data_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        for task_file in os.listdir(model_path):
            if not task_file.endswith('.h5'):
                continue
                
            filepath = os.path.join(model_path, task_file)
            
            with h5py.File(filepath, 'r') as f:
                # Check all hooks present
                expected_hooks = 64  # 32 layers × 2 (mid + post)
                actual_hooks = len([k for k in f.keys() if k.startswith('blocks')])
                if actual_hooks != expected_hooks:
                    issues.append(f"{filepath}: Expected {expected_hooks} hooks, got {actual_hooks}")
                
                # Check no NaN/Inf
                for key in f.keys():
                    if key.startswith('blocks'):
                        data = f[key][:]
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            issues.append(f"{filepath}/{key}: Contains NaN or Inf")
                
                # Check reasonable magnitudes
                sample_key = "blocks.15.hook_resid_post"  # Middle layer
                if sample_key in f:
                    data = f[sample_key][:]
                    mag = np.linalg.norm(data, axis=-1).mean()
                    if mag < 1 or mag > 1000:
                        issues.append(f"{filepath}: Unusual activation magnitude: {mag}")
    
    if issues:
        print("Quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All quality checks passed")
    
    return issues
```

---

## Week 5-6: Static Geometric Analysis

### 5.1 Core Geometric Measures

Now we compute the geometric signatures. Each measure captures a different aspect of activation structure.

```python
# geometric_measures.py
import numpy as np
from scipy.linalg import svd, subspace_angles
from scipy.stats import entropy
from typing import Dict, Tuple

def compute_effective_rank(activations: np.ndarray) -> float:
    """
    Effective rank: exponential of entropy of normalized singular values.
    
    Measures how many dimensions are "actively used" in the representation.
    Low effective rank = concentrated on few dimensions (sparse/compressed)
    High effective rank = spread across many dimensions (distributed)
    
    Args:
        activations: (n_samples, d_model)
    
    Returns:
        Effective rank (scalar between 1 and min(n_samples, d_model))
    """
    # SVD
    _, s, _ = svd(activations, full_matrices=False)
    
    # Normalize to probability distribution
    s_normalized = s / s.sum()
    
    # Remove zeros for entropy calculation
    s_normalized = s_normalized[s_normalized > 1e-10]
    
    # Effective rank = exp(entropy)
    return np.exp(entropy(s_normalized))


def compute_spectral_decay(activations: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Fit power-law to singular value decay: s_i ∝ i^(-α)
    
    Higher α = faster decay = more concentrated spectrum
    Lower α = slower decay = more distributed spectrum
    
    Returns:
        alpha: Power-law exponent
        singular_values: Raw singular values for inspection
    """
    _, s, _ = svd(activations, full_matrices=False)
    
    # Fit log-log regression: log(s) = -α * log(i) + c
    log_i = np.log(np.arange(1, len(s) + 1))
    log_s = np.log(s + 1e-10)  # Add epsilon for numerical stability
    
    # Least squares fit
    A = np.vstack([log_i, np.ones_like(log_i)]).T
    alpha_neg, _ = np.linalg.lstsq(A, log_s, rcond=None)[0]
    
    return -alpha_neg, s


def compute_subspace_preservation(
    base_activations: np.ndarray,
    finetuned_activations: np.ndarray,
    k: int = 100
) -> Tuple[float, np.ndarray]:
    """
    Measure how much of base model's top-k subspace is preserved after fine-tuning.
    
    Uses principal angles between subspaces.
    Preservation = 1: Perfect preservation (identical subspaces)
    Preservation = 0: Orthogonal subspaces (no overlap)
    
    Args:
        base_activations: (n_samples, d_model) from base model
        finetuned_activations: (n_samples, d_model) from fine-tuned model  
        k: Number of top singular vectors to compare
    
    Returns:
        preservation_score: Mean cosine of principal angles
        angles: Individual principal angles (in radians)
    """
    # Get top-k right singular vectors
    _, _, Vt_base = svd(base_activations, full_matrices=False)
    _, _, Vt_ft = svd(finetuned_activations, full_matrices=False)
    
    V_base_k = Vt_base[:k, :].T  # (d_model, k)
    V_ft_k = Vt_ft[:k, :].T
    
    # Compute principal angles using scipy
    angles = subspace_angles(V_base_k, V_ft_k)
    
    # Preservation score
    preservation = np.cos(angles).mean()
    
    return preservation, angles


def compute_local_curvature(
    model,
    texts: List[str],
    layer_idx: int,
    n_perturbations: int = 20,
    eps: float = 0.01
) -> float:
    """
    Estimate local curvature via perturbation analysis.
    
    Idea: If the manifold is flat, small input perturbations cause small 
    representation changes. High curvature = perturbations amplified.
    
    We perturb at the token embedding level and measure activation variance.
    
    Args:
        model: TransformerLens model
        texts: Input texts to analyze
        layer_idx: Which layer to measure curvature at
        n_perturbations: Number of random perturbations per input
        eps: Perturbation magnitude (relative to embedding norm)
    
    Returns:
        Mean local curvature estimate
    """
    curvatures = []
    
    for text in texts:
        tokens = model.to_tokens(text)
        
        with torch.no_grad():
            # Get base embeddings
            base_embed = model.embed(tokens)  # (1, seq_len, d_model)
            embed_norm = base_embed.norm()
            
            # Base activation at target layer
            _, base_cache = model.run_with_cache(tokens)
            base_act = base_cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :]
            
            # Perturbed activations
            perturbed_acts = []
            for _ in range(n_perturbations):
                # Random perturbation
                noise = torch.randn_like(base_embed) * eps * embed_norm / np.sqrt(base_embed.numel())
                perturbed_embed = base_embed + noise
                
                # Forward from embeddings
                # Note: This requires model surgery or manual forward pass
                # Simplified: we'll perturb input tokens instead
                perturbed_acts.append(base_act.clone())  # Placeholder
            
            perturbed_acts = torch.stack(perturbed_acts)
            
            # Curvature = variance of perturbation responses
            variance = (perturbed_acts - base_act).pow(2).mean()
            curvatures.append(variance.item() / (eps ** 2))
    
    return np.mean(curvatures)
```

### 5.2 Analysis Pipeline

```python
# analyze_geometry.py
import pandas as pd
from collections import defaultdict

def analyze_all_models(
    data_dir: str,
    models: List[str],
    tasks: List[str],
    k_subspace: int = 100,
    layers_to_analyze: List[int] = None  # None = all layers
) -> pd.DataFrame:
    """
    Compute all geometric measures for all model/task combinations.
    
    Returns:
        DataFrame with columns:
        - model, task, layer
        - effective_rank, spectral_decay_alpha
        - preservation_vs_base (only for fine-tuned models)
    """
    results = []
    
    # Load base model activations for preservation comparison
    base_activations = {}  # task -> layer -> activations
    for task in tasks:
        base_path = os.path.join(data_dir, "olmo3_base", f"{task}.h5")
        with h5py.File(base_path, 'r') as f:
            base_activations[task] = {}
            for key in f.keys():
                if key.startswith('blocks') and 'resid_post' in key:
                    layer = int(key.split('.')[1])
                    base_activations[task][layer] = f[key][:]
    
    for model_name in models:
        print(f"Analyzing {model_name}")
        model_dir = os.path.join(data_dir, model_name)
        
        for task in tasks:
            filepath = os.path.join(model_dir, f"{task}.h5")
            
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    if not (key.startswith('blocks') and 'resid_post' in key):
                        continue
                    
                    layer = int(key.split('.')[1])
                    if layers_to_analyze and layer not in layers_to_analyze:
                        continue
                    
                    activations = f[key][:]
                    
                    # Compute measures
                    eff_rank = compute_effective_rank(activations)
                    alpha, _ = compute_spectral_decay(activations)
                    
                    # Subspace preservation (vs base model)
                    preservation = None
                    if model_name != "olmo3_base" and task in base_activations:
                        base_acts = base_activations[task].get(layer)
                        if base_acts is not None:
                            preservation, _ = compute_subspace_preservation(
                                base_acts, activations, k=k_subspace
                            )
                    
                    results.append({
                        'model': model_name,
                        'task': task,
                        'layer': layer,
                        'effective_rank': eff_rank,
                        'spectral_decay_alpha': alpha,
                        'preservation_vs_base': preservation,
                    })
    
    return pd.DataFrame(results)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-layer results to model-level summaries.
    """
    # Group by model and task
    grouped = df.groupby(['model', 'task']).agg({
        'effective_rank': ['mean', 'std'],
        'spectral_decay_alpha': ['mean', 'std'],
        'preservation_vs_base': ['mean', 'std'],
    })
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    return grouped.reset_index()
```

### 5.3 Statistical Tests

```python
# statistical_tests.py
from scipy import stats

def test_rlvr_vs_sft_preservation(df: pd.DataFrame) -> Dict:
    """
    Test H1: RLVR preserves subspace better than SFT.
    """
    # Get preservation scores
    rlvr_pres = df[df['model'] == 'olmo3_rl_zero']['preservation_vs_base'].dropna()
    sft_pres = df[df['model'] == 'olmo3_instruct']['preservation_vs_base'].dropna()
    
    # Two-sample t-test (alternative: RLVR > SFT)
    t_stat, p_value = stats.ttest_ind(rlvr_pres, sft_pres, alternative='greater')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((rlvr_pres.var() + sft_pres.var()) / 2)
    cohens_d = (rlvr_pres.mean() - sft_pres.mean()) / pooled_std
    
    return {
        'test': 'RLVR vs SFT subspace preservation',
        'rlvr_mean': rlvr_pres.mean(),
        'sft_mean': sft_pres.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
    }


def test_effective_rank_difference(df: pd.DataFrame) -> Dict:
    """
    Test: Does RLVR have lower effective rank (more compressed)?
    """
    rlvr_rank = df[df['model'] == 'olmo3_rl_zero']['effective_rank']
    sft_rank = df[df['model'] == 'olmo3_instruct']['effective_rank']
    
    t_stat, p_value = stats.ttest_ind(rlvr_rank, sft_rank, alternative='less')
    
    return {
        'test': 'RLVR vs SFT effective rank',
        'rlvr_mean': rlvr_rank.mean(),
        'sft_mean': sft_rank.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
```

### 5.4 Visualization

```python
# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_preservation_by_layer(df: pd.DataFrame, output_path: str):
    """
    Plot subspace preservation vs layer for RLVR and SFT.
    
    Expected: RLVR shows higher preservation, especially in middle layers.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in ['olmo3_rl_zero', 'olmo3_instruct']:
        model_df = df[df['model'] == model]
        
        # Average across tasks
        layer_means = model_df.groupby('layer')['preservation_vs_base'].mean()
        layer_stds = model_df.groupby('layer')['preservation_vs_base'].std()
        
        label = 'RLVR' if 'rl' in model else 'SFT'
        ax.plot(layer_means.index, layer_means.values, label=label, linewidth=2)
        ax.fill_between(
            layer_means.index,
            layer_means - layer_stds,
            layer_means + layer_stds,
            alpha=0.2
        )
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Subspace Preservation (vs Base)')
    ax.set_title('Subspace Preservation by Layer: RLVR vs SFT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_effective_rank_heatmap(df: pd.DataFrame, output_path: str):
    """
    Heatmap of effective rank: models × layers.
    """
    pivot = df.pivot_table(
        values='effective_rank',
        index='model',
        columns='layer',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, ax=ax, cmap='viridis', cbar_kws={'label': 'Effective Rank'})
    ax.set_title('Effective Rank by Model and Layer')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

---

## Decision Points and Success Criteria

### End of Week 2: Go/No-Go on Data Collection

**Requirements to proceed:**
- [x] All three OLMo models load and generate coherent text
- [ ] Activation extraction produces expected shapes
- [ ] Storage pipeline works without data corruption
- [ ] Test collection on 10 samples completes in < 5 minutes

**If blocked:** Debug infrastructure before collecting large datasets.

### End of Week 4: Go/No-Go on Analysis

**Requirements to proceed:**
- [ ] All activations collected without errors
- [ ] Quality checks pass (no NaN, reasonable magnitudes)
- [ ] At least 400/500 samples per task usable

**If issues:** Fix collection bugs, re-run affected model/task combinations.

### End of Week 6: Decision on Phase 2

**Criteria for continuing to Phase 2 (trajectory analysis):**

| Outcome | Signal Strength | Action |
|---------|-----------------|--------|
| RLVR shows higher preservation with p < 0.01, d > 0.5 | Strong | Proceed to Phase 2 |
| RLVR shows higher preservation with p < 0.05, d > 0.3 | Moderate | Proceed cautiously |
| No significant difference | Weak | Pivot: analyze what DOES differ |
| SFT shows higher preservation | Unexpected | Investigate, may invalidate hypothesis |

**Secondary success criteria:**
- Effective rank shows systematic differences between paradigms
- Layer-wise analysis reveals consistent patterns across tasks
- Results replicate across task domains

---

## Compute and Storage Summary

### Compute Requirements

| Task | GPU Hours (A100) | Notes |
|------|------------------|-------|
| Model loading/verification | 2 | One-time setup |
| OLMo activation collection | 15-20 | 3 models × 3 tasks × 500 samples |
| SVD/geometric analysis | 5-10 | CPU-bound, can parallelize |
| Visualization/reporting | 2 | Minimal |
| **Phase 1 Total** | **25-35** | ~$50-70 at $2/hr |

### Storage Requirements

| Data | Size | Format |
|------|------|--------|
| OLMo activations (3 models × 3 tasks) | ~2.5 GB | HDF5, gzip compressed |
| Full trajectories (subset) | ~5 GB | HDF5 |
| Analysis outputs | ~100 MB | CSV, PNG |
| **Total** | **~8 GB** | |

---

## Appendix: File Organization

```
geometric_transfer/
├── README.md
├── requirements.txt
├── configs/
│   └── models.yaml
├── src/
│   ├── __init__.py
│   ├── activation_collector.py
│   ├── task_data.py
│   ├── geometric_measures.py
│   ├── analyze_geometry.py
│   ├── statistical_tests.py
│   └── visualize.py
├── scripts/
│   ├── verify_pipeline.py
│   ├── collect_activations.py
│   └── run_analysis.py
├── data/
│   ├── activations/
│   │   ├── olmo3_base/
│   │   ├── olmo3_instruct/
│   │   └── olmo3_rl_zero/
│   └── trajectories/
├── results/
│   ├── geometric_summary.csv
│   ├── statistical_tests.json
│   └── figures/
└── notebooks/
    └── exploration.ipynb
```

---

## Next Steps After Phase 1

If Phase 1 shows signal (geometric differences exist and are statistically significant):

1. **Phase 2**: Implement path signature analysis on full trajectories
2. **Phase 2**: Compute local Jacobian sensitivity measures

Note: 
Full Trajectory Storage Estimates
Dataset	Avg Tokens	Samples	Per Model (raw)	Per Model (gzip)
GSM8K	72	500	~9.4 GB	~3-4 GB
HumanEval	121	164	~5.2 GB	~2-2.5 GB
Total per model			~14.6 GB	~5-6 GB
Full collection (4 models × 2 tasks):

Raw: ~58 GB
Compressed: ~20-25 GB
Which isnt too much - but lets see. this is only for the 7b models - we might have to scale up to show perf on the bigger models. 
3. **Phase 3**: Correlate geometric measures with actual transfer performance

If Phase 1 shows no signal:

1. **Pivot A**: Investigate what DOES differ between RLVR and SFT (even if not geometric)
2. **Pivot B**: Test whether OLMo models are actually different in transfer behavior
3. **Pivot C**: Try different geometric measures (manifold curvature, topological features)
