# Phase 2 & 3 Implementation Plan: Flow Analysis and Transfer Correlation

## Phase 1 Results Summary

Phase 1 established that RLVR and SFT produce measurably different activation geometry:

| Model | Preservation vs Base (GSM8K) | Preservation vs Base (HumanEval) |
|-------|------------------------------|----------------------------------|
| RL-Zero | 0.986 ± 0.011 | 0.974 ± 0.029 |
| SFT | 0.524 ± 0.126 | 0.503 ± 0.155 |
| Think | 0.504 ± 0.129 | 0.497 ± 0.160 |

**Key finding**: RL-Zero preserves ~98% of base model subspace structure. SFT variants preserve only ~50%. This replicates Jin et al.'s findings in a different model family.

---

# Phase 2: Flow and Trajectory Analysis

## Core Question

**Do RLVR and SFT produce different *dynamic* properties in activation space, beyond static geometry?**

Phase 1 measured static properties (subspace structure of final representations). Phase 2 measures how activations *evolve* through the model—the computational flow.

## What We're Measuring

### 2.1 Two Types of "Trajectory"

**Layer trajectory**: How the residual stream evolves across layers for a single token position.
- Shape: (n_layers, d_model) per sample
- Captures: The computational transformation from input to output

**Token trajectory**: How activations evolve across token positions at a fixed layer.
- Shape: (n_tokens, d_model) per sample per layer
- Captures: How context builds up during generation

For Phase 2, we focus on **layer trajectories** (more tractable, directly measures computational flow).

### 2.2 Flow Measures

| Measure | What It Captures | Computation |
|---------|------------------|-------------|
| **Local Jacobian spectral radius** | Sensitivity/stability at each layer transition | max(svdvals(∂h_{l+1}/∂h_l)) |
| **Path signature** | Shape of trajectory (curvature, winding, self-intersection) | signatory library, depth 3-4 |
| **Cross-domain signature consistency** | Do trajectories look similar across tasks? | cosine_sim(sig_task_i, sig_task_j) |
| **Sample variance** | For same input, how variable are trajectories? | var(signatures across samples) |

---

## Week 1-2: Trajectory Data Collection

### 2.1 Collection Strategy

We need full layer-by-layer activations, not just final layer.

```python
# trajectory_collector.py
import torch
import h5py
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Dict

class TrajectoryCollector:
    """
    Collects full layer trajectories for flow analysis.
    
    Key difference from Phase 1: We store activations at ALL layers
    for each sample, enabling layer-to-layer analysis.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        layers_to_sample: List[int] = None  # None = all layers
    ):
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            dtype=dtype
        )
        self.model.eval()
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        
        # Strategic layer sampling to reduce storage
        if layers_to_sample is None:
            # Default: every 4th layer + first and last
            self.layers = [0] + list(range(4, self.n_layers, 4)) + [self.n_layers - 1]
            self.layers = sorted(set(self.layers))
        else:
            self.layers = layers_to_sample
    
    def get_trajectory_hooks(self) -> List[str]:
        """Return hook names for sampled layers."""
        return [f"blocks.{l}.hook_resid_post" for l in self.layers]
    
    def collect_trajectories(
        self,
        texts: List[str],
        token_position: str = "last"  # "last", "mean", or "all"
    ) -> Dict[str, np.ndarray]:
        """
        Collect layer trajectories for each input.
        
        Returns:
            trajectories: Dict with key "trajectories"
                Shape: (n_samples, n_layers_sampled, d_model)
        """
        hook_names = self.get_trajectory_hooks()
        all_trajectories = []
        
        with torch.no_grad():
            for text in texts:
                tokens = self.model.to_tokens(text)
                _, cache = self.model.run_with_cache(tokens, names_filter=hook_names)
                
                # Build trajectory: (n_layers, d_model)
                layer_acts = []
                for hook in hook_names:
                    act = cache[hook]  # (1, seq_len, d_model)
                    
                    if token_position == "last":
                        act = act[0, -1, :]  # (d_model,)
                    elif token_position == "mean":
                        act = act[0].mean(dim=0)  # (d_model,)
                    else:  # "all" - for token trajectory analysis
                        act = act[0]  # (seq_len, d_model)
                    
                    layer_acts.append(act.cpu().numpy())
                
                # Stack into trajectory
                if token_position in ["last", "mean"]:
                    trajectory = np.stack(layer_acts)  # (n_layers, d_model)
                else:
                    trajectory = layer_acts  # List of (seq_len, d_model)
                
                all_trajectories.append(trajectory)
        
        if token_position in ["last", "mean"]:
            return {"trajectories": np.stack(all_trajectories).astype(np.float16)}
        else:
            return {"trajectories": all_trajectories}  # Variable length
    
    def save_trajectories(
        self,
        trajectories: Dict,
        filepath: str,
        metadata: Dict = None
    ):
        """Save trajectories with metadata."""
        with h5py.File(filepath, 'w') as f:
            f.create_dataset(
                "trajectories", 
                data=trajectories["trajectories"],
                compression='gzip'
            )
            f.create_dataset("layers_sampled", data=self.layers)
            
            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = value
```

### 2.2 Collection Script

```python
# collect_trajectories.py
from trajectory_collector import TrajectoryCollector
from task_data import TASKS, MODELS
import os
from tqdm import tqdm

def collect_all_trajectories(
    output_dir: str,
    models: List[str] = ["olmo3_base", "olmo3_rl_zero", "olmo3_sft"],
    tasks: List[str] = ["gsm8k", "humaneval"],
    n_samples: int = 200,  # Reduced from 500 for storage
    layers: List[int] = [0, 8, 16, 24, 31]  # Strategic sampling
):
    """
    Collect trajectories for all model/task combinations.
    
    Storage estimate:
    200 samples × 5 layers × 4096 dims × 2 bytes × 3 models × 2 tasks
    ≈ 100 MB total (very manageable)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Collecting trajectories: {model_name}")
        print(f"{'='*50}")
        
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        collector = TrajectoryCollector(
            MODELS[model_name],
            layers_to_sample=layers
        )
        
        for task_name in tasks:
            output_path = os.path.join(model_dir, f"{task_name}_trajectories.h5")
            
            if os.path.exists(output_path):
                print(f"  {task_name}: exists, skipping")
                continue
            
            print(f"  {task_name}: collecting...")
            
            # Get prompts
            prompts_data = TASKS[task_name](n_samples)
            prompts = [p[0] for p in prompts_data[:n_samples]]
            
            # Collect
            trajectories = collector.collect_trajectories(prompts, token_position="last")
            
            # Save
            collector.save_trajectories(
                trajectories,
                output_path,
                metadata={
                    "model": model_name,
                    "task": task_name,
                    "n_samples": len(prompts),
                    "token_position": "last"
                }
            )
            print(f"    Saved: {output_path}")
        
        del collector
        torch.cuda.empty_cache()

if __name__ == "__main__":
    collect_all_trajectories("data/trajectories")
```

---

## Week 3-4: Flow Measure Computation

### 3.1 Path Signatures

Path signatures are reparameterization-invariant features that capture trajectory shape.

```python
# flow_measures.py
import torch
import numpy as np
from typing import Tuple, List, Dict
import signatory

def compute_path_signature(
    trajectory: np.ndarray,
    depth: int = 3,
    proj_dim: int = 64
) -> np.ndarray:
    """
    Compute path signature for a layer trajectory.
    
    Args:
        trajectory: (n_layers, d_model) - path through activation space
        depth: Signature truncation depth (3-4 typical)
        proj_dim: Project to lower dim before signature (required for d_model=4096)
    
    Returns:
        signature: (sig_dim,) where sig_dim depends on depth and proj_dim
    """
    # Random projection to reduce dimensionality
    # Fixed seed for reproducibility
    np.random.seed(42)
    proj_matrix = np.random.randn(trajectory.shape[1], proj_dim) / np.sqrt(proj_dim)
    
    # Project trajectory
    projected = trajectory @ proj_matrix  # (n_layers, proj_dim)
    
    # Convert to torch and add batch dimension
    path = torch.tensor(projected, dtype=torch.float32).unsqueeze(0)  # (1, n_layers, proj_dim)
    
    # Compute signature
    sig = signatory.signature(path, depth=depth)
    
    return sig.squeeze(0).numpy()


def compute_signature_stats(
    trajectories: np.ndarray,
    depth: int = 3,
    proj_dim: int = 64
) -> Dict[str, np.ndarray]:
    """
    Compute signature statistics for a batch of trajectories.
    
    Args:
        trajectories: (n_samples, n_layers, d_model)
    
    Returns:
        Dict with:
            - signatures: (n_samples, sig_dim)
            - mean_signature: (sig_dim,)
            - signature_variance: scalar
    """
    signatures = []
    for traj in trajectories:
        sig = compute_path_signature(traj, depth=depth, proj_dim=proj_dim)
        signatures.append(sig)
    
    signatures = np.stack(signatures)
    
    return {
        "signatures": signatures,
        "mean_signature": signatures.mean(axis=0),
        "signature_variance": signatures.var(axis=0).mean(),  # Scalar summary
        "signature_std": signatures.std(axis=0)
    }


def compute_cross_domain_consistency(
    signatures_by_task: Dict[str, np.ndarray]
) -> float:
    """
    Measure how consistent signatures are across different tasks.
    
    High consistency = trajectories look similar regardless of task domain
    Low consistency = task-specific trajectory shapes
    
    Args:
        signatures_by_task: {"gsm8k": (n_samples, sig_dim), "humaneval": ...}
    
    Returns:
        Mean pairwise cosine similarity between task mean signatures
    """
    tasks = list(signatures_by_task.keys())
    
    # Compute mean signature per task
    mean_sigs = {task: sigs.mean(axis=0) for task, sigs in signatures_by_task.items()}
    
    # Pairwise cosine similarity
    similarities = []
    for i, task_i in enumerate(tasks):
        for task_j in tasks[i+1:]:
            sig_i = mean_sigs[task_i]
            sig_j = mean_sigs[task_j]
            cos_sim = np.dot(sig_i, sig_j) / (np.linalg.norm(sig_i) * np.linalg.norm(sig_j) + 1e-8)
            similarities.append(cos_sim)
    
    return np.mean(similarities) if similarities else 0.0
```

### 3.2 Local Jacobian Analysis

```python
def estimate_layer_jacobian(
    model,
    activation: torch.Tensor,
    layer_idx: int,
    eps: float = 1e-4,
    n_samples: int = 100  # Sample dimensions for efficiency
) -> np.ndarray:
    """
    Estimate Jacobian of layer transition via finite differences.
    
    Full Jacobian is O(d^2) = 16M parameters. We sample dimensions.
    
    Args:
        model: HookedTransformer
        activation: (d_model,) activation at layer_idx
        layer_idx: Which layer transition to analyze
        eps: Perturbation size
        n_samples: Number of input dimensions to sample
    
    Returns:
        Sampled Jacobian columns: (d_model, n_samples)
    """
    d_model = activation.shape[0]
    
    # Sample dimensions to perturb
    sample_dims = np.random.choice(d_model, n_samples, replace=False)
    
    jacobian_samples = []
    
    with torch.no_grad():
        # Base output (run from layer_idx to layer_idx+1)
        # This requires hooks to inject activation and read output
        base_output = run_single_layer(model, activation, layer_idx)
        
        for dim in sample_dims:
            # Perturb
            perturbed = activation.clone()
            perturbed[dim] += eps
            
            # Forward
            perturbed_output = run_single_layer(model, perturbed, layer_idx)
            
            # Gradient estimate
            grad = (perturbed_output - base_output) / eps
            jacobian_samples.append(grad.cpu().numpy())
    
    return np.stack(jacobian_samples, axis=1)  # (d_model, n_samples)


def run_single_layer(model, activation, layer_idx):
    """
    Run a single layer transition.
    
    Requires model surgery - inject activation at layer_idx,
    read output at layer_idx + 1.
    """
    # Implementation depends on TransformerLens internals
    # Basic approach: use hooks to inject and read
    
    result = None
    
    def inject_hook(tensor, hook):
        return activation.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    
    def read_hook(tensor, hook):
        nonlocal result
        result = tensor[0, 0, :].clone()
        return tensor
    
    with model.hooks(
        fwd_hooks=[
            (f"blocks.{layer_idx}.hook_resid_pre", inject_hook),
            (f"blocks.{layer_idx}.hook_resid_post", read_hook)
        ]
    ):
        # Dummy forward
        model.forward(torch.zeros(1, 1, dtype=torch.long, device=model.device))
    
    return result


def compute_local_sensitivity(
    jacobian_samples: np.ndarray
) -> Dict[str, float]:
    """
    Compute sensitivity metrics from sampled Jacobian.
    
    Returns:
        max_singular_value: Approximates spectral radius (stability)
        mean_singular_value: Average amplification
        condition_estimate: Ratio of max to min singular value
    """
    # SVD of sampled Jacobian
    _, s, _ = np.linalg.svd(jacobian_samples, full_matrices=False)
    
    return {
        "max_sv": s[0],
        "mean_sv": s.mean(),
        "condition_estimate": s[0] / (s[-1] + 1e-10)
    }


def compute_flow_stability_profile(
    model,
    trajectories: np.ndarray,
    layers: List[int],
    n_samples_per_trajectory: int = 10,
    n_dim_samples: int = 50
) -> Dict[str, np.ndarray]:
    """
    Compute stability profile across layers for a set of trajectories.
    
    Args:
        model: HookedTransformer
        trajectories: (n_trajectories, n_layers, d_model)
        layers: Which layers were sampled
        
    Returns:
        stability_profile: (n_layer_transitions,) max singular values
        stability_variance: (n_layer_transitions,) variance across samples
    """
    n_transitions = len(layers) - 1
    
    sensitivities = [[] for _ in range(n_transitions)]
    
    # Sample trajectories
    traj_indices = np.random.choice(
        len(trajectories), 
        min(n_samples_per_trajectory, len(trajectories)),
        replace=False
    )
    
    for traj_idx in traj_indices:
        trajectory = trajectories[traj_idx]
        
        for trans_idx in range(n_transitions):
            layer_idx = layers[trans_idx]
            activation = torch.tensor(
                trajectory[trans_idx], 
                dtype=torch.float32,
                device=model.device
            )
            
            # Estimate Jacobian
            jac = estimate_layer_jacobian(
                model, activation, layer_idx, 
                n_samples=n_dim_samples
            )
            
            # Get sensitivity
            sens = compute_local_sensitivity(jac)
            sensitivities[trans_idx].append(sens["max_sv"])
    
    # Aggregate
    stability_profile = np.array([np.mean(s) for s in sensitivities])
    stability_variance = np.array([np.var(s) for s in sensitivities])
    
    return {
        "stability_profile": stability_profile,
        "stability_variance": stability_variance,
        "layer_transitions": [(layers[i], layers[i+1]) for i in range(n_transitions)]
    }
```

---

## Week 5-6: Statistical Analysis

### 5.1 Comparing Flow Measures Across Models

```python
# analyze_flow.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List
import h5py

def load_trajectories(data_dir: str, models: List[str], tasks: List[str]) -> Dict:
    """Load all trajectory data."""
    data = {}
    for model in models:
        data[model] = {}
        for task in tasks:
            path = f"{data_dir}/{model}/{task}_trajectories.h5"
            with h5py.File(path, 'r') as f:
                data[model][task] = {
                    "trajectories": f["trajectories"][:],
                    "layers": f["layers_sampled"][:]
                }
    return data


def analyze_all_flow_measures(
    data: Dict,
    models: List[str],
    tasks: List[str]
) -> pd.DataFrame:
    """
    Compute all flow measures for all model/task combinations.
    """
    results = []
    
    for model in models:
        # Collect signatures per task
        signatures_by_task = {}
        
        for task in tasks:
            trajectories = data[model][task]["trajectories"]
            
            # Compute signatures
            sig_stats = compute_signature_stats(trajectories, depth=3, proj_dim=64)
            signatures_by_task[task] = sig_stats["signatures"]
            
            results.append({
                "model": model,
                "task": task,
                "signature_variance": sig_stats["signature_variance"],
                "signature_norm_mean": np.linalg.norm(sig_stats["mean_signature"]),
            })
        
        # Cross-domain consistency (per model, across tasks)
        consistency = compute_cross_domain_consistency(signatures_by_task)
        
        # Add to all task rows for this model
        for r in results:
            if r["model"] == model:
                r["cross_domain_consistency"] = consistency
    
    return pd.DataFrame(results)


def test_flow_hypotheses(df: pd.DataFrame) -> Dict:
    """
    Statistical tests for Phase 2 hypotheses.
    """
    results = {}
    
    # H3: Trajectory Consistency - RLVR has lower signature variance
    rlvr_var = df[df['model'] == 'olmo3_rl_zero']['signature_variance'].values
    sft_var = df[df['model'] == 'olmo3_sft']['signature_variance'].values
    
    t_stat, p_val = stats.ttest_ind(rlvr_var, sft_var, alternative='less')
    results['signature_variance_test'] = {
        'rlvr_mean': rlvr_var.mean(),
        'sft_mean': sft_var.mean(),
        't_statistic': t_stat,
        'p_value': p_val,
        'significant': p_val < 0.05
    }
    
    # H3: Cross-domain consistency - RLVR has higher consistency
    rlvr_cons = df[df['model'] == 'olmo3_rl_zero']['cross_domain_consistency'].iloc[0]
    sft_cons = df[df['model'] == 'olmo3_sft']['cross_domain_consistency'].iloc[0]
    
    results['cross_domain_consistency'] = {
        'rlvr': rlvr_cons,
        'sft': sft_cons,
        'difference': rlvr_cons - sft_cons
    }
    
    return results
```

### 5.2 Phase 2 Success Criteria

| Outcome | Signal | Action |
|---------|--------|--------|
| RLVR shows lower signature variance (p < 0.05) | Strong | Proceed to Phase 3 |
| RLVR shows higher cross-domain consistency | Strong | Proceed to Phase 3 |
| Stability profiles differ systematically | Moderate | Proceed to Phase 3 |
| No significant differences in flow measures | Weak | Reconsider - static geometry may be sufficient |

---

# Phase 3: Transfer Correlation

## Core Question

**Do the geometric/flow measures we've computed actually predict transfer performance?**

This is where we connect representation structure to behavioral outcomes.

## What "Transfer" Means Operationally

**Transfer score**: Performance gain on target task that isn't explained by source task performance.

```
Transfer(source → target) = Accuracy(target) - f(Accuracy(source))
```

Where f() controls for baseline capability. Simplest: just use target accuracy and control for source in regression.

---

## Week 1-2: Transfer Measurement

### 1.1 Task Selection for Transfer Matrix

| Source Domain | Target Domains | Why |
|---------------|----------------|-----|
| GSM8K (math) | MATH, HumanEval, LogiQA | Math → harder math, code, logic |
| HumanEval (code) | MBPP, GSM8K, LogiQA | Code → different code, math, logic |
| LogiQA (logic) | FOLIO, GSM8K, HumanEval | Logic → formal logic, math, code |

### 1.2 Evaluation Script

```python
# evaluate_transfer.py
import numpy as np
from typing import Dict, List, Tuple
from datasets import load_dataset
import json

# Evaluation functions for each task
def evaluate_gsm8k(model, tokenizer, n_samples: int = 200) -> float:
    """Evaluate on GSM8K, return accuracy."""
    ds = load_dataset("gsm8k", "main", split="test")
    
    correct = 0
    for item in ds.select(range(min(n_samples, len(ds)))):
        prompt = f"Solve step by step:\n{item['question']}\nAnswer:"
        
        # Generate
        response = generate(model, tokenizer, prompt, max_tokens=512)
        
        # Extract final answer (GSM8K format: #### <number>)
        pred_answer = extract_gsm8k_answer(response)
        true_answer = extract_gsm8k_answer(item['answer'])
        
        if pred_answer == true_answer:
            correct += 1
    
    return correct / n_samples


def evaluate_humaneval(model, tokenizer, n_samples: int = 164) -> float:
    """Evaluate on HumanEval, return pass@1."""
    ds = load_dataset("openai_humaneval", split="test")
    
    passed = 0
    for item in ds.select(range(min(n_samples, len(ds)))):
        prompt = item['prompt']
        
        # Generate completion
        completion = generate(model, tokenizer, prompt, max_tokens=256, stop="\ndef")
        
        # Test
        full_code = prompt + completion
        if run_tests(full_code, item['test']):
            passed += 1
    
    return passed / n_samples


def evaluate_math(model, tokenizer, n_samples: int = 200) -> float:
    """Evaluate on MATH dataset (harder than GSM8K)."""
    ds = load_dataset("hendrycks/competition_math", split="test")
    
    correct = 0
    for item in ds.select(range(min(n_samples, len(ds)))):
        prompt = f"Problem: {item['problem']}\n\nSolution:"
        response = generate(model, tokenizer, prompt, max_tokens=1024)
        
        if check_math_answer(response, item['solution']):
            correct += 1
    
    return correct / n_samples


def evaluate_mbpp(model, tokenizer, n_samples: int = 200) -> float:
    """Evaluate on MBPP (different code style than HumanEval)."""
    ds = load_dataset("mbpp", split="test")
    
    passed = 0
    for item in ds.select(range(min(n_samples, len(ds)))):
        prompt = f"# {item['text']}\ndef solution("
        completion = generate(model, tokenizer, prompt, max_tokens=256)
        
        full_code = prompt + completion
        if run_mbpp_tests(full_code, item['test_list']):
            passed += 1
    
    return passed / n_samples


EVAL_FUNCTIONS = {
    "gsm8k": evaluate_gsm8k,
    "humaneval": evaluate_humaneval,
    "math": evaluate_math,
    "mbpp": evaluate_mbpp,
    # Add more as needed
}


def compute_transfer_matrix(
    models: List[str],
    source_tasks: List[str],
    target_tasks: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute full transfer matrix.
    
    Returns:
        {model: {source: {target: accuracy}}}
    """
    results = {}
    
    for model_name in models:
        print(f"\nEvaluating: {model_name}")
        model, tokenizer = load_model(model_name)
        
        results[model_name] = {}
        
        all_tasks = set(source_tasks) | set(target_tasks)
        
        for task in all_tasks:
            print(f"  {task}...")
            eval_fn = EVAL_FUNCTIONS[task]
            accuracy = eval_fn(model, tokenizer)
            results[model_name][task] = accuracy
            print(f"    Accuracy: {accuracy:.3f}")
        
        del model
        torch.cuda.empty_cache()
    
    return results
```

### 1.3 Transfer Score Computation

```python
def compute_transfer_scores(
    accuracy_matrix: Dict,
    source_tasks: List[str],
    target_tasks: List[str]
) -> pd.DataFrame:
    """
    Compute transfer scores from accuracy matrix.
    
    Transfer score = target accuracy (we'll control for source in regression)
    
    For more sophisticated version:
    Transfer score = target accuracy - baseline (random or zero-shot)
    """
    rows = []
    
    for model, accuracies in accuracy_matrix.items():
        for source in source_tasks:
            for target in target_tasks:
                if source == target:
                    continue  # Skip same-task
                
                rows.append({
                    "model": model,
                    "source": source,
                    "target": target,
                    "source_accuracy": accuracies.get(source, np.nan),
                    "target_accuracy": accuracies.get(target, np.nan),
                    "transfer_pair": f"{source}→{target}"
                })
    
    return pd.DataFrame(rows)
```

---

## Week 3-4: Correlation Analysis

### 3.1 Combining Geometric Measures with Transfer

```python
# correlate_geometry_transfer.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

def merge_geometry_and_transfer(
    geometry_df: pd.DataFrame,  # From Phase 1
    flow_df: pd.DataFrame,       # From Phase 2
    transfer_df: pd.DataFrame    # From Phase 3 eval
) -> pd.DataFrame:
    """
    Merge all data sources for correlation analysis.
    """
    # Aggregate geometry to model level (mean across tasks/layers)
    geo_agg = geometry_df.groupby('model').agg({
        'effective_rank': 'mean',
        'spectral_decay_alpha': 'mean',
        'preservation_vs_base': 'mean'
    }).reset_index()
    
    # Aggregate flow to model level
    flow_agg = flow_df.groupby('model').agg({
        'signature_variance': 'mean',
        'cross_domain_consistency': 'first'  # Same for all tasks
    }).reset_index()
    
    # Merge with transfer
    merged = transfer_df.merge(geo_agg, on='model')
    merged = merged.merge(flow_agg, on='model')
    
    return merged


def analyze_transfer_predictors(merged_df: pd.DataFrame) -> Dict:
    """
    Which geometric/flow measures predict transfer?
    """
    results = {}
    
    # Predictors
    predictors = [
        'effective_rank',
        'spectral_decay_alpha', 
        'preservation_vs_base',
        'signature_variance',
        'cross_domain_consistency',
        'source_accuracy'  # Control variable
    ]
    
    target = 'target_accuracy'
    
    # Filter to complete cases
    df = merged_df.dropna(subset=predictors + [target])
    
    if len(df) < 5:
        return {"error": "Not enough data points"}
    
    # 1. Univariate correlations
    correlations = {}
    for pred in predictors:
        r, p = stats.pearsonr(df[pred], df[target])
        correlations[pred] = {"r": r, "p": p}
    results["univariate_correlations"] = correlations
    
    # 2. Partial correlations (controlling for source accuracy)
    # For each predictor, correlate residuals after regressing out source
    partial_corrs = {}
    source_control = df['source_accuracy'].values.reshape(-1, 1)
    
    for pred in predictors:
        if pred == 'source_accuracy':
            continue
        
        # Residualize predictor
        reg = LinearRegression().fit(source_control, df[pred])
        pred_resid = df[pred] - reg.predict(source_control)
        
        # Residualize target
        reg = LinearRegression().fit(source_control, df[target])
        target_resid = df[target] - reg.predict(source_control)
        
        # Correlate residuals
        r, p = stats.pearsonr(pred_resid, target_resid)
        partial_corrs[pred] = {"r": r, "p": p}
    
    results["partial_correlations"] = partial_corrs
    
    # 3. Multiple regression
    X = df[predictors].values
    y = df[target].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    reg = Ridge(alpha=1.0).fit(X_scaled, y)
    
    results["regression"] = {
        "r_squared": reg.score(X_scaled, y),
        "coefficients": dict(zip(predictors, reg.coef_))
    }
    
    return results
```

### 3.2 Key Analyses

```python
def run_phase3_analyses(
    geometry_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    transfer_df: pd.DataFrame
) -> Dict:
    """
    Main Phase 3 analysis pipeline.
    """
    results = {}
    
    # Merge data
    merged = merge_geometry_and_transfer(geometry_df, flow_df, transfer_df)
    
    # 1. Does subspace preservation predict transfer?
    results["preservation_analysis"] = analyze_preservation_transfer(merged)
    
    # 2. Does flow consistency predict transfer?
    results["flow_analysis"] = analyze_flow_transfer(merged)
    
    # 3. Combined model
    results["combined_model"] = analyze_transfer_predictors(merged)
    
    # 4. Per transfer-pair analysis (which transfers are predicted?)
    results["per_pair"] = {}
    for pair in merged['transfer_pair'].unique():
        pair_df = merged[merged['transfer_pair'] == pair]
        if len(pair_df) >= 3:  # Need enough models
            results["per_pair"][pair] = analyze_transfer_predictors(pair_df)
    
    return results


def analyze_preservation_transfer(df: pd.DataFrame) -> Dict:
    """
    Primary hypothesis: preservation_vs_base predicts transfer.
    """
    # Correlation
    r, p = stats.pearsonr(df['preservation_vs_base'], df['target_accuracy'])
    
    # Controlling for source
    X = df[['preservation_vs_base', 'source_accuracy']].values
    y = df['target_accuracy'].values
    
    reg = LinearRegression().fit(X, y)
    
    return {
        "raw_correlation": {"r": r, "p": p},
        "controlled_coefficient": reg.coef_[0],  # Preservation coefficient
        "r_squared": reg.score(X, y),
        "interpretation": "positive" if reg.coef_[0] > 0 else "negative"
    }
```

---

## Week 5-6: Validation and Interpretation

### 5.1 Cross-Validation

```python
def cross_validate_predictor(
    merged_df: pd.DataFrame,
    predictor: str,
    n_folds: int = 5
) -> Dict:
    """
    Leave-one-out cross-validation for transfer prediction.
    """
    from sklearn.model_selection import LeaveOneOut
    
    loo = LeaveOneOut()
    
    predictions = []
    actuals = []
    
    X = merged_df[[predictor, 'source_accuracy']].values
    y = merged_df['target_accuracy'].values
    
    for train_idx, test_idx in loo.split(X):
        reg = LinearRegression().fit(X[train_idx], y[train_idx])
        pred = reg.predict(X[test_idx])
        
        predictions.append(pred[0])
        actuals.append(y[test_idx][0])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    r, _ = stats.pearsonr(predictions, actuals)
    
    return {
        "mse": mse,
        "mae": mae,
        "correlation": r,
        "predictions": predictions,
        "actuals": actuals
    }
```

### 5.2 Visualization

```python
# visualize_phase3.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_preservation_vs_transfer(merged_df: pd.DataFrame, output_path: str):
    """
    Key figure: Does preservation predict transfer?
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Raw correlation
    ax = axes[0]
    for model in merged_df['model'].unique():
        model_df = merged_df[merged_df['model'] == model]
        ax.scatter(
            model_df['preservation_vs_base'], 
            model_df['target_accuracy'],
            label=model, s=50
        )
    
    ax.set_xlabel('Subspace Preservation vs Base')
    ax.set_ylabel('Transfer Accuracy')
    ax.set_title('Subspace Preservation Predicts Transfer')
    ax.legend()
    
    # Right: By transfer pair
    ax = axes[1]
    for pair in merged_df['transfer_pair'].unique():
        pair_df = merged_df[merged_df['transfer_pair'] == pair]
        ax.scatter(
            pair_df['preservation_vs_base'],
            pair_df['target_accuracy'],
            label=pair, alpha=0.7
        )
    
    ax.set_xlabel('Subspace Preservation vs Base')
    ax.set_ylabel('Transfer Accuracy')
    ax.set_title('Transfer by Domain Pair')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_geometry_flow_comparison(results: Dict, output_path: str):
    """
    Which measures predict transfer best?
    """
    # Extract partial correlations
    partial_corrs = results["partial_correlations"]
    
    predictors = list(partial_corrs.keys())
    r_values = [partial_corrs[p]["r"] for p in predictors]
    p_values = [partial_corrs[p]["p"] for p in predictors]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]
    bars = ax.barh(predictors, r_values, color=colors)
    
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Partial Correlation with Transfer (controlling for source)')
    ax.set_title('Which Geometric/Flow Measures Predict Transfer?')
    
    # Add significance markers
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        if p < 0.01:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, ' **', 
                   va='center', fontsize=12)
        elif p < 0.05:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, ' *',
                   va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

---

## Success Criteria: Phase 3

| Outcome | What It Means | Action |
|---------|---------------|--------|
| Preservation correlates with transfer (r > 0.3, p < 0.05) | Static geometry predicts transfer | Strong result, write up |
| Flow measures add predictive power beyond preservation | Dynamic properties matter | Even stronger result |
| No geometric measures predict transfer | Geometry ≠ transfer capability | Negative result, but informative |
| Geometry only predicts certain transfer pairs | Domain-specific effects | Nuanced finding, analyze further |

---

## Compute and Storage Summary

### Phase 2

| Task | GPU Hours | Storage |
|------|-----------|---------|
| Trajectory collection | 10-15 | ~100 MB |
| Signature computation | 2-3 (CPU) | ~50 MB |
| Jacobian estimation | 20-30 | Minimal |
| **Total** | **~45** | **~200 MB** |

### Phase 3

| Task | GPU Hours | Storage |
|------|-----------|---------|
| Transfer evaluation (4 models × 6 tasks) | 40-60 | Minimal |
| Correlation analysis | 1 (CPU) | Minimal |
| **Total** | **~60** | **Minimal** |

### Combined Phase 2+3

| | GPU Hours | Cost @ $2/hr |
|---|-----------|--------------|
| **Phase 2** | 45 | $90 |
| **Phase 3** | 60 | $120 |
| **Buffer** | 20 | $40 |
| **Total** | **125** | **$250** |

---

## Timeline

### Phase 2 (4 weeks)
- Week 1-2: Trajectory data collection
- Week 3-4: Flow measure computation
- Week 5-6: Statistical analysis, decision point

### Phase 3 (4 weeks)
- Week 1-2: Transfer evaluation (expensive)
- Week 3-4: Correlation analysis
- Week 5-6: Validation, visualization, writeup

---

## Deliverables

### Phase 2
- Trajectory dataset (shareable)
- Flow measure comparison: RLVR vs SFT
- Statistical tests for consistency hypotheses

### Phase 3
- Transfer matrix for all models
- Correlation: geometry/flow → transfer
- Key figures for paper
- Reproducibility package

---

## File Organization

```
geometric_transfer/
├── data/
│   ├── activations/          # Phase 1
│   ├── trajectories/         # Phase 2
│   │   ├── olmo3_base/
│   │   ├── olmo3_rl_zero/
│   │   └── olmo3_sft/
│   └── transfer_results/     # Phase 3
├── src/
│   ├── phase1/               # Existing
│   ├── phase2/
│   │   ├── trajectory_collector.py
│   │   ├── flow_measures.py
│   │   └── analyze_flow.py
│   └── phase3/
│       ├── evaluate_transfer.py
│       ├── correlate_geometry_transfer.py
│       └── visualize_phase3.py
├── results/
│   ├── phase1_geometry.csv
│   ├── phase2_flow.csv
│   ├── phase3_transfer.csv
│   └── figures/
└── notebooks/
    ├── phase2_exploration.ipynb
    └── phase3_analysis.ipynb
```
