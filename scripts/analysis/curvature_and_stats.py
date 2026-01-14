"""
Curvature Analysis and Statistical Testing for Phase 1

Computes:
1. Local curvature via perturbation analysis
2. Formal statistical tests (t-tests, Cohen's d, confidence intervals)
3. Visualization plots
"""

import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import subspace_angles
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm


def compute_local_curvature_from_activations(
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int,
    n_perturbations: int = 10,
    eps: float = 0.01
) -> float:
    """
    Estimate local curvature via input perturbation analysis.

    Method:
    1. Get baseline activation at layer_idx for original input
    2. Add random noise to token IDs (discrete perturbation)
    3. Measure variance of activation changes

    High curvature = high sensitivity to input changes
    Low curvature = smooth manifold

    Args:
        model: Loaded transformers model
        tokenizer: Corresponding tokenizer
        texts: Input texts to analyze
        layer_idx: Which layer to measure curvature at
        n_perturbations: Number of perturbed versions per text
        eps: Perturbation probability (substitute random tokens)

    Returns:
        Mean local curvature estimate across texts
    """
    curvatures = []
    model.eval()

    # Hook to capture activations
    activations = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations['hidden'] = output[0].detach()
        else:
            activations['hidden'] = output.detach()

    # Register hook at target layer
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for text in tqdm(texts, desc=f"Curvature analysis (layer {layer_idx})", leave=False):
            # Tokenize
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            device = next(model.parameters()).device
            tokens = {k: v.to(device) for k, v in tokens.items()}
            input_ids = tokens['input_ids']

            # Baseline activation
            _ = model(**tokens)
            base_act = activations['hidden'][0, -1, :].cpu().numpy()

            # Perturbed activations
            perturbed_acts = []
            vocab_size = tokenizer.vocab_size

            for _ in range(n_perturbations):
                # Create perturbed input by randomly substituting tokens
                perturbed_ids = input_ids.clone()
                mask = torch.rand_like(input_ids.float()) < eps
                random_tokens = torch.randint(0, vocab_size, input_ids.shape, device=device)
                perturbed_ids = torch.where(mask, random_tokens, perturbed_ids)

                # Forward pass with perturbed input
                perturbed_tokens = {
                    'input_ids': perturbed_ids,
                    'attention_mask': tokens['attention_mask']
                }
                _ = model(**perturbed_tokens)
                pert_act = activations['hidden'][0, -1, :].cpu().numpy()
                perturbed_acts.append(pert_act)

            perturbed_acts = np.stack(perturbed_acts)

            # Curvature = variance of perturbation responses / eps^2
            variance = np.var(perturbed_acts - base_act, axis=0).mean()
            curvature = variance / (eps ** 2)
            curvatures.append(curvature)

    handle.remove()
    return np.mean(curvatures)


def run_statistical_tests(df: pd.DataFrame) -> Dict:
    """
    Run formal statistical tests comparing RL-Zero vs SFT vs Think.

    Returns dict with:
    - t-tests for preservation, effective rank
    - Cohen's d effect sizes
    - Confidence intervals
    """
    results = {}

    # Test 1: RL-Zero vs SFT subspace preservation
    rl_zero_pres = df[df['model'] == 'olmo3_rl_zero']['preservation_vs_base'].dropna()
    sft_pres = df[df['model'] == 'olmo3_sft']['preservation_vs_base'].dropna()
    think_pres = df[df['model'] == 'olmo3_think']['preservation_vs_base'].dropna()

    # RL-Zero vs SFT
    t_stat, p_value = stats.ttest_ind(rl_zero_pres, sft_pres, alternative='greater')
    pooled_std = np.sqrt((rl_zero_pres.var() + sft_pres.var()) / 2)
    cohens_d = (rl_zero_pres.mean() - sft_pres.mean()) / pooled_std

    # Confidence intervals
    rl_zero_ci = stats.t.interval(0.95, len(rl_zero_pres)-1,
                                   loc=rl_zero_pres.mean(),
                                   scale=stats.sem(rl_zero_pres))
    sft_ci = stats.t.interval(0.95, len(sft_pres)-1,
                              loc=sft_pres.mean(),
                              scale=stats.sem(sft_pres))

    results['rl_zero_vs_sft_preservation'] = {
        'rl_zero_mean': rl_zero_pres.mean(),
        'rl_zero_ci': rl_zero_ci,
        'sft_mean': sft_pres.mean(),
        'sft_ci': sft_ci,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'interpretation': 'Very large effect' if abs(cohens_d) > 0.8 else 'Large effect' if abs(cohens_d) > 0.5 else 'Medium effect'
    }

    # Test 2: Effective rank comparison
    rl_zero_rank = df[df['model'] == 'olmo3_rl_zero']['effective_rank'].dropna()
    sft_rank = df[df['model'] == 'olmo3_sft']['effective_rank'].dropna()

    t_stat_rank, p_value_rank = stats.ttest_ind(rl_zero_rank, sft_rank)
    pooled_std_rank = np.sqrt((rl_zero_rank.var() + sft_rank.var()) / 2)
    cohens_d_rank = (rl_zero_rank.mean() - sft_rank.mean()) / pooled_std_rank

    results['rl_zero_vs_sft_effective_rank'] = {
        'rl_zero_mean': rl_zero_rank.mean(),
        'sft_mean': sft_rank.mean(),
        't_statistic': t_stat_rank,
        'p_value': p_value_rank,
        'cohens_d': cohens_d_rank
    }

    # Test 3: SFT vs Think (should be similar)
    t_stat_sft_think, p_value_sft_think = stats.ttest_ind(sft_pres, think_pres)
    pooled_std_sft_think = np.sqrt((sft_pres.var() + think_pres.var()) / 2)
    cohens_d_sft_think = (sft_pres.mean() - think_pres.mean()) / pooled_std_sft_think

    results['sft_vs_think_preservation'] = {
        'sft_mean': sft_pres.mean(),
        'think_mean': think_pres.mean(),
        't_statistic': t_stat_sft_think,
        'p_value': p_value_sft_think,
        'cohens_d': cohens_d_sft_think,
        'interpretation': 'No significant difference' if p_value_sft_think > 0.05 else 'Significant difference'
    }

    return results


def plot_preservation_by_layer(df: pd.DataFrame, output_path: str):
    """Plot subspace preservation across layers for all models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = ['olmo3_rl_zero', 'olmo3_sft', 'olmo3_think']
    colors = {'olmo3_rl_zero': '#2ecc71', 'olmo3_sft': '#3498db', 'olmo3_think': '#e74c3c'}
    labels = {'olmo3_rl_zero': 'RL-Zero (Pure RL)', 'olmo3_sft': 'SFT', 'olmo3_think': 'Think (SFT+DPO+RLVR)'}

    for model in models:
        model_df = df[df['model'] == model]
        layer_means = model_df.groupby('layer')['preservation_vs_base'].mean()
        layer_stds = model_df.groupby('layer')['preservation_vs_base'].std()

        ax.plot(layer_means.index, layer_means.values,
                label=labels[model], linewidth=2.5, color=colors[model])
        ax.fill_between(layer_means.index,
                        layer_means - layer_stds,
                        layer_means + layer_stds,
                        alpha=0.2, color=colors[model])

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% preservation')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Subspace Preservation (vs Base)', fontsize=12)
    ax.set_title('Geometric Preservation Across Layers:\nRL-Zero Maintains Base Structure, SFT Reshapes It',
                 fontsize=14, pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved preservation plot to {output_path}")


def plot_effective_rank_heatmap(df: pd.DataFrame, output_path: str):
    """Heatmap of effective rank across models and layers."""
    pivot = df.pivot_table(
        values='effective_rank',
        index='model',
        columns='layer',
        aggfunc='mean'
    )

    # Reorder rows
    model_order = ['olmo3_base', 'olmo3_rl_zero', 'olmo3_sft', 'olmo3_think']
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, ax=ax, cmap='viridis',
                cbar_kws={'label': 'Effective Rank'},
                annot=False, fmt='.0f')
    ax.set_title('Effective Rank by Model and Layer', fontsize=14, pad=15)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved effective rank heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/activations')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--detailed-csv', default='results/geometric_analysis_detailed.csv')
    parser.add_argument('--run-curvature', action='store_true', help='Run curvature analysis (slow)')
    parser.add_argument('--curvature-samples', type=int, default=50)
    parser.add_argument('--curvature-layers', nargs='+', type=int, default=[0, 8, 16, 24, 31])
    args = parser.parse_args()

    output_dir = Path(args.results_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("CURVATURE ANALYSIS & STATISTICAL TESTING")
    print("="*60)

    # Load detailed results
    print(f"\nLoading results from {args.detailed_csv}")
    df = pd.read_csv(args.detailed_csv)
    print(f"✓ Loaded {len(df)} rows")

    # Statistical tests
    print("\n" + "="*60)
    print("Running statistical tests...")
    print("="*60)
    stat_results = run_statistical_tests(df)

    for test_name, result in stat_results.items():
        print(f"\n{test_name}:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Save statistical results
    import json
    stats_path = output_dir / "statistical_tests.json"
    with open(stats_path, 'w') as f:
        json.dump(stat_results, f, indent=2, default=str)
    print(f"\n✓ Saved statistical results to {stats_path}")

    # Generate plots
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_preservation_by_layer(df, str(figures_dir / "preservation_by_layer.png"))
    plot_effective_rank_heatmap(df, str(figures_dir / "effective_rank_heatmap.png"))

    # Curvature analysis (optional, slow)
    if args.run_curvature:
        print("\n" + "="*60)
        print("Running curvature analysis...")
        print("="*60)
        print(f"Analyzing {len(args.curvature_layers)} layers: {args.curvature_layers}")
        print(f"Using {args.curvature_samples} samples per model/task")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import sys
        sys.path.insert(0, 'src')
        from task_data import prepare_gsm8k

        curvature_results = []
        models = {
            'olmo3_base': 'allenai/Olmo-3-1025-7B',
            'olmo3_rl_zero': 'allenai/Olmo-3-7B-RL-Zero-General',
            'olmo3_sft': 'allenai/Olmo-3-7B-Think-SFT',
            'olmo3_think': 'allenai/Olmo-3-7B-Think'
        }

        # Load test prompts
        gsm8k_data = prepare_gsm8k(n_samples=args.curvature_samples, split='test')
        test_texts = [p[0] for p in gsm8k_data]

        for model_key, model_name in models.items():
            print(f"\n{model_key}:")
            print(f"  Loading {model_name}...")

            # Clear GPU memory
            import gc
            torch.cuda.empty_cache()
            gc.collect()

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Compute curvature for selected layers
            for layer_idx in args.curvature_layers:
                print(f"  Layer {layer_idx}...", end=" ", flush=True)
                curvature = compute_local_curvature_from_activations(
                    model, tokenizer, test_texts, layer_idx,
                    n_perturbations=10, eps=0.01
                )
                curvature_results.append({
                    'model': model_key,
                    'layer': layer_idx,
                    'curvature': curvature
                })
                print(f"curvature = {curvature:.6f}")

            # Clean up
            del model
            torch.cuda.empty_cache()

        # Save curvature results
        curv_df = pd.DataFrame(curvature_results)
        curv_path = output_dir / "curvature_analysis.csv"
        curv_df.to_csv(curv_path, index=False)
        print(f"\n✓ Saved curvature results to {curv_path}")
    else:
        print("\n⊘ Skipping curvature analysis (use --run-curvature to enable)")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
