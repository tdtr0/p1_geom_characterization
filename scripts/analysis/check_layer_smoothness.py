#!/usr/bin/env python3
"""
Check if geometric properties vary smoothly across layers.
If large jumps exist between consecutive layers, half-layer sampling might miss them.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load detailed results
df = pd.read_csv('results/geometric_analysis_detailed.csv')

# Focus on fine-tuned models with preservation scores
ft_models = ['olmo3_sft', 'olmo3_rl_zero', 'olmo3_think']

print("=" * 80)
print("LAYER-WISE SMOOTHNESS ANALYSIS")
print("=" * 80)
print()

for model in ft_models:
    for task in ['gsm8k', 'humaneval', 'logiqa']:
        subset = df[(df['model'] == model) & (df['task'] == task)].sort_values('layer')

        if len(subset) == 0:
            continue

        layers = subset['layer'].values
        preservation = subset['preservation_vs_base'].values
        eff_rank = subset['effective_rank'].values

        # Calculate consecutive differences
        pres_diffs = np.abs(np.diff(preservation))
        rank_diffs = np.abs(np.diff(eff_rank))

        # Find large jumps (> 2 std from mean)
        pres_threshold = np.mean(pres_diffs) + 2 * np.std(pres_diffs)
        rank_threshold = np.mean(rank_diffs) + 2 * np.std(rank_diffs)

        pres_jumps = np.where(pres_diffs > pres_threshold)[0]
        rank_jumps = np.where(rank_diffs > rank_threshold)[0]

        print(f"\n{model} - {task}")
        print(f"  Preservation variance: {np.std(preservation):.2f}%")
        print(f"  Mean consecutive diff: {np.mean(pres_diffs):.2f}% ± {np.std(pres_diffs):.2f}%")
        print(f"  Max consecutive jump: {np.max(pres_diffs):.2f}% (layers {layers[np.argmax(pres_diffs)]} → {layers[np.argmax(pres_diffs)+1]})")

        if len(pres_jumps) > 0:
            print(f"  ⚠️  Large jumps at layers: {layers[pres_jumps]} → {layers[pres_jumps+1]}")
        else:
            print(f"  ✓ No large jumps detected (smooth progression)")

        # Check if even/odd layers would miss critical info
        even_layers = layers[::2]
        odd_layers = layers[1::2]

        even_pres = preservation[::2]
        odd_pres = preservation[1::2]

        full_mean = np.mean(preservation)
        even_mean = np.mean(even_pres)
        odd_mean = np.mean(odd_pres)

        print(f"  Even layers mean: {even_mean:.2f}% (bias: {even_mean - full_mean:+.2f}%)")
        print(f"  Odd layers mean:  {odd_mean:.2f}% (bias: {odd_mean - full_mean:+.2f}%)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Overall assessment
all_ft = df[df['model'].isin(ft_models)].copy()
all_ft = all_ft.sort_values(['model', 'task', 'layer'])

by_model_task = all_ft.groupby(['model', 'task'])

max_jumps = []
for (model, task), group in by_model_task:
    pres = group['preservation_vs_base'].values
    if len(pres) > 1:
        diffs = np.abs(np.diff(pres))
        max_jumps.append(np.max(diffs))

max_jump_overall = np.max(max_jumps)
mean_jump_overall = np.mean(max_jumps)

print(f"\nMax consecutive jump across all models/tasks: {max_jump_overall:.2f}%")
print(f"Mean max jump per model/task: {mean_jump_overall:.2f}%")

if max_jump_overall < 10:
    print("\n✓ RECOMMENDATION: Half-layer collection is SAFE")
    print("  Geometric properties vary smoothly - no critical transitions missed")
elif max_jump_overall < 20:
    print("\n⚠️  RECOMMENDATION: Half-layer collection is ACCEPTABLE")
    print("  Some variation exists but unlikely to miss major transitions")
else:
    print("\n❌ RECOMMENDATION: Use FULL layer collection")
    print("  Large non-linear transitions detected - half-sampling may miss critical dynamics")
