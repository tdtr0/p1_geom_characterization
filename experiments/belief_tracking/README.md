# Belief State Tracking Experiment

## Overview

Tracks belief state evolution per-clause within model generations to test whether RLVR and SFT models show different belief dynamics.

## Hypotheses

- **H_belief**: RLVR models show smooth, continuous belief evolution; SFT models show discrete jumps
- **H_style**: SFT layer jumps (L16-18 spike) might encode style (formatting), not reasoning
- **H_context**: (Separate) Models have shrinking "effective context window" during generation

## Key Insight

Token probability P(next_token | context) ≠ belief state P(task_success | understanding).

We use activation-based correctness probes to track **task-level belief**, not token-level probability.

## Methodology

### Phase 1: Bootstrap with Existing Data
1. Load HDF5 trajectories with model_outputs
2. Parse model_outputs into clauses (sentence boundaries + reasoning markers)
3. Train belief probe on mean final-layer activations
4. Apply retroactively to clause boundary positions
5. Compare belief dynamics: correct vs incorrect, rl_zero vs think vs sft

### Phase 2: Cross-Model Transfer
Train probe on model A, apply to model B. If transfer works (AUC > 0.6), belief state captures something universal.

## Files

- `analyze_belief_dynamics.py` - Main analysis script
- `run_belief_analysis.sbatch` - SLURM job script
- `results/` - Output directory

## Usage

```bash
# On SLURM cluster (ai_inst)
sbatch experiments/belief_tracking/run_belief_analysis.sbatch

# Monitor
tail -f ~/belief_tracking_out.txt
```

## Expected Outcomes

### If H_belief is TRUE:
- RLVR models: high smoothness, monotonic belief increase
- SFT models: low smoothness, discrete belief jumps

### If H_belief is FALSE:
- All models show smooth geodesic
- Clause-level may be too coarse; need attention analysis

## Connection to Previous Work

- Must beat static probe baseline (AUC ~0.75)
- Explains Wynroe findings (L16-18 spike = style vs reasoning?)
- Tests if sequence-wise dynamics succeed where layer-wise (Lyapunov) failed
