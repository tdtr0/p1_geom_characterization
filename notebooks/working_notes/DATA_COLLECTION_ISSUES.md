# Data Collection Issues Report

**Date**: 2026-01-20
**Purpose**: Document data issues for another agent/model to fix during recollection

---

## Executive Summary

The Phase 2 trajectory collection has significant data quality issues that need addressing before robust H1/H2 analysis:

1. **0-shot LogiQA**: 3 of 4 models have CORRUPTED files (truncated during upload/download)
2. **0-shot HumanEval**: 3 of 4 models severely IMBALANCED (<5% correct rate)
3. **8-shot HumanEval**: ALL 4 models MISSING (never collected)

---

## Detailed Data Audit (2026-01-20)

### 0-shot Data (`/data/thanhdo/trajectories_0shot/`)

| Model | GSM8K | HumanEval | LogiQA |
|-------|-------|-----------|--------|
| olmo3_base | 63/500 (12.6%) OK | 19/500 (3.8%) **IMBALANCED** | 127/500 (25.4%) OK |
| olmo3_sft | 297/500 (59.4%) OK | 24/500 (4.8%) **IMBALANCED** | **CORRUPTED** |
| olmo3_rl_zero | 70/500 (14.0%) OK | 67/500 (13.4%) OK | **CORRUPTED** |
| olmo3_think | 197/500 (39.4%) OK | 25/500 (5.0%) **IMBALANCED** | **CORRUPTED** |

### 8-shot Data (`/data/thanhdo/trajectories_8shot/`)

| Model | GSM8K | HumanEval | LogiQA |
|-------|-------|-----------|--------|
| olmo3_base | 369/500 (73.8%) OK | **MISSING** | 146/500 (29.2%) OK |
| olmo3_sft | 344/500 (68.8%) OK | **MISSING** | 98/500 (19.6%) OK |
| olmo3_rl_zero | 378/500 (75.6%) OK | **MISSING** | 146/500 (29.2%) OK |
| olmo3_think | 222/500 (44.4%) OK | **MISSING** | 137/500 (27.4%) OK |

---

## Issue Details

### Issue 1: 0-shot LogiQA Corrupted Files

**Affected files**:
- `olmo3_sft/logiqa_trajectories.h5` - truncated (eof=6215MB, expected=6396MB)
- `olmo3_rl_zero/logiqa_trajectories.h5` - truncated (eof=4968MB, expected=5080MB)
- `olmo3_think/logiqa_trajectories.h5` - truncated (eof=4715MB, expected=4763MB)

**Cause**: Likely interrupted upload to B2 or download from B2

**Fix Required**:
1. Check if valid files exist in B2 bucket (`b2://ml-activations-store/trajectories/`)
2. If B2 files are also corrupted, re-run collection on these 3 models
3. Collection script: `scripts/collection/collect_trajectories_with_labels.py`

**Only WORKING 0-shot LogiQA file**: `olmo3_base/logiqa_trajectories.h5`

---

### Issue 2: 0-shot HumanEval Severely Imbalanced

**Affected files** (all have <5% correct rate):
- `olmo3_base/humaneval_trajectories.h5` - 19/500 (3.8%)
- `olmo3_sft/humaneval_trajectories.h5` - 24/500 (4.8%)
- `olmo3_think/humaneval_trajectories.h5` - 25/500 (5.0%)

**Only balanced file**: `olmo3_rl_zero/humaneval_trajectories.h5` - 67/500 (13.4%)

**Cause**: HumanEval correctness check likely broken. The labels report only ~4% correct rate for base/SFT/think models, but RL-Zero shows 13.4%.

**Investigation needed**:
1. Check if the `is_correct` labels are actually correct by manually inspecting `model_outputs` vs `ground_truth`
2. The issue may be in how the correctness was computed (markdown extraction? syntax check?)
3. Review the correctness checking logic in the collection script

**Potential Fix**:
- Re-run correctness labeling on existing trajectories (don't need to recollect activations)
- Script should extract code from markdown blocks (`\`\`\`python ... \`\`\``) before syntax check
- May need actual execution against test cases for accurate labels

---

### Issue 3: 8-shot HumanEval Missing

**Affected**: ALL 4 models have no HumanEval data in 8-shot

**Cause**: 8-shot collection focused on GSM8K and LogiQA only. HumanEval was not collected.

**Fix Required**: Run HumanEval collection for all 4 models in 8-shot setting:
```bash
python scripts/collection/collect_trajectories_with_labels.py \
    --model <model_name> \
    --task humaneval \
    --n-shot 8 \
    --output-dir /data/thanhdo/trajectories_8shot/<model_name>/
```

---

## Recollection Priority

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| **HIGH** | 0-shot LogiQA (3 models corrupted) | 4-6 GPU hours | Critical for H2 cross-domain |
| **MEDIUM** | 0-shot HumanEval labels | ~1 hour (CPU only, relabel) | Better H1 analysis |
| **LOW** | 8-shot HumanEval | 2-4 GPU hours | Nice to have for completeness |

---

## Usable Data for Analysis

Despite issues, we have sufficient data for preliminary analysis:

### For H1 (Within-Domain Classification)
- **GSM8K**: All 4 models × 2 shots = 8 files (OK)
- **LogiQA**: 1 model (base) × 0-shot + 4 models × 8-shot = 5 files (OK)
- **HumanEval**: 1 model (rl_zero) × 0-shot = 1 file (OK, others need label fix)

### For H2 (Cross-Domain Transfer)
- **0-shot**: Only `olmo3_base` has all 3 tasks with valid data
- **8-shot**: All 4 models have GSM8K + LogiQA (no HumanEval)

### Current Analysis Approach
Given data limitations, we focused Phase 3 analysis on:
- `olmo3_base` 0-shot: HumanEval + LogiQA (both valid)
- This allows H2 cross-domain testing but on single model

---

## HDF5 File Structure Reference

Each trajectory file should have:
```
Keys: ['ground_truth', 'is_correct', 'model_outputs', 'prompts', 'sequence_lengths', 'trajectories']
  ground_truth: shape=(500,), dtype=object
  is_correct: shape=(500,), dtype=bool
  model_outputs: shape=(500,), dtype=object
  prompts: shape=(500,), dtype=object
  sequence_lengths: shape=(500,), dtype=int32
  trajectories: shape=(500, 512, 16, 4096), dtype=float16
```

---

## Commands for Recollection

### Check B2 for valid files
```bash
b2 ls ml-activations-store trajectories/0shot/olmo3_sft/
b2 ls ml-activations-store trajectories/0shot/olmo3_rl_zero/
b2 ls ml-activations-store trajectories/0shot/olmo3_think/
```

### Re-download from B2 (if valid)
```bash
python scripts/storage/b2_download.py \
    --remote-prefix trajectories/0shot/olmo3_sft/logiqa \
    --local-dir /data/thanhdo/trajectories_0shot/olmo3_sft/
```

### Recollect (if B2 also corrupted)
```bash
# On vast.ai or SLURM cluster with GPU
python scripts/collection/collect_logiqa_vllm_fully_optimized.py \
    --model olmo3_sft \
    --n-samples 500 \
    --output-dir /data/thanhdo/trajectories_0shot/olmo3_sft/
```

---

## Notes for Recollection Agent

1. **Environment**: Use eyecog server or SLURM cluster (H100 node)
2. **Conda**: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base`
3. **HDF5 locking**: Always set `export HDF5_USE_FILE_LOCKING=FALSE`
4. **Verify after collection**: Check file integrity with `h5py.File(path, 'r')`
5. **Upload to B2**: Run `python scripts/storage/b2_upload.py` after collection
