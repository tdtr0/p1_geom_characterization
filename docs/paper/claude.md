# Navigation Guide for LLMs: ManiVer Paper Development

**Last Updated**: 2026-01-12

---

## 🎯 Project Focus

**Main Research Question**: Can we learn the geometry of correct reasoning from verifiable domains and use it on non-verifiable domains?

**Core Hypotheses** (H1-H5 in `RESEARCH_PLAN.md`):
- H1: Correct vs incorrect trajectories are distinguishable
- H2: The signature is domain-invariant (CRITICAL TEST)
- H3: Detector works on non-verifiable domains
- H4: Trajectories can be steered (causal intervention)
- H5: Correct reasoning has lower curvature

---

## 📁 File Organization

### Research Documents (This Directory: `main/paper/`)

| File | Purpose | When to Read |
|------|---------|--------------|
| `RESEARCH_PLAN.md` | Main research plan with H1-H5, experimental design, timeline | Always start here |
| `LITERATURE_REVIEW_SHORT.md` | Concise lit review for each hypothesis (supporting + critical) | For quick reference |
| `LITERATURE_REVIEW_LONG.md` | Extended lit review with methodology and strategic assessment | For deep understanding |
| `geometric_compression_research_plan.md` | Background on RLVR vs SFT (NOT the main focus) | For context only |
| `claude.md` | This file - navigation instructions | Update when files change |

### Phase Plans (Parent Directory: `main/`)

| File | Phase | Status | Purpose |
|------|-------|--------|---------|
| `PHASE1_DETAILED_PLAN.md` | Phase 1 | ✅ Complete | Static geometry characterization (RLVR vs SFT) |
| `PHASE2_DETAILED_PLAN.md` | Phase 2 | 🔄 In Progress | Trajectory collection with correctness labels |
| `PHASE3_DETAILED_PLAN.md` | Phase 3 | ⏳ Pending | Cross-domain transfer testing (H2) |
| `PHASE4_DETAILED_PLAN.md` | Phase 4 | ⏳ Pending | Trajectory steering (H4 - causal test) |
| `PHASE5_DETAILED_PLAN.md` | Phase 5 | ⏳ Pending | Write-up and publication |

### Implementation Code (Parent Directory: `main/`)

| Directory/File | Purpose |
|----------------|---------|
| `src/activation_collector.py` | Collect activations using TransformerLens or transformers |
| `src/geometric_measures.py` | Compute effective rank, spectral decay, subspace preservation |
| `src/task_data.py` | Load datasets (GSM8K, HumanEval, LogiQA) |
| `scripts/collect_trajectories_with_labels.py` | Main collection script for Phase 2 |
| `scripts/run_analysis.py` | Geometric analysis pipeline |
| `configs/models.yaml` | Model configurations |

---

## 🔄 When You Update Files

### Always Update These Two Files

1. **This file** (`main/paper/claude.md`):
   - Add new files to the tables above
   - Update status markers (✅ ⏳ 🔄)
   - Add to File Update Log at bottom

2. **Master algorithm** (`main/master_algorithm.md`):
   - Add one-line description of new file's purpose
   - Update directory structure if needed

### File Update Log

**Format**: `YYYY-MM-DD: Brief description of changes`

**2026-01-12**: 
- Created PHASE1-5_DETAILED_PLAN.md (detailed week-by-week execution plans)
- Created LITERATURE_REVIEW_SHORT.md and LITERATURE_REVIEW_LONG.md
- Updated RESEARCH_PLAN.md with Novelty section, confound controls, baseline comparisons
- Fixed HumanEval correctness check in collect_trajectories_with_labels.py
- Moved PDFs from /Papers to read/ to lit_review/papers/
- Created this navigation file

---

## 🚨 Critical Reminders

### The Main Focus

**This project is about**: Correct vs incorrect reasoning geometry (H1-H5 from RESEARCH_PLAN.md)

**This project is NOT about**: RLVR vs SFT comparison (that was Phase 1 background)

### The Critical Test

**H2 (cross-domain transfer)** is the linchpin:
- If H2 succeeds → Universal reasoning geometry exists → Proceed to H4
- If H2 fails → Geometry is domain-specific → Pivot to characterizing differences

### Key Confounds to Control

1. **Problem difficulty** (easy problems may have different geometry)
2. **Output length** (correct/incorrect may differ in length)
3. **Surface format** (math vs code formatting)
4. **Random labels** (control task to validate probe)

### Baselines to Beat

- Model confidence (logit-based)
- Semantic entropy (sampling-based)
- Self-consistency (majority vote)
- Output length (simple heuristic)

---

## 📊 Current Status

**Phase 1**: ✅ Complete (showed RLVR vs SFT have different static geometry)

**Phase 2**: 🔄 In Progress (collecting trajectories with correctness labels)
- Need: 500 samples × 3 tasks × 4 models = 6,000 samples
- Storage: ~56 GB
- Timeline: Weeks 1-4

**Next**: Phase 3 (test H2 cross-domain transfer) - the critical test

---

## 🎓 For New LLMs Joining This Project

### Start Here

1. Read `RESEARCH_PLAN.md` (main hypotheses and experimental design)
2. Read `LITERATURE_REVIEW_SHORT.md` (supporting/critical evidence)
3. Read current phase plan (e.g., `PHASE2_DETAILED_PLAN.md`)
4. Check `master_algorithm.md` for complete file map

### Common Tasks

**If asked to implement data collection**:
- See `scripts/collect_trajectories_with_labels.py`
- Follow Phase 2 plan specifications
- Ensure correctness checking is robust

**If asked to implement analysis**:
- See `scripts/run_analysis.py` for geometric measures
- Follow Phase 3 plan for H2 transfer testing
- Implement confound controls from RESEARCH_PLAN.md

**If asked to implement steering**:
- See Phase 4 plan for steering methods
- Start with subspace projection (Method 1)
- Validate on training set before held-out evaluation

**If asked about literature**:
- Check LITERATURE_REVIEW_SHORT.md first
- Use LITERATURE_REVIEW_LONG.md for detailed analysis
- All citations are real (not fabricated)

---

## ⚠️ Common Mistakes to Avoid

1. **Confusing projects**: Don't mix up the main project (correct vs incorrect geometry) with background docs (RLVR vs SFT)
2. **Forgetting controls**: Always implement difficulty stratification, baseline comparisons
3. **Overclaiming**: Be precise about what's shown (correlation vs causation)
4. **Ignoring confounds**: Problem difficulty, output length, format are real issues
5. **Not updating this file**: Update claude.md when you create new files!

---

## 📝 Quick Reference

**Research question**: Correct vs incorrect reasoning geometry  
**Critical test**: H2 (cross-domain transfer)  
**Causal test**: H4 (steering)  
**Main confound**: Problem difficulty  
**Sample size**: 500 per task  
**Models**: Base, SFT, RL-Zero, Think  
**Tasks**: GSM8K (math), HumanEval (code), LogiQA (logic)
