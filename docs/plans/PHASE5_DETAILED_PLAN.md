# Phase 5: Write-Up and Publication

**Status**: ⏳ Pending (depends on Phase 4 completion)  
**Duration**: Weeks 11-12 (2 weeks)  
**Objective**: Document results, write paper, prepare for submission

---

## Overview

Phase 5 synthesizes all findings from Phases 1-4 into a coherent narrative, prepares publication materials, and releases code/data.

**Deliverables**:
1. Research paper (conference submission)
2. Code release (GitHub)
3. Data release (HuggingFace or Zenodo)
4. Blog post / Twitter thread (dissemination)

---

## Week 11: Paper Writing

### Day 1-2: Results Synthesis

**Consolidate findings**:
- Phase 1: Static geometry differences (RLVR vs SFT)
- Phase 2: H1 results (within-domain classification)
- Phase 3: H2 results (cross-domain transfer)
- Phase 4: H4 results (steering intervention)

**Create master results table**:
- All accuracies with confidence intervals
- Statistical tests (p-values, effect sizes)
- Comparison to baselines

### Day 3-4: Draft Paper Sections

**Abstract** (250 words):
- Research question
- Key findings (H1/H2/H4 outcomes)
- Main contribution
- Implications

**Introduction** (2 pages):
- Motivation: Why geometric analysis of reasoning?
- Research gap: Prior work limitations
- Our approach: Trajectory-based analysis
- Contributions: Novel findings

**Related Work** (1.5 pages):
- Probing and interpretability
- Truth representations (Marks & Tegmark)
- Activation steering (Turner et al.)
- CoT faithfulness (Turpin et al.)

**Method** (3 pages):
- Data collection protocol
- Geometric features (path signatures, curvature)
- Classification pipeline
- Steering algorithm

**Results** (4 pages):
- H1: Within-domain classification
- H2: Cross-domain transfer (critical test)
- H4: Steering intervention
- Ablations and controls

**Discussion** (2 pages):
- Interpretation of findings
- Limitations (confounds, sample size, etc.)
- Decision-before-reasoning problem
- Future work

**Conclusion** (0.5 pages):
- Summary of contributions
- Broader implications

### Day 5-7: Create Figures and Tables

**Figure 1**: Overview schematic
- Trajectory collection → Signature computation → Classification/Steering

**Figure 2**: H1 results
- Bar chart: Accuracy per model/task
- Comparison to baselines

**Figure 3**: H2 transfer matrix
- Heatmap: Train domain × Test domain
- Highlight successful transfers

**Figure 4**: H4 steering results
- Accuracy improvement bar chart
- Layer-wise effectiveness

**Figure 5**: Ablations
- Steering strength curves
- Manifold comparison (correct vs random vs incorrect)

**Tables**:
- Table 1: Model and task specifications
- Table 2: H1 classification results
- Table 3: H2 transfer matrix (numerical)
- Table 4: H4 steering results with statistics

---

## Week 12: Finalization and Submission

### Day 1-2: Internal Review

**Self-review checklist**:
- [ ] All claims supported by results
- [ ] No overclaiming (especially if H2/H4 failed)
- [ ] Limitations clearly stated
- [ ] Figures are clear and informative
- [ ] Tables are formatted correctly
- [ ] References are complete
- [ ] Code/data availability statement

**Revise based on self-review**

### Day 3-4: Code and Data Release Preparation

**Code release** (GitHub):
- Clean up code, remove debugging statements
- Add README with:
  - Installation instructions
  - Usage examples
  - Citation information
- Add LICENSE (MIT or Apache 2.0)
- Create requirements.txt / environment.yml
- Tag release version (v1.0.0)

**Data release** (HuggingFace or Zenodo):
- Package trajectory data (if shareable)
- Include metadata (model, task, correctness labels)
- Add data card with:
  - Description
  - Collection methodology
  - Intended use
  - Limitations
- Get DOI (via Zenodo)

### Day 5: Supplementary Materials

**Appendix**:
- Extended results tables
- Additional ablations
- Hyperparameter search details
- Failure case analysis

**Supplementary code**:
- Jupyter notebooks with analysis
- Visualization scripts
- Reproducibility instructions

### Day 6-7: Final Submission

**Submission checklist**:
- [ ] Paper PDF (formatted for venue)
- [ ] Supplementary materials PDF
- [ ] Code release URL
- [ ] Data release URL (if applicable)
- [ ] Author information
- [ ] Conflict of interest statement
- [ ] Ethics statement

**Submit to target venue**

---

## Publication Strategy

### Venue Selection (Depends on Results)

**If H1 + H2 + H4 all succeed**:
- **Target**: NeurIPS, ICML, ICLR (main conference)
- **Backup**: ACL, EMNLP (main conference)
- **Rationale**: Novel finding + causal evidence + practical application

**If H1 + H2 succeed, H4 fails**:
- **Target**: NeurIPS, ICLR (main conference or workshop)
- **Backup**: ACL, EMNLP
- **Rationale**: Interesting correlation, negative causal result

**If H1 succeeds, H2 fails**:
- **Target**: ACL, EMNLP (main conference)
- **Backup**: NeurIPS workshop, ICLR workshop
- **Rationale**: Domain-specific findings, characterizes limitations

**If H1 fails**:
- **Target**: ICBINB workshop (I Can't Believe It's Not Better)
- **Backup**: Insights from Negative Results workshop
- **Rationale**: Well-characterized negative result

### Timeline to Submission

| Venue | Submission Deadline | Notification | Conference |
|-------|-------------------|--------------|------------|
| ICML 2026 | Feb 2026 | May 2026 | Jul 2026 |
| NeurIPS 2026 | May 2026 | Sep 2026 | Dec 2026 |
| ICLR 2027 | Oct 2026 | Jan 2027 | May 2027 |
| ACL 2026 | Feb 2026 | May 2026 | Aug 2026 |

**Recommendation**: Target ICML 2026 (Feb deadline) if results are ready by end of Jan 2026.

---

## Dissemination Plan

### Academic Dissemination

**Preprint**:
- Post to arXiv immediately after submission
- Tweet link with key findings
- Post on relevant subreddits (r/MachineLearning)

**Conference presentation** (if accepted):
- Prepare poster or slides
- Practice talk
- Prepare demo (if applicable)

### Broader Dissemination

**Blog post**:
- Write accessible summary for general audience
- Include visualizations
- Post on personal blog / Medium / Substack

**Twitter thread**:
- 10-15 tweets summarizing key findings
- Include figures
- Tag relevant researchers

**LinkedIn post**:
- Professional summary
- Implications for industry

---

## Reproducibility Package

### Code Release Contents

```
ManiVer/
├── README.md                    # Installation and usage
├── LICENSE                      # MIT or Apache 2.0
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── configs/
│   └── models.yaml             # Model configurations
├── src/
│   ├── activation_collector.py # Trajectory collection
│   ├── geometric_measures.py   # Feature computation
│   ├── trajectory_steering.py  # Steering implementation
│   └── task_data.py            # Dataset loading
├── scripts/
│   ├── collect_trajectories_with_labels.py
│   ├── test_h1.py              # Within-domain classification
│   ├── test_h2_transfer.py     # Cross-domain transfer
│   └── test_h4_steering.py     # Steering evaluation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_h1_analysis.ipynb
│   ├── 03_h2_transfer.ipynb
│   └── 04_h4_steering.ipynb
└── paper/
    ├── main.pdf                # Paper
    └── supplementary.pdf       # Appendix
```

### Data Release Contents

```
maniver-data/
├── README.md                   # Data card
├── trajectories/
│   ├── olmo3_base/
│   │   ├── gsm8k_trajectories.h5
│   │   ├── humaneval_trajectories.h5
│   │   └── logiqa_trajectories.h5
│   └── [similar for other models]
├── signatures/
│   └── [path signature files]
└── metadata/
    ├── collection_details.json
    └── model_info.json
```

---

## Writing Guidelines

### Tone and Style

- **Be precise**: Use exact numbers, confidence intervals, p-values
- **Be honest**: Clearly state limitations and negative results
- **Be balanced**: Present both supporting and contradicting evidence
- **Be clear**: Avoid jargon where possible, define technical terms

### Common Pitfalls to Avoid

**Overclaiming**:
- ❌ "We solved reasoning interpretability"
- ✅ "We show that trajectory geometry correlates with correctness on verifiable tasks"

**Ignoring limitations**:
- ❌ Skip discussion of confounds
- ✅ "Our results may be confounded by problem difficulty, which we partially address via stratification"

**Cherry-picking**:
- ❌ Only report best results
- ✅ Report all results, including negative findings

**Unclear causality**:
- ❌ "Geometry causes correct reasoning" (unless H4 succeeds)
- ✅ "Geometry correlates with correct reasoning" (if only H1 succeeds)

---

## Contingency Plans

### If Results Are Borderline

**Problem**: H2 transfer is 54% (barely above chance).

**Solution**:
- Report as "weak transfer"
- Emphasize effect size over p-value
- Discuss why transfer is difficult
- Frame as "first step toward understanding transfer"

### If Reviewers Request More Experiments

**Common requests**:
- More models (Llama, GPT)
- More tasks (MMLU, ARC)
- More baselines (semantic entropy)

**Response strategy**:
- Prioritize requests that strengthen main claims
- Defer others to future work
- Provide preliminary results if possible

### If Paper Is Rejected

**Next steps**:
1. Read reviews carefully
2. Identify valid criticisms
3. Run additional experiments if needed
4. Revise paper
5. Resubmit to next venue

**Timeline**: Add 3-4 months for revision and resubmission

---

## Success Criteria

**Minimum viable**:
- Paper drafted and submitted
- Code released on GitHub
- Results documented

**Target**:
- Paper accepted at top-tier venue
- Code and data released
- Positive reception in community

**Stretch**:
- Paper accepted at NeurIPS/ICML/ICLR main conference
- High citation count (>50 in first year)
- Follow-up work by other researchers

---

## Post-Publication

### Maintenance

**Code maintenance**:
- Respond to GitHub issues
- Fix bugs as reported
- Update dependencies

**Data maintenance**:
- Ensure data remains accessible
- Update documentation if needed

### Follow-Up Work

**Potential extensions**:
1. Test on more models (Llama 3, GPT-4, Claude)
2. Test on more tasks (MMLU, ARC, HellaSwag)
3. Improve steering methods (learned maps, RL-based)
4. Apply to non-verifiable domains (H3)
5. Investigate decision-before-reasoning more deeply

**Collaboration opportunities**:
- Other researchers may want to build on this work
- Be open to collaborations
- Share insights and lessons learned

---

## Estimated Effort

| Activity | Days | Notes |
|----------|------|-------|
| Results synthesis | 2 | Consolidate all findings |
| Paper writing | 5 | Draft all sections |
| Figures/tables | 2 | Create visualizations |
| Code cleanup | 2 | Prepare for release |
| Data packaging | 1 | Prepare for release |
| Supplementary materials | 1 | Appendix, notebooks |
| Final review | 1 | Self-review and polish |
| **Total** | **14** | **2 weeks** |

---

## Deliverables Checklist

### Paper
- [ ] Abstract
- [ ] Introduction
- [ ] Related Work
- [ ] Method
- [ ] Results
- [ ] Discussion
- [ ] Conclusion
- [ ] References
- [ ] Figures (5)
- [ ] Tables (4)
- [ ] Supplementary materials

### Code
- [ ] GitHub repository created
- [ ] README with instructions
- [ ] LICENSE file
- [ ] requirements.txt / environment.yml
- [ ] All scripts cleaned and documented
- [ ] Example notebooks
- [ ] Release tagged (v1.0.0)

### Data
- [ ] Data packaged (HDF5 or similar)
- [ ] Data card / README
- [ ] Uploaded to HuggingFace or Zenodo
- [ ] DOI obtained
- [ ] License specified

### Dissemination
- [ ] arXiv preprint posted
- [ ] Twitter thread
- [ ] Blog post
- [ ] LinkedIn post

---

## Final Thoughts

Phase 5 is about **communicating** the research effectively. The quality of the write-up is as important as the quality of the research itself.

**Key principles**:
1. **Clarity**: Make it easy for readers to understand
2. **Honesty**: Report limitations and negative results
3. **Reproducibility**: Provide all necessary details
4. **Impact**: Frame contributions clearly

**Remember**: Even negative results are valuable if well-characterized and honestly reported. The goal is to advance scientific understanding, not just to publish positive results.
