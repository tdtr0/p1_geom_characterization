# Aha Moment Experiment: Error Detection & Phase Transitions

**Status**: Experiment A ✅ | Experiment A' ✅ (Flat Profile!) | Experiment B ⚠️ | Experiment C ✅
**Objective**: Test whether OLMo 3 models have error-detection features (Wynroe-style) and whether training paradigm affects active error correction.

---

## Results Summary (2026-01-19)

### Experiment A: Error Detection Probing (Original)
| Model | Pairs | Best Layer | Effect Size (d) | p-value |
|-------|-------|------------|-----------------|---------|
| **rl_zero** | 92 | Layer 14 | **1.70** | < 10⁻¹⁵ |
| **think** | 92 | Layer 14 | **1.65** | < 10⁻¹⁵ |

**Issue**: This was probing (correlational), not patching (causal). Signal present from layer 0 (d=1.1).

### Experiment A': Proper Wynroe Activation Patching ✅ COMPLETE

| Aspect | Wynroe et al. (DeepSeek-R1) | Our Replication (OLMo-3-Think) |
|--------|----------------------------|-------------------------------|
| **Method** | Activation patching | Activation patching |
| **Dataset** | MATH | GSM8K (harder subset) |
| **Metric** | Logit-diff recovery % | Logit-diff recovery % |
| **Finding** | **Layer 20 spike** (~70% recovery) | **FLAT profile** (~98% at ALL layers) |

**Layer Profile (think model, N=50):**
```
Layer  0:  97.9% ± 14.4  ███████████████████
Layer 10:  97.9% ± 14.9  ███████████████████
Layer 20:  98.0% ± 13.9  ███████████████████
Layer 28:  98.1% ± 13.6  ███████████████████  ← "best" (within noise)
Layer 30:  98.0% ± 13.8  ███████████████████
```

**Result**: **OPPOSITE of Wynroe's finding!** Patching ANY layer gives ~98% recovery. No critical layer.

**Interpretation**:
1. **Task may still be too easy**: Harder GSM8K still not challenging enough
2. **OLMo has distributed error detection**: Unlike DeepSeek-R1's localized circuit at layer 20
3. **Methodological issue**: Simple arithmetic error injection may be trivially detectable

**Conclusion**: OLMo-3 does NOT have the same localized error-detection circuit that Wynroe found in DeepSeek-R1.

### Experiment B: Natural Pivot Detection ⚠️ TRIVIAL
**Issue**: Lower curvature at pivot tokens is likely just surface-level pattern:
- "Wait..." and "But..." are transition tokens → naturally smoother dynamics
- No comparison across models (only tested `olmo3_think`)
- Measures where pivots OCCUR, not whether model CAN correct

**Verdict**: Experiment B doesn't isolate anything interesting. Replaced by Experiment C.

### Experiment C: Active Error Correction ✅ COMPLETE

| Model | Correction Rate | Explicit Corrections | Error Propagation |
|-------|-----------------|---------------------|-------------------|
| **base** | 8.9% | 0.0% | **77.8%** |
| **rl_zero** | 11.1% | 2.2% | **75.6%** |
| **think** | 11.1% | **26.7%** | **24.4%** |

**Finding**: Think models show 3× lower error propagation and 13× more explicit correction attempts, but **same final accuracy** (11%). Detection ≠ correction.

See [RESULTS.md](RESULTS.md) for full analysis.

---

## Quick Start

### Current GPU Server (2026-01-17)
```bash
# SSH to vast.ai instance (RTX 5060 Ti 16GB, ~7-8 hours available)
ssh -p 47319 root@171.248.243.88 -L 8080:localhost:8080
```

### Run Experiment B (CPU-only, local)
```bash
cd /Users/thanhdo/CascadeProjects/ManiVer/main
bash experiments/aha_moment/run_experiment_b.sh
```

### Run Experiment A (GPU required)
```bash
# On GPU server:
cd /workspace
git clone https://github.com/tdtr0/p1_geom_characterization.git maniver
cd maniver
pip install torch transformers datasets h5py scipy matplotlib tqdm

# Run collection (3-6 hours)
python experiments/aha_moment/collect_clean_corrupted_pairs.py \
    --n_problems 100 \
    --models rl_zero,think \
    --output experiments/aha_moment/data/wynroe_replication/

# Analyze results
python experiments/aha_moment/analyze_wynroe_direction.py \
    --input experiments/aha_moment/data/wynroe_replication/wynroe_trajectories.h5 \
    --output experiments/aha_moment/results/wynroe/
```

---

## What We Have

### Phase 2 Trajectories (B2 Storage)
- 11/12 HDF5 files with full trajectories + correctness labels
- 4 models × 3 tasks (missing: olmo3_base/gsm8k)
- Shape: (500 samples, 512 tokens, 16 layers, 4096 dims)
- **Location**: `b2://ml-activations-store/trajectories/`

### What Phase 2 Data CAN Do
- Average correctness direction (difference-in-means on correct vs incorrect)
- Test direction transfer across domains
- Model comparison (Base vs SFT vs RL-Zero vs Think)

### What Phase 2 Data CANNOT Do
- Detect phase transitions (trajectories are completely different problems)
- Token-level error detection (no alignment between correct/incorrect)

### What Phase 2 Data DOES Have (for Experiment B)
- **`model_outputs` dataset**: Full generated text stored in HDF5 files
- Can parse for pivot patterns ("but", "wait", "actually")
- Need to re-tokenize to align text positions → trajectory positions

---

## What We Need

### For Experiment A: Wynroe Replication (Clean/Corrupted Pairs)
```
1. Generate CoT from RL-Zero/Think on GSM8K (N=200 problems)
2. Parse <<expr=result>> annotations to find numbers
3. Create corrupted versions (change ONE number per trace)
4. Run BOTH clean and corrupted through ALL 4 models
5. Collect trajectories at error token position
6. Extract direction, test for phase transition
```

**GPU Time**: ~3-6 hours

### For Experiment B: Natural Pivot Detection
```
⚠️ CANNOT reuse Phase 2 data - it only has prompt trajectories!

NEW APPROACH:
1. Generate text from olmo3_think model on GSM8K (N=200)
2. Collect activations at EACH generation step (not just prompt)
3. Detect pivot positions in generated text
4. Analyze trajectory dynamics at pivot vs random tokens
```

**GPU Time**: ~2-3 hours (must collect generation trajectories)
**Script**: `collect_pivot_trajectories.py`

---

## Experiment A: Wynroe-Style Error Detection

### Background
Keith Wynroe (2025) found that DeepSeek-R1 has a linear "error-detection" direction that shows discontinuous activation spikes at tokens where the model writes incorrect content.

### Our Questions
1. Does this direction exist in OLMo 3 models?
2. Is it present in Base, or only after SFT/RLVR?
3. Does direction strength correlate with training paradigm?

### Data Flow
```
Step 1: Generate CoT (RL-Zero only - it produces structured traces)
┌─────────────────────────────────────────────────────────────────┐
│ Input: GSM8K problem "What is 5×3+2?"                           │
│ Output: "<think> 5×3=<<5*3=15>>15, 15+2=<<15+2=17>>17 </think>  │
│          #### 17"                                               │
└─────────────────────────────────────────────────────────────────┘

Step 2: Parse and Corrupt
┌─────────────────────────────────────────────────────────────────┐
│ Clean:   "5×3=<<5*3=15>>15, 15+2=<<15+2=17>>17"                 │
│ Corrupt: "5×3=<<5*3=15>>15, 15+2=<<15+2=18>>18"  # 17→18       │
│                                          ↑ error token          │
└─────────────────────────────────────────────────────────────────┘

Step 3: Run Through ALL Models
┌─────────────────────────────────────────────────────────────────┐
│                    Clean prefix    Corrupted prefix             │
│                    ───────────     ─────────────────            │
│ Base model         traj_clean      traj_corrupt                 │
│ SFT model          traj_clean      traj_corrupt                 │
│ RL-Zero model      traj_clean      traj_corrupt                 │
│ Think model        traj_clean      traj_corrupt                 │
└─────────────────────────────────────────────────────────────────┘

Step 4: Extract Direction and Analyze
┌─────────────────────────────────────────────────────────────────┐
│ direction = mean(corrupt_at_error) - mean(clean_at_same_pos)    │
│                                                                 │
│ For each model, measure:                                        │
│ - Direction magnitude (effect size d)                           │
│ - Phase transition sharpness (activation jump at error token)   │
│ - Layer profile (which layer shows strongest signal)            │
└─────────────────────────────────────────────────────────────────┘
```
┌───────────────────────────────────────────────────────── ────┐
│                   Research Strategy                          │
├──────────────────────────────────────────────────────── ─────┤
│                                                              │
│  OPEN MODELS (OLMo, Qwen, Llama)                             │
│  ├── Full activation analysis                                │
│  ├── Error-detection direction extraction                    │
│  ├── Phase transition measurement                            │
│  └── Causal intervention                                     │
│                                                              │
│            ↓ Transfer findings ↓                             │
│                                                              │
│  CLOSED MODELS (Claude, o1, GPT-4)                           │
│  ├── Behavioral probing (does behavior match?)               │
│  ├── Correction rate analysis                                │
│  ├── Self-consistency probes                                 │
│  ├── Extended thinking mining (Claude)                       │
│  └── Logprob analysis (when available)                       │
│                                                              │
│            ↓ Validate ↓                                      │
│                                                              │
│  QUESTION: Do closed models show the same behavioral         │
│  signatures that correlate with error-detection in open      │
│  models?                                                     │
│                                                              │
└────────────────────────────────────────────────────────────-─┘
### Expected Results

| Model | Error Direction | Hypothesis |
|-------|-----------------|------------|
| Base | Weak/None | No outcome-based training signal |
| SFT | Weak | Trained on format, not correctness |
| RL-Zero | Strong | RLVR provides outcome signal |
| Think | Strong | Full RLVR training |

### Implementation

**Script**: `collect_clean_corrupted_pairs.py` (to be created)

```python
# Pseudocode
def main():
    # 1. Load GSM8K
    problems = load_gsm8k(n=200)

    # 2. Generate CoT from RL-Zero
    cot_traces = []
    for problem in problems:
        output = generate(rl_zero_model, problem)
        cot_traces.append(parse_cot(output))

    # 3. Create clean/corrupted pairs
    pairs = []
    for trace in cot_traces:
        annotations = find_annotations(trace)  # <<expr=result>>
        if annotations:
            clean = trace
            corrupt = corrupt_last_number(trace, annotations[-1])
            error_token_idx = find_error_token(corrupt)
            pairs.append((clean, corrupt, error_token_idx))

    # 4. Collect trajectories from all models
    for model_name in ['base', 'sft', 'rl_zero', 'think']:
        model = load_model(model_name)
        for clean, corrupt, error_idx in pairs:
            traj_clean = collect_trajectory(model, clean)
            traj_corrupt = collect_trajectory(model, corrupt)
            save(model_name, traj_clean, traj_corrupt, error_idx)

    # 5. Analyze
    analyze_error_detection_direction()
```

---

## Experiment B: Natural Pivot Detection

### Background
Thinking models sometimes self-correct mid-generation with phrases like "BUT wait...", "actually...", "no, that's wrong...". If these are genuine corrections (not just stylistic), the trajectory should show phase transition signatures.

### Our Questions
1. Do pivot tokens show higher velocity than surrounding tokens?
2. Is there a direction change (trajectory bending) at pivots?
3. Does Lyapunov exponent spike at pivots (instability → new basin)?

### ⚠️ CRITICAL: Why Phase 2 Data Doesn't Work

Phase 2 data has a fundamental limitation:
```
Phase 2 HDF5 structure:
  - trajectories: (500, 512, 16, 4096)  ← BUT only first 66 positions are non-zero!
  - model_outputs: Full generation text (500+ tokens)
  - prompts: Input prompts (~66 tokens)

Problem:
  - Trajectories only contain PROMPT activations (positions 0-65)
  - Positions 66-511 are ZERO-PADDED (no generation data)
  - Pivots occur in GENERATION (positions 100+) → all zeros!
```

**This is why Experiment B failed initially** — we were analyzing zeros, not actual pivot dynamics.

### Data Flow (NEW: Collect Generation Trajectories)
```
Step 1: Generate with Activation Collection
┌─────────────────────────────────────────────────────────────────┐
│ For each GSM8K problem:                                         │
│   1. Encode prompt → input_ids                                  │
│   2. For each generation step:                                  │
│      a. Forward pass through model                              │
│      b. Extract hidden states at all layers                     │
│      c. Store: trajectory[sample, token_idx, layer, hidden]    │
│      d. Sample next token, append to input_ids                  │
│   3. Store generated text + full trajectory                     │
└─────────────────────────────────────────────────────────────────┘

Step 2: Detect Pivots in Generated Text
┌─────────────────────────────────────────────────────────────────┐
│ generated_text = "Let me calculate... BUT WAIT, I made an error"│
│                                        ↑                        │
│ Pivot at token position 47 (in generation, not prompt)         │
└─────────────────────────────────────────────────────────────────┘

Step 3: Analyze Trajectory at Pivot
┌─────────────────────────────────────────────────────────────────┐
│ trajectory[sample, pivot_idx, :, :] ← actual generation dynamics│
│ Compare velocity/direction to surrounding tokens                │
└─────────────────────────────────────────────────────────────────┘
```
Step 4: replicate in close models - 

these models can actually check if they are good or not - which might be useful for us - this is not interventional but just descriptional - we might need something else to be interventional. 
### Pivot Detection: Multi-Token Challenge

"But wait" might tokenize to multiple tokens. We need smarter detection:

#### Option A: Rule-Based (Simple)
```python
PIVOT_PATTERNS = [
    r'\bBUT\b', r'\bWait\b', r'\bwait\b',
    r'\bactually\b', r'\bActually\b',
    r'\bhowever\b', r'\bHowever\b',
    r'\blet me reconsider\b', r'\bon second thought\b',
    r'\bI was wrong\b', r'\bno,\s', r'\bhmm\b',
]
# Find char position in text, then align to first token of match
```
- Pro: Fast, no training
- Con: Misses semantic pivots ("this doesn't seem right...")

#### Option B: Zero-Shot Classifier (No Training)
```python
from transformers import pipeline

# Download once (~300MB), runs on CPU
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

def is_pivot(text_window: str) -> bool:
    """Check if window contains self-correction."""
    result = classifier(
        text_window,
        candidate_labels=["self-correction", "continuation"],
    )
    return result['labels'][0] == "self-correction"

# Usage: check 50-char windows around regex matches
for match in re.finditer(r'\b(but|wait|actually)\b', text, re.I):
    window = text[max(0, match.start()-25):match.end()+25]
    if is_pivot(window):
        pivots.append(match.start())
```
- Pro: No training, semantic understanding
- Con: Slower (~0.5s per window), but we only check regex matches

#### Option C: Free LLM API (OpenRouter)
```python
import openai

# Free models on OpenRouter
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],  # Free tier available
)

def detect_pivots_llm(text: str) -> List[int]:
    """Use free LLM to identify pivot positions."""
    response = client.chat.completions.create(
        model="meta-llama/llama-3.2-3b-instruct:free",  # Free!
        messages=[{
            "role": "user",
            "content": f"""Find self-correction points in this reasoning trace.
Return character positions where the model changes/corrects its thinking.

Text: "{text}"

Return JSON: {{"pivots": [pos1, pos2, ...]}}"""
        }]
    )
    return json.loads(response.choices[0].message.content)["pivots"]
```
- Pro: Best semantic understanding, free API
- Con: Rate limits, network dependency

#### Option D: Just Regex + Heuristics (Simplest)
```python
def detect_pivots_simple(text: str) -> List[int]:
    """Regex-only detection - surprisingly effective."""
    pivots = []
    # Strong pivot indicators (high precision)
    strong_patterns = [
        r'\bBUT WAIT\b',
        r'\bWait,\s*(no|that)',
        r'\bActually,\s*(no|that|I)',
        r'\bI was wrong\b',
        r'\blet me reconsider\b',
        r'\bno,\s*that\'s (not|wrong)',
    ]
    for pattern in strong_patterns:
        for match in re.finditer(pattern, text, re.I):
            pivots.append(match.start())
    return pivots
```
- Pro: Fast, no dependencies, no API
- Con: Misses subtle pivots

### Recommended Approach: Start Simple, Escalate If Needed

```
1. Start with Option D (regex) — fast, no setup
2. If too many false positives: add Option B (zero-shot classifier as filter)
3. If still noisy: use Option C (free LLM API) for final verification
```

```python
def detect_pivots(text: str, tokenizer, use_classifier=False) -> List[int]:
    """Detect pivot token positions."""

    # Stage 1: Regex candidates (always run)
    candidates = []
    patterns = [r'\bBUT\b', r'\bWait\b', r'\bactually\b', r'\bhowever\b']
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            candidates.append(match.start())

    # Stage 2: Zero-shot filter (optional, if too many false positives)
    if use_classifier:
        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli")
        filtered = []
        for pos in candidates:
            window = text[max(0, pos-30):pos+50]
            result = classifier(window, ["self-correction", "continuation"])
            if result['labels'][0] == "self-correction":
                filtered.append(pos)
        candidates = filtered

    # Stage 3: Align to token positions
    tokens = tokenizer(text, return_offsets_mapping=True)
    token_positions = []
    for char_pos in candidates:
        for idx, (start, end) in enumerate(tokens.offset_mapping):
            if start <= char_pos < end:
                token_positions.append(idx)
                break

    return token_positions
```

### Expected Results

| Metric | Pivot Token | Random Token | Hypothesis |
|--------|-------------|--------------|------------|
| Velocity | High | Low | Pivot = sudden direction change |
| Direction change | High | Low | Trajectory bending at pivot |
| Lyapunov | High (spike) | Stable | Transitioning between basins |

### Actual Results (Experiment B)

| Metric | Pivot Mean | Random Mean | Effect Size (d) | Finding |
|--------|------------|-------------|-----------------|---------|
| Velocity | 14.90 | 15.42 | **-0.22** | Pivots are SLOWER |
| Direction Change | 1.37 | 1.38 | -0.05 | No difference |
| Menger Curvature | 0.111 | 0.109 | 0.17 | No difference |
| Gaussian Proxy | 0.757 | 0.768 | **-0.31** | Pivots MORE LINEAR |

**Conclusion**: Results are opposite of hypothesis. Pivots are "pause tokens" with smoother dynamics, not sharp turns. This is likely trivial - transition words naturally have different dynamics than calculation tokens. **Experiment C provides a more meaningful test.**

### Critical Confound: Induction Heads (2026-01-19)

The lower velocity at pivots is likely a **trivial artifact of induction heads**, not evidence of error detection:

1. **Observation**: "Wait..." tokens are often followed by **repeats** of previous content
   - Example: "Wait, let me check: 2+2=4" → model copies earlier calculation

2. **Mechanism**: [Induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) detect [A][B]...[A] patterns and predict [B]
   - When pivots trigger re-checking, the model is often just **copying from context**
   - Copying = less novel computation = lower velocity

3. **The direction question**: What direction is the velocity pointing?
   - If it's induction (copying), activations move toward the copied representation
   - This is predictable and trivial - tells us nothing about error detection

4. **Proposed ablation** (future work):
   - Check induction circuit SAE features (Neuronpedia) at pivot vs random positions
   - If induction features are HIGH at pivots → velocity slowdown is trivial confound
   - If induction features are LOW → velocity slowdown might be meaningful

**Bottom line**: Experiment B results are likely just measuring induction head activity, not error detection capability.

---

## Experiment C: Active Error Correction (NEW)

### Motivation

Experiment A shows models CAN detect errors (passive signal exists).
Experiment B measures where pivot words occur (trivial finding).

**Missing question**: Do models ACTIVELY correct errors when given the chance?

### Hypothesis

**Think-trained models (rl_zero, think) should correct errors that base models propagate.**

If a model processes a corrupted solution with an arithmetic error:
- **Base model**: Continues with wrong answer (no correction ability)
- **Think model**: Recognizes error and corrects it (active correction)

### Experimental Design

```
Step 1: Take corrupted traces from Experiment A
┌─────────────────────────────────────────────────────────────────┐
│ Corrupted trace: "The cost is $430 + $320 = $751"              │
│                                               ↑ ERROR (should be $750)
│ Ground truth answer: 750                                        │
└─────────────────────────────────────────────────────────────────┘

Step 2: Truncate BEFORE the final answer
┌─────────────────────────────────────────────────────────────────┐
│ Prefix: "Question: ... The cost is $430 + $320 = $751          │
│          Therefore, the answer is "                             │
│                                    ↑ TRUNCATE HERE              │
│                                                                 │
│ The model must now complete: what number comes next?            │
└─────────────────────────────────────────────────────────────────┘

Step 3: Continue generation with different models
┌─────────────────────────────────────────────────────────────────┐
│ Model         │ Continuation           │ Correct? │ Behavior    │
│───────────────┼────────────────────────┼──────────┼─────────────│
│ base          │ "751"                  │ ❌       │ Propagates  │
│ sft           │ "751"                  │ ❌       │ Propagates  │
│ rl_zero       │ "Wait, 430+320=750..."│ ✅       │ Corrects!   │
│ think         │ "750 (fixing error)"   │ ✅       │ Corrects!   │
└─────────────────────────────────────────────────────────────────┘

Step 4: Collect activation trajectories during continuation
┌─────────────────────────────────────────────────────────────────┐
│ For models that CORRECT:                                        │
│   - What do activations look like at correction point?          │
│   - Higher curvature? Velocity spike? Direction change?         │
│                                                                 │
│ For models that PROPAGATE:                                      │
│   - Smooth continuation (no correction dynamics)                │
│   - Lower curvature, stable trajectory                          │
└─────────────────────────────────────────────────────────────────┘
```

### Metrics

**Primary (Functional)**:
| Metric | Description |
|--------|-------------|
| Correction Rate | % of errors corrected (final answer matches ground truth) |
| Correction Type | Explicit ("Wait, that's wrong") vs Implicit (just outputs correct number) |

**Secondary (Geometric)**:
| Metric | Description |
|--------|-------------|
| Trajectory Divergence | How much does correcting model diverge from propagating model? |
| Curvature at Correction | Menger curvature when correction happens |
| Error Direction Projection | Does error-detection direction predict correction? |

### Expected Results

| Model | Correction Rate | Hypothesis |
|-------|-----------------|------------|
| **base** | ~0% | No outcome training, just continues pattern |
| **sft** | ~10-20% | Some format understanding, mostly follows input |
| **rl_zero** | ~50-70% | RLVR teaches outcome verification |
| **think** | ~60-80% | Full training pipeline, best correction |

**Key comparison**: rl_zero vs base reveals what RLVR training adds.

### Implementation

**Script**: `run_error_correction.py`

```python
def main():
    # 1. Load corrupted traces from Experiment A
    with h5py.File('data/wynroe_replication/wynroe_trajectories.h5', 'r') as f:
        metadata = load_metadata(f)  # Has error positions, ground truth

    # 2. Create truncated prefixes
    prefixes = []
    for pair in metadata:
        corrupted_trace = pair['corrupted_trace']
        # Truncate before "#### <answer>" or "the answer is <number>"
        truncated = truncate_before_answer(corrupted_trace)
        prefixes.append({
            'prefix': truncated,
            'ground_truth': pair['answer'],
            'error_value': pair['corrupt_value'],  # The wrong number
        })

    # 3. Continue with each model
    models = ['base', 'sft', 'rl_zero', 'think']
    results = {}

    for model_name in models:
        model = load_model(model_name)

        for prefix_data in prefixes:
            # Generate continuation
            continuation = model.generate(prefix_data['prefix'], max_tokens=50)

            # Extract final answer from continuation
            predicted = extract_number(continuation)
            correct = (predicted == prefix_data['ground_truth'])

            # Collect trajectory during generation (if needed for geometry)
            trajectory = collect_generation_trajectory(model, prefix_data['prefix'])

            results[model_name].append({
                'correct': correct,
                'predicted': predicted,
                'continuation': continuation,
                'trajectory': trajectory,
            })

    # 4. Analyze
    for model_name in models:
        correction_rate = mean([r['correct'] for r in results[model_name]])
        print(f"{model_name}: {correction_rate:.1%} correction rate")
```

### Success Criteria

1. **Primary**: Think models (rl_zero, think) correct significantly more errors than base model (p < 0.05)
2. **Secondary**: Correction rate correlates with error-detection strength from Experiment A
3. **Geometric**: Correcting trajectories show distinct dynamics vs propagating trajectories

### GPU Requirements

- ~1-2 hours on single GPU (92 prefixes × 4 models × ~50 tokens each)
- Same GPU as Experiment A (eyecog or box1)

### Implementation

**Scripts**:
- `detect_pivots.py` — Pivot detection (regex + optional zero-shot filter) (DEPRECATED)
- `analyze_pivot_dynamics.py` — Analyze velocity/direction at pivots (DEPRECATED)

**Dependencies** (for zero-shot option):
```bash
pip install transformers  # for BART-large-mnli
# Model downloads automatically on first use (~300MB)
```

---

## Execution Plan

### Phase 1: Experiment A (Wynroe Replication) — 3-6 GPU hours

```bash
# Step 1: Create collection script
# (to be created: collect_clean_corrupted_pairs.py)

# Step 2: Run on vast.ai
python experiments/aha_moment/collect_clean_corrupted_pairs.py \
    --n_problems 200 \
    --models base,sft,rl_zero,think \
    --output experiments/aha_moment/data/wynroe_replication/

# Step 3: Analyze
python experiments/aha_moment/analyze_wynroe_direction.py \
    --input experiments/aha_moment/data/wynroe_replication/ \
    --output experiments/aha_moment/results/
```

### Phase 2: Experiment B (Natural Pivots) — 2-3 GPU hours

```bash
# Step 1: Collect generation trajectories (GPU required)
python experiments/aha_moment/collect_pivot_trajectories.py \
    --n_samples 200 \
    --model olmo3_think \
    --max_tokens 512 \
    --output experiments/aha_moment/data/pivot_collection/

# Step 2: Detect pivots and analyze (CPU only)
python experiments/aha_moment/analyze_pivot_trajectories.py \
    --input experiments/aha_moment/data/pivot_collection/pivot_trajectories.h5 \
    --output experiments/aha_moment/results/pivot_analysis/
```

**Note**: Cannot reuse Phase 2 data because it only contains prompt trajectories.

---

## Success Criteria

### Experiment A (Wynroe Replication)
- **Positive**: Error direction exists in RL-Zero/Think (p < 0.05, d > 0.5)
- **Training effect**: Direction stronger in RL-Zero than Base
- **Phase transition**: Sharp activation jump at error token (not gradual)

### Experiment B (Natural Pivots)
- **Positive**: Pivot tokens show significantly higher velocity/direction change (p < 0.05)
- **Consistent**: Effect holds across multiple pivot types (BUT, wait, actually)
- **Comparison**: Pivot effect stronger in Think model than Base

### What Would Failure Mean
- **Exp A fails**: Error detection may be R1-specific (distillation artifact?)
- **Exp B fails**: "Aha moments" are stylistic, not computational transitions
- Both still valuable negative results

---

## Files

| File | Purpose | Status |
|------|---------|--------|
| `collect_clean_corrupted_pairs.py` | Generate and collect Wynroe-style data | ✅ DONE |
| `analyze_wynroe_direction.py` | Extract direction, test phase transition | ✅ DONE |
| `collect_pivot_trajectories.py` | Collect generation trajectories for pivot analysis | ✅ DONE |
| `analyze_pivot_trajectories.py` | Detect pivots and analyze dynamics | ✅ DONE |
| `run_experiment_a.sh` | Runner script for Experiment A | ✅ DONE |
| `run_experiment_b.sh` | Runner script for Experiment B | ✅ DONE |
| `detect_pivots.py` | (Legacy) Detect pivots in Phase 2 - doesn't work | ⚠️ DEPRECATED |
| `analyze_phase2_pivots.py` | (Legacy) Phase 2 analysis - doesn't work | ⚠️ DEPRECATED |

---

## GPU Time Summary

| Experiment | Data Collection | Analysis | Total | Status |
|------------|-----------------|----------|-------|--------|
| A: Wynroe replication | ~20 min (GPU) | ~1 min (CPU) | **~20 min** | ✅ Complete |
| B: Natural pivots | 2-3 hrs (GPU) | ~5 min (CPU) | **2-3 hrs** | 🔄 Running |
| **Combined** | | | **~3 hrs GPU** | |

**Note**: Experiment B requires new data collection because Phase 2 only captured prompt trajectories.

---

## Connection to ManiVer

This experiment is **separate from Phase 3** but provides supporting evidence:

- If Exp A succeeds: Validates that error-detection is learnable (supports H1)
- If direction transfers across domains: Supports H2
- Results feed into Phase 3 analysis as additional features

**NOT** blocking Phase 3 — these experiments run in parallel.

---

## References

- **Wynroe (2025)**: "Finding an Error-Detection Feature in DeepSeek-R1" — LessWrong
- **Zhao et al. (2025)**: "Can Aha Moments Be Fake?" — TrueThinking scores
- **Ren & Liu (2026)**: HRM paper — attractor dynamics and grokking transitions
