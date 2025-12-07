# Invalid Response Detection for Creativity Assessments

Automated detection of invalid responses in creativity task data using LLM judges, reward models, and perplexity scoring.

## Background

Creativity assessments like the Alternate Uses Task (AUT), Design Problems Task (DPT), and Metaphor Completion collect open-ended human responses. These datasets inevitably contain **invalid responses** that should be excluded from analysis:

| Type | Example | Why Invalid |
|------|---------|-------------|
| Gibberish | "asdfkj", "xxx" | Random characters |
| Refusal | "idk", "I don't know", "n/a" | No attempt |
| Off-topic | "bricks are red" (for AUT) | Describes object, not a use |
| Truncated | "use it to" | Incomplete |
| Wrong task | Valid metaphor for wrong stem | Mismatched |

Manual review doesn't scale. This tool automates detection using a **Quality Trio** of complementary scoring methods.

## Method

### The Quality Trio

| Scorer | What It Measures | API | Best For |
|--------|------------------|-----|----------|
| **Gemini** | Task-appropriate quality (1-5) | Google | Overall validity judgment |
| **NVIDIA** | Response helpfulness | NVIDIA | Detecting empty/unhelpful |
| **Perplexity** | Text surprisingness | Local | Misspellings, nonsense |

### Why 1-5 Quality Scale?

Our first approach used binary valid/invalid classification. It failed badly on metaphor completion (AUC 0.56—barely better than random). The model couldn't distinguish "valid but literal" from "invalid gibberish."

Switching to a **1-5 quality scale** improved metaphor AUC from 0.56 to 0.89. The scale gives the model room to reason about borderline cases, and the distribution naturally separates:
- **1-2**: Invalid (gibberish, refusal, off-topic)
- **3**: Borderline (weak attempt)
- **4-5**: Valid (clear, appropriate response)

### Why Task-Specific Prompts?

Generic prompts like "Is this response valid?" fail because validity depends on the task:

| Task | Valid Response Should Be |
|------|-------------------------|
| AUT (Alternative Uses) | A *use* for the object, not a description |
| DPT (Design Problems) | A *solution* to the problem, not restating it |
| META (Metaphor) | A *figurative* completion, not literal |

Each task gets a custom prompt explaining what we're looking for and listing specific invalid patterns.

## Results

Performance on balanced evaluation sets (50% valid, 50% invalid, n=200):

| Task | Gemini | NVIDIA | Perplexity | Best |
|------|--------|--------|------------|------|
| AUT | 0.82 | 0.82 | 0.66 | Tie |
| DPT | **0.92** | 0.75 | 0.75 | Gemini |
| META | 0.89 | **0.93** | 0.77 | NVIDIA |

**AUC interpretation**: 0.92 means if you randomly pick one valid and one invalid response, the scorer correctly ranks the valid one higher 92% of the time.

## Data

### Datasets

Each task directory contains:

| File | Contents | Invalid Rate |
|------|----------|--------------|
| `train_stratified.csv` | Human-labeled only | ~5% natural |
| `train_stratified_augmented.csv` | Human + synthetic invalids | 50% balanced |

The **augmented** dataset combines:
- **Human-labeled ground truth**: Original responses with human validity labels
- **Synthetic invalids**: Rule-based augmentations applied to valid responses

### Augmentation Types

| Type | Transformation | Example |
|------|---------------|---------|
| `dont_know` | Replace with refusal | "idk", "n/a", "I don't know" |
| `misspell` | Character corruption | "doorstop" → "dorsto" |
| `strip` | Truncate | "use as doorstop" → "use as" |
| `nonsensical` | Random word substitution | "doorstop" → "neptunium wall" |
| `wrong_task` | Swap with different prompt's response | Valid but mismatched |

The `source` column in output files tracks which augmentation was applied (empty = original human data).

## Usage

### Setup

```bash
cd quality/invalids
pip install -r requirements.txt
```

Create `.env` with API keys:
```
GEMINI_API_KEY=your-key
NVIDIA_API_KEY=nvapi-xxx
```

### Running

```bash
# Test run (200 samples per task, ~5 min)
python score_invalid_v4.py --task all --limit 200

# Full dataset (all data, ~1-2 hours depending on size)
python score_invalid_v4.py --task all --limit 0

# Single task
python score_invalid_v4.py --task meta

# Gemini only (fastest)
python score_invalid_v4.py --task all --skip-nvidia --skip-ppl

# Original (non-augmented) data
python score_invalid_v4.py --task aut --original
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--task {aut,dpt,meta,all}` | Which task(s) to score | `aut` |
| `--limit N` | Max rows per task (0 = all) | 200 |
| `--original` | Use original data instead of augmented | augmented |
| `--skip-gemini` | Skip Gemini scorer | run |
| `--skip-nvidia` | Skip NVIDIA scorer | run |
| `--skip-ppl` | Skip perplexity scorer | run |
| `--workers N` | Parallel API workers | 10 |
| `--rps N` | API rate limit (requests/sec) | 2.0 |

## Output

### Files Generated

**Per-task scored data:**
```
aut/aut_scored_augmented_v4.csv
dpt/dpt_scored_augmented_v4.csv
meta/meta_scored_augmented_v4.csv
```

**Cross-task summary:**
```
results_summary_augmented_v4.csv
```

### Output Columns

| Column | Type | Description |
|--------|------|-------------|
| `gemini_quality` | int 1-5 | Quality rating (higher = more valid) |
| `gemini_rationale` | str | Model's explanation |
| `nvidia_reward` | float | Helpfulness score (higher = more valid) |
| `perplexity` | float | GPT-2 surprisingness (higher = more likely invalid) |
| `source` | str | Augmentation type or "original" |

### Console Output

The script reports AUC, Spearman correlation, and F1 for each scorer:

```
Gemini Quality (1-5):
  AUC: 0.920 | Spearman: -0.755 | F1: 0.864
  n=200 (valid=100, invalid=100)
  Mean quality by label:
    Valid (label=0): 3.30
    Invalid (label=1): 1.53
```

## Directory Structure

```
invalids/
├── score_invalid_v4.py           # Main script (use this)
├── score_invalid.py              # v1: binary (deprecated)
├── score_invalid_v2.py           # v2: task-specific binary (deprecated)
├── score_invalid_v3.py           # v3: JSON schema (deprecated)
├── .env                          # API keys (not in git)
├── requirements.txt
├── README.md
├── results_summary_augmented_v4.csv
│
├── aut/
│   ├── train_stratified.csv           # Original human data
│   ├── train_stratified_augmented.csv # Human + synthetic
│   └── aut_scored_augmented_v4.csv    # Output
│
├── dpt/
│   └── ...
│
└── meta/
    ├── train_stratified_augmented.csv
    ├── Metaphor_Rated_Data_Cleaned.csv  # Raw source
    ├── generate_data.py                  # Data prep script
    └── README.md
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" | Multiple env vars | Harmless warning |
| Perplexity n < total | BPE tokenization | Normal for short responses |
| NVIDIA 429 errors | Rate limit | Reduce `--workers` or `--rps` |
| `gemini_quality` is None | API error | Check key/quota |

## Version History

| Version | Change | META AUC |
|---------|--------|----------|
| v1 | Binary valid/invalid, generic prompt | 0.56 |
| v2 | Task-specific binary prompts | 0.56 |
| v3 | Native JSON schema with CoT | 0.58 |
| **v4** | **1-5 quality scale, task-specific** | **0.89** |

## Intended Use

This tool is designed as a **data filter**, not a benchmark metric. The goal is to automatically flag invalid responses for exclusion before computing creativity scores (originality, fluency, etc.)—not to score creativity itself.

**Workflow:**
1. Run scorer on raw response data
2. Flag responses below threshold (e.g., `gemini_quality <= 2`)
3. Exclude flagged responses from downstream analysis
4. Optionally: manual review of borderline cases (`gemini_quality == 2-3`)

## Next Steps

- **Bias analysis**: Check whether detection rates differ systematically between human vs. AI-generated responses. If the filter disproportionately flags one group, it could skew benchmark comparisons.
- **Extend to other tasks**: Apply to Scientific Creativity Task and Creative Writing (short story) once data is ready.
- **Threshold calibration**: Current results use AUC (threshold-free). Production use needs calibrated thresholds per task.
- **Ensemble scoring**: Combine Gemini + NVIDIA + Perplexity into a single score for improved coverage and robustness.

## References

- Laverghetta et al., *Automated Detection of Invalid Responses in Open-Ended Creativity Assessments*
- Boussioux et al., *The Artificial Hivemind* (reward model + perplexity methodology)
