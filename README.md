# Invalid Response Detection for Creativity Assessments

Automated detection of invalid responses in open-ended creativity tasks using LLM judges and reward models.

## Overview

Open-ended creativity assessments—Alternative Uses Task (AUT), Design Problems Task (DPT), Metaphor Completion—generate free-text responses that inevitably include invalid submissions: gibberish, refusals, off-topic answers, and incomplete text. These must be filtered before analysis.

This repository provides scripts for automated invalid detection using Gemini LLM judges and NVIDIA reward models, achieving **0.85–0.91 AUC** on individual scorers and **0.89–0.93 AUC** with ensemble scoring.

### What Counts as Invalid?

| Type | Example | Description |
|------|---------|-------------|
| Refusal | "idk", "n/a", "I don't know" | No genuine attempt |
| Gibberish | "asdfkj", "xxx" | Random characters |
| Off-topic | "bricks are red" (for AUT) | Describes rather than uses the object |
| Truncated | "use it to" | Incomplete response |
| Wrong task | Valid response to a different prompt | Coherent but mismatched |
| Primary use | "reading" for book (AUT) | Not an *alternative* use |

## Method

### Scoring Approaches

| Scorer | What It Does | Best For |
|--------|--------------|----------|
| **Gemini 2.0 Flash** | Task-specific validity rating (1–5 scale) | AUT, DPT |
| **Nemotron 70B** | Reward model helpfulness score | Metaphor completion |
| **Ensemble** | Normalized average of Gemini + Nemotron | All tasks |

### Why a 1–5 Scale?

Binary classification (valid/invalid) performed poorly on nuanced cases. A graded scale lets the model reason about borderline responses:

- **1–2**: Invalid (flag for exclusion)
- **3**: Borderline (weak but genuine attempt)
- **4–5**: Valid (clear, task-appropriate)

### Why Task-Specific Prompts?

Generic "is this a valid response?" prompts fail because validity criteria differ by task. An AUT response must propose an *alternative* use; a metaphor response must be *figurative*, not literal. Each task gets a custom prompt specifying expected format and common failure modes.

## Results

Evaluated on balanced datasets with synthetic invalid responses (see Data section).

| Task | Gemini | Nemotron | Ensemble | n |
|------|--------|----------|----------|---|
| Alternative Uses (AUT) | 0.845 | 0.823 | **0.888** | 2000 |
| Design Problems (DPT) | 0.906 | 0.773 | **0.926** | 2000 |
| Metaphor (META) | 0.855 | 0.901 | **0.914** | 1560 |

### Key Findings

1. **Task-specific prompts matter.** Custom prompts improve AUC by ~0.10 over generic quality assessment.

2. **Different scorers suit different tasks.** Gemini works best for divergent thinking tasks (AUT, DPT); the reward model works best for metaphor completion.

3. **Ensemble is most robust.** Combining both scorers achieves 0.89–0.93 AUC across all tasks.

4. **Perplexity doesn't help here.** GPT-2 perplexity was tested but underperforms (AUC 0.62–0.74). Invalid responses have high perplexity, but so do creative valid responses. We dropped it from the final pipeline.

5. **Ground truth is noisy.** Many "false positives" are legitimately low-quality (e.g., primary uses, literal completions). The model may be stricter than human labels.

## Installation

```bash
git clone <repo-url>
cd invalids
pip install -r requirements.txt
```

Required packages:
```
pandas numpy scipy scikit-learn tqdm
torch transformers  # for perplexity (optional)
google-genai        # for Gemini
openai              # for NVIDIA API
python-dotenv
```

Create `.env` with API keys:
```
GEMINI_API_KEY=your-key
NVIDIA_API_KEY=nvapi-your-key
```

## Usage

```bash
# Quick test (200 samples per task)
python score_invalid_v5.py --task all --limit 200

# Full run
python score_invalid_v5.py --task all --limit 0

# Single task, Gemini only
python score_invalid_v5.py --task aut --skip-nvidia --skip-ppl
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--task {aut,dpt,meta,all}` | Which task(s) | `aut` |
| `--limit N` | Max rows (0 = all) | 200 |
| `--original` | Use original data (not augmented) | augmented |
| `--skip-gemini` | Disable Gemini | enabled |
| `--skip-nvidia` | Disable Nemotron | enabled |
| `--skip-ppl` | Disable perplexity | enabled |
| `--workers N` | Parallel workers | 10 |
| `--rps N` | Rate limit | 2.0 |

## Output

Scored files per task:
```
aut/aut_scored_augmented_v5.csv
dpt/dpt_scored_augmented_v5.csv
meta/meta_scored_augmented_v5.csv
```

Summary across tasks:
```
results_summary_augmented_v5.csv
```

### Output Columns

| Column | Description |
|--------|-------------|
| `gemini_quality` | Validity rating (1–5) |
| `gemini_rationale` | Model's explanation |
| `nvidia_reward` | Reward model score |
| `perplexity` | GPT-2 perplexity (if enabled) |
| `source` | "original" or augmentation type |
| `label` | Ground truth (0=valid, 1=invalid) |

## Data

### Input Format

Each task directory expects:

| File | Description |
|------|-------------|
| `train_stratified.csv` | Original responses with `prompt`, `response`, `label` columns |
| `train_stratified_augmented.csv` | Balanced version with synthetic invalids |

### Synthetic Invalid Generation

Valid responses were augmented to create balanced evaluation data:

| Type | Method |
|------|--------|
| `dont_know` | Replace with "idk", "n/a", etc. |
| `misspell` | Character-level corruption |
| `strip` | Truncate mid-sentence |
| `random_multi_word` | Random word combinations |
| `wrong_task` | Swap response from different prompt |

## Recommended Workflow

1. Run scorer: `python score_invalid_v5.py --task all --limit 0`
2. Filter: exclude responses with `gemini_quality ≤ 2`
3. Optionally review borderline cases (`gemini_quality = 3`)
4. Proceed with creativity scoring on filtered data

For metaphor completion, consider using `nvidia_reward` or the ensemble instead of Gemini alone.

## Limitations

- **Evaluated on synthetic invalids.** Real-world invalid detection may differ from augmented test sets.
- **No out-of-item evaluation.** We haven't tested generalization to completely new prompts.
- **API costs.** Scoring large datasets requires Gemini and NVIDIA API credits.
- **English only.** Prompts and scoring are designed for English responses.

## Acknowledgments

This work builds on approaches from:

- Laverghetta Jr., A., Luchini, S. A., Pronchick, J., & Beaty, R. E. (in prep). *Automated Detection of Invalid Responses to Creativity Assessments.* — finetuned TLM approach and augmentation strategy

## License

MIT
