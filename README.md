# Invalid Response Detection for Creativity Assessments

Automated detection of invalid responses in open-ended creativity tasks using LLM judges and reward models.

## Overview

Open-ended creativity assessments—Alternative Uses Task (AUT), Design Problems Task (DPT), Metaphor Completion—generate free-text responses that inevitably include invalid submissions: gibberish, refusals, off-topic answers, and incomplete text. These must be filtered before analysis.

This repository provides scripts for automated invalid detection using Gemini LLM judges, NVIDIA reward models, and GPT-2 perplexity, achieving **0.85–0.91 AUC** on individual scorers and **0.89–0.93 AUC** with ensemble scoring.

### What Counts as Invalid?

| Type | Example | Description |
|------|---------|-------------|
| Refusal | "idk", "n/a", "I'm giving up" | No genuine attempt |
| Gibberish | "asdfkj", "xxx", "wei" | Random characters or words |
| Off-topic | Response to a different prompt | Coherent but mismatched |
| Truncated | "a ca...", "powerfu..." | Incomplete, cut-off response |
| Misspelled | "moru bussAs hRading" | Corrupted/garbled text |

## Method

### Scoring Approaches

| Scorer | What It Does |
|--------|--------------|
| **Gemini 2.0 Flash** | Task-specific validity rating (1–5 scale) |
| **Nemotron 70B** | Reward model helpfulness score |
| **GPT-2 Perplexity** | Local perplexity scoring |
| **Ensemble** | Normalized average of Gemini + Nemotron |

### Why a 1–5 Scale?

Binary classification (valid/invalid) performed poorly on nuanced cases. A graded scale lets the model reason about borderline responses:

- **1–2**: Invalid (flag for exclusion)
- **3**: Borderline (weak but genuine attempt)
- **4–5**: Valid (clear, task-appropriate)

### Task-Specific Prompts

Generic "is this a valid response?" prompts fail because validity criteria differ by task:

**AUT (Alternative Uses Task)**
> Rate whether the response proposes a use for the given object. Invalid if: gibberish, blank, "idk", just describes the object (not a USE), or completely unrelated.

**DPT (Design Problems Task)**
> Rate whether the response proposes a solution to the problem. Invalid if: gibberish, blank, "idk", just restates the problem, or completely unrelated.

**META (Metaphor Completion)**
> Rate whether the response completes the sentence with a metaphor. Invalid if: gibberish, blank, "idk", repeats the prompt, or completely unrelated. Purely literal completions score low (2) but not necessarily invalid.

## Results

Evaluated on balanced datasets with synthetic invalid responses (see Data section).

### Overall AUC by Task

| Task | Gemini | Nemotron | Perplexity | Ensemble | n |
|------|--------|----------|------------|----------|---|
| Alternative Uses (AUT) | 0.845 | 0.823 | 0.616 | **0.888** | 2000 |
| Design Problems (DPT) | 0.906 | 0.773 | 0.741 | **0.926** | 2000 |
| Metaphor (META) | 0.855 | 0.901 | 0.700 | **0.914** | 1560 |

*Ensemble = normalized average of Gemini + Nemotron. Perplexity not included in ensemble. META has lower n for perplexity (n=1144) because single-word responses lack sufficient context for conditional probability calculation.*

### Detection Rate by Invalid Type

Percentage of invalid responses correctly flagged by Gemini (threshold ≤2).

**AUT (Alternative Uses Task)**

| Invalid Type | Detection Rate | n |
|--------------|----------------|---|
| Refusal ("idk", "n/a") | 100% | 180 |
| Random words | 95% | 194 |
| Wrong task (DPT response) | 84% | 186 |
| Truncated | 83% | 167 |
| Misspelled | 81% | 186 |

**DPT (Design Problems Task)**

| Invalid Type | Detection Rate | n |
|--------------|----------------|---|
| Wrong task (AUT response) | 99% | 209 |
| Refusal ("idk", "n/a") | 98% | 195 |
| Random words | 88% | 190 |
| Truncated | 70% | 207 |
| Misspelled | 53% | 183 |

**META (Metaphor Completion)**

| Invalid Type | Detection Rate | n |
|--------------|----------------|---|
| Refusal ("idk", "n/a") | 92% | 82 |
| Truncated | 90% | 162 |
| Misspelled | 80% | 161 |
| Random words | 70% | 165 |
| Human-labeled invalid | 67% | 51 |
| Wrong task | 27% | 159 |

Wrong-task detection is low for META because coherent completions from other prompts can still read as valid metaphors.

### False Positive Rates

Percentage of valid responses incorrectly flagged as invalid (Gemini ≤2):

| Task | False Positive Rate | n | Common Causes |
|------|---------------------|---|---------------|
| AUT | 30% | 1000 | Single-word responses, vague uses |
| DPT | 13% | 1000 | Vague or low-effort solutions |
| META | 17% | 780 | Literal (non-metaphorical) completions |

Many "false positives" are legitimately low-quality responses that arguably should be filtered.

### Key Findings

1. **Task-specific prompts matter.** Custom prompts improve AUC by ~0.10 over generic quality assessment.

2. **Different scorers suit different tasks.** Gemini works best for divergent thinking tasks (AUT, DPT); the reward model works best for metaphor completion.

3. **Ensemble is most robust.** Combining Gemini and Nemotron achieves 0.89–0.93 AUC across all tasks.

4. **Perplexity has limited utility.** GPT-2 perplexity underperforms the LLM-based approaches (AUC 0.62–0.74). Invalid responses have high perplexity, but so do creative valid responses—these signals conflict. Perplexity may be better suited for detecting generic/clichéd responses in creativity scoring rather than invalid detection.

5. **Ground truth is noisy.** Many "false positives" are legitimately low-quality (e.g., literal completions). The model may be stricter than human labels.

## Installation

```bash
git clone <repo-url>
cd invalids
pip install -r requirements.txt
```

Required packages:
```
pandas numpy scipy scikit-learn tqdm
torch transformers  # for perplexity
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
| `perplexity` | GPT-2 perplexity |
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

## Acknowledgments

This work builds on the augmentation strategy from:

- Laverghetta Jr., A., Luchini, S. A., Pronchick, J., & Beaty, R. E. (in prep). *Automated Detection of Invalid Responses to Creativity Assessments.*

## License

MIT