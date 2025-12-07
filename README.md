# Invalid Response Detection for Creativity Assessments

Automated detection of invalid responses in open-ended creativity assessments using large language model judges, reward models, and perplexity scoring.

## Overview

Open-ended creativity assessments such as the Alternative Uses Task (AUT), Design Problems Task (DPT), and Metaphor Completion generate large volumes of free-text responses. These datasets inevitably contain invalid responses—gibberish, refusals, off-topic answers, and incomplete submissions—that must be excluded before analysis.

This tool provides an automated filtering system that achieves **0.89–0.93 AUC** across creativity tasks, enabling researchers to process large datasets without manual review.

### Types of Invalid Responses

| Type | Example | Description |
|------|---------|-------------|
| Gibberish | "asdfkj", "xxx" | Random or meaningless characters |
| Refusal | "I don't know", "n/a" | No genuine attempt at the task |
| Off-topic | "bricks are red" (for AUT) | Describes the object rather than providing a use |
| Truncated | "use it to" | Incomplete response |
| Wrong task | Valid response to different prompt | Coherent but mismatched to the task |

## Method

### Scoring Models

The system uses three complementary scoring approaches:

| Model | Description | Strengths |
|-------|-------------|-----------|
| **Gemini 2.0 Flash** | LLM judge with task-specific prompts rating validity on a 1–5 scale | Strong performance on divergent thinking tasks |
| **Llama 3.1 Nemotron 70B** | Reward model trained on human preferences | Excels at detecting incoherent or unhelpful responses |
| **GPT-2 Perplexity** | Local perplexity scoring (no API required) | Detects misspellings, gibberish, and nonsensical text |

### Design Decisions

**1–5 Quality Scale**: Binary classification (valid/invalid) performed poorly on nuanced cases like metaphor completion (AUC 0.56). A graded scale allows the model to reason about borderline cases:
- **1–2**: Invalid (gibberish, refusal, off-topic)
- **3**: Borderline (weak but genuine attempt)
- **4–5**: Valid (clear, task-appropriate response)

**Task-Specific Prompts**: Generic validity prompts fail because what constitutes a valid response differs by task. Each task type receives a custom prompt specifying the expected response format and common invalid patterns.

## Results

Evaluated on balanced datasets (50% valid, 50% invalid) with sample sizes of n=2000 for AUT and DPT, and n=1560 for META.

| Task | Gemini | Nemotron | Perplexity | Ensemble |
|------|--------|----------|------------|----------|
| Alternative Uses (AUT) | 0.845 | 0.823 | 0.662 | **0.888** |
| Design Problems (DPT) | 0.906 | 0.773 | 0.745 | **0.926** |
| Metaphor Completion (META) | 0.855 | 0.901 | 0.765 | **0.914** |

*Ensemble: normalized average of Gemini and Nemotron scores. Perplexity not included in ensemble (see Key Findings).*

### Key Findings

1. **Task-specific prompts are essential.** Custom prompts improve AUC by 0.10–0.15 compared to generic quality assessment.

2. **Different models excel at different tasks.** Gemini performs best on divergent thinking tasks (AUT, DPT), while the reward model excels at metaphor completion where fluency and coherence are more diagnostic.

3. **Ensemble scoring provides robust performance.** Combining Gemini and Nemotron achieves 0.89–0.93 AUC across all tasks, consistently outperforming either model alone.

4. **Perplexity has limited utility for invalid detection.** While perplexity can detect gibberish and misspellings, it underperforms the LLM-based approaches (AUC 0.66–0.77). Invalid responses often have high perplexity, but so do creative valid responses. Perplexity may be better suited for detecting generic or clichéd responses rather than invalid ones.

5. **Human labels contain noise.** Analysis of false positives reveals that many flagged "valid" responses are legitimately low-quality (e.g., primary uses instead of alternative uses, literal completions instead of figurative metaphors).

## Installation

```bash
git clone https://github.com/rebeaty/invalids.git
cd invalids
pip install -r requirements.txt
```

Create a `.env` file with your API credentials:
```
GEMINI_API_KEY=your-gemini-api-key
NVIDIA_API_KEY=your-nvidia-api-key
```

## Usage

```bash
# Score all tasks (limited sample for testing)
python score_invalid_v5.py --task all --limit 200

# Score full dataset
python score_invalid_v5.py --task all --limit 0

# Score a single task
python score_invalid_v5.py --task aut

# Use only Gemini (faster, no NVIDIA API required)
python score_invalid_v5.py --task all --skip-nvidia --skip-ppl
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--task {aut,dpt,meta,all}` | Task(s) to score | `aut` |
| `--limit N` | Maximum rows per task (0 for all) | 200 |
| `--original` | Use original data instead of augmented | augmented |
| `--skip-gemini` | Skip Gemini scoring | enabled |
| `--skip-nvidia` | Skip Nemotron scoring | enabled |
| `--skip-ppl` | Skip perplexity scoring | enabled |
| `--workers N` | Parallel API workers | 10 |
| `--rps N` | API rate limit (requests/second) | 2.0 |

## Output

### Generated Files

Per-task scored data:
```
aut/aut_scored_augmented_v5.csv
dpt/dpt_scored_augmented_v5.csv
meta/meta_scored_augmented_v5.csv
```

Cross-task summary:
```
results_summary_augmented_v5.csv
```

### Output Columns

| Column | Type | Description |
|--------|------|-------------|
| `gemini_quality` | int (1–5) | Validity rating from Gemini |
| `gemini_rationale` | string | Model explanation for the rating |
| `nvidia_reward` | float | Helpfulness score from Nemotron |
| `perplexity` | float | GPT-2 perplexity score |
| `source` | string | Augmentation type or "original" |

## Data

### Dataset Structure

Each task directory contains:

| File | Description |
|------|-------------|
| `train_stratified.csv` | Original human-labeled responses (~5% invalid) |
| `train_stratified_augmented.csv` | Balanced dataset with synthetic invalid responses (50% invalid) |

### Synthetic Invalid Generation

To create balanced evaluation data, valid responses were augmented with synthetic invalid responses:

| Type | Transformation |
|------|---------------|
| Refusal | Replace with "idk", "n/a", "I don't understand" |
| Misspelling | Introduce character-level corruption |
| Truncation | Cut response mid-sentence |
| Nonsense | Replace with random word combinations |
| Wrong task | Swap with response from different prompt |

## Recommended Usage

For production filtering:

1. **AUT and DPT tasks**: Use Gemini with threshold ≤2
2. **Metaphor completion**: Use Nemotron reward model or ensemble
3. **All tasks**: Ensemble scoring provides the most robust performance

Typical workflow:
1. Run scorer on response data
2. Flag responses with `gemini_quality ≤ 2` as invalid
3. Exclude flagged responses from downstream analysis
4. Optionally review borderline cases (`gemini_quality = 3`)

## Citation

If you use this tool in your research, please cite:

```
@misc{invalids2025,
  author = {Beaty, Roger},
  title = {Invalid Response Detection for Creativity Assessments},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/rebeaty/invalids}
}
```

## References

- Beaty, R. E., & Johnson, D. R. (2021). Automating creativity assessment with SemDis: An open platform for computing semantic distance. *Behavior Research Methods*, 53, 757-780.
- Boussioux, L., et al. (2024). The Crowdless Future? Generative AI and Creative Problem-Solving. *Harvard Business School Working Paper*.

## License

MIT License
