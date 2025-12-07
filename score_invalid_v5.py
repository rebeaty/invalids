#!/usr/bin/env python3
"""
Invalid Response Detection for Creativity Assessments (v5)
===========================================================

Scores creativity task responses to detect invalid entries for exclusion before
downstream analysis. Invalid responses include gibberish, refusals ("idk", "n/a"),
off-topic answers, truncated text, misspellings, and wrong-task responses.

Part of the Amazon Creativity Benchmark project.

Scoring Methods
---------------
  1. Gemini LLM Judge (1-5 validity scale)
     Task-specific prompts explain what constitutes a valid response for each
     task type. The 1-5 scale lets the model reason about borderline cases;
     scores naturally separate into valid (3-5) vs invalid (1-2). Returns
     both a validity rating and rationale.

  2. NVIDIA Nemotron-70B Reward Model
     Measures "helpfulness" of the response given the prompt. Invalid responses
     (empty, off-topic, nonsensical) score low on helpfulness. Provides a
     complementary signal to the Gemini judge.

  3. GPT-2 Conditional Perplexity (local, no API required)
     Computes P(response | prompt)—how surprising is this response given the
     prompt? Effective at catching misspellings and garbled text. Note: returns
     NaN for single-word responses due to BPE tokenization edge cases.

Supported Tasks
---------------
  - AUT (Alternative Uses): "What is a use for a [object]?"
  - DPT (Design Problems): "How would you [problem]?"
  - META (Metaphor Completion): Complete sentence stems figuratively

Usage
-----
  python score_invalid_v5.py --task all --limit 200   # Quick test
  python score_invalid_v5.py --task all --limit 0     # Full run
  python score_invalid_v5.py --task meta --skip-ppl   # Single task

Output
------
  - Per-task: {task}/{task}_scored_augmented_v5.csv
  - Summary: results_summary_augmented_v5.csv

See README.md for full methodology and results.

Requirements
------------
  pip install pandas numpy scipy scikit-learn tqdm torch transformers
  pip install google-genai openai python-dotenv

  API keys in .env:
    GEMINI_API_KEY=...
    NVIDIA_API_KEY=nvapi-...
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "run_gemini": True,
    "run_nvidia": True,
    "run_perplexity": True,
    "workers": 10,
    "rps": 2.0,
}

# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
import time
import json
import argparse
import threading
from pathlib import Path
from typing import Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve

# =============================================================================
# ENVIRONMENT
# =============================================================================

try:
    from dotenv import load_dotenv
    script_dir = Path(__file__).parent
    load_dotenv(script_dir / ".env", override=True)
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps if rps > 0 else 0
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()


# =============================================================================
# TASK-SPECIFIC 1-5 VALIDITY PROMPTS
# =============================================================================

PROMPT_AUT = """You are evaluating whether a response is valid for an Alternative Uses Task.

In this task, participants propose alternative uses for everyday objects.
The prompt gives an object (e.g., "brick") and the response should be a use for it.

OBJECT: {object}
RESPONSE: {response}

Rate the validity from 1 to 5:
  1 = Invalid - nonsensical, gibberish, blank, "I don't know", or completely unrelated
  2 = Very poor - barely coherent, just describes the object, or extremely unclear
  3 = Okay - a recognizable attempt at an alternative use but weak or vague
  4 = Good - a clear alternative use that makes sense
  5 = Excellent - a well-articulated alternative use

A response is INVALID (rating 1) if it:
- Is gibberish or random words
- Is blank, "idk", "n/a", "???", or similar non-responses
- Just describes what the object is (not a USE)
- Is an incomplete fragment that makes no sense
- Says something completely unrelated to the object

Return JSON only: {{"quality": <1-5>, "rationale": "<brief explanation>"}}"""


PROMPT_DPT = """You are evaluating whether a response is valid for a Design Problems Task.

In this task, participants propose solutions to open-ended design challenges.
The prompt gives a problem (e.g., "reduce traffic in cities") and the response should propose a solution.

PROBLEM: How would you {problem}?
RESPONSE: {response}

Rate the validity from 1 to 5:
  1 = Invalid - nonsensical, gibberish, blank, "I don't know", or completely unrelated
  2 = Very poor - barely coherent, not a solution attempt, or extremely unclear
  3 = Okay - a recognizable attempt at a solution but weak or vague
  4 = Good - a clear solution that addresses the problem
  5 = Excellent - a well-articulated solution

A response is INVALID (rating 1) if it:
- Is gibberish or random words
- Is blank, "idk", "n/a", "???", or similar non-responses
- Just restates the problem without proposing anything
- Is an incomplete fragment that makes no sense
- Says something completely unrelated to the problem

Return JSON only: {{"quality": <1-5>, "rationale": "<brief explanation>"}}"""


PROMPT_META = """You are evaluating whether a response is valid for a Metaphor Completion Task.

In this task, participants complete a sentence stem with a metaphor.
The prompt gives the beginning (e.g., "The tall tree is") and the response should be a metaphorical completion.

PROMPT: {stem}
RESPONSE: {response}

Rate the validity from 1 to 5:
  1 = Invalid - nonsensical, gibberish, blank, "I don't know", or completely unrelated
  2 = Very poor - barely coherent, not a metaphor (purely literal), or extremely unclear
  3 = Okay - a recognizable attempt at metaphor but weak or awkward
  4 = Good - a clear metaphorical response that fits the prompt
  5 = Excellent - a well-formed metaphor that completes the prompt

A response is INVALID (rating 1) if it:
- Is gibberish or random words
- Is blank, "idk", "n/a", "???", or similar non-responses
- Just repeats the prompt or says something completely unrelated
- Is an incomplete fragment that makes no sense

Return JSON only: {{"quality": <1-5>, "rationale": "<brief explanation>"}}"""


# =============================================================================
# GEMINI SCORER (1-5 VALIDITY)
# =============================================================================

def init_gemini():
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"  Gemini error: {e}")
        return None


def parse_gemini_quality(text: str) -> Tuple[Optional[int], str]:
    """Extract quality rating (1-5) and rationale from Gemini response."""
    try:
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            quality = int(data.get("quality", data.get("rating", 3)))
            rationale = data.get("rationale", "")
            return quality, rationale
    except:
        pass
    # Fallback: find any 1-5 number
    nums = re.findall(r'\b([1-5])\b', text)
    if nums:
        return int(nums[0]), text
    return None, text


def score_gemini(args) -> Tuple[int, Dict]:
    """Score one item with Gemini using task-specific 1-5 prompt."""
    idx, task, prompt_text, response, client, limiter = args
    result = {
        "gemini_quality": None,  # 1-5 score
        "gemini_rationale": ""
    }
    
    if not client:
        return idx, result
    
    response_text = str(response).strip() if pd.notna(response) else ""
    prompt_text = str(prompt_text).strip() if pd.notna(prompt_text) else ""
    
    # Select task-specific prompt
    if task == "aut":
        evaluation_prompt = PROMPT_AUT.format(object=prompt_text, response=response_text)
    elif task == "dpt":
        evaluation_prompt = PROMPT_DPT.format(problem=prompt_text.rstrip('.'), response=response_text)
    else:  # meta
        evaluation_prompt = PROMPT_META.format(stem=prompt_text, response=response_text)
    
    for attempt in range(MAX_RETRIES):
        try:
            limiter.wait()
            resp = client.models.generate_content(model="gemini-2.0-flash", contents=evaluation_prompt)
            if resp.text:
                quality, rationale = parse_gemini_quality(resp.text)
                result["gemini_quality"] = quality
                result["gemini_rationale"] = rationale
                return idx, result
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                result["gemini_rationale"] = f"Error: {e}"
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    
    return idx, result


# =============================================================================
# NVIDIA REWARD SCORER
# =============================================================================

def score_nvidia(args) -> Tuple[int, Dict]:
    """Score one item with NVIDIA Nemotron reward model."""
    idx, task, prompt_text, response, limiter = args
    result = {"nvidia_reward": None, "nvidia_raw": ""}
    
    if not NVIDIA_API_KEY:
        return idx, result
    
    response_text = str(response).strip() if pd.notna(response) else ""
    prompt_text = str(prompt_text).strip() if pd.notna(prompt_text) else ""
    
    from openai import OpenAI
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1")
    
    # Reconstruct prompt for context
    if task == "aut":
        user_msg = f"What is a use for a {prompt_text}?"
    elif task == "dpt":
        user_msg = f"How would you {prompt_text.rstrip('.')}?"
    else:  # meta
        user_msg = f"Complete the sentence: {prompt_text}"
    
    messages = [{"role": "user", "content": user_msg}, {"role": "assistant", "content": response_text}]
    
    for attempt in range(MAX_RETRIES):
        try:
            limiter.wait()
            resp = client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-reward",
                messages=messages, max_tokens=64, temperature=0
            )
            text = resp.choices[0].message.content or ""
            result["nvidia_raw"] = text
            
            match = re.search(r'helpfulness[:\s]*([-+]?\d+\.?\d*)', text.lower())
            if match:
                result["nvidia_reward"] = float(match.group(1))
            else:
                nums = re.findall(r'([-+]?\d+\.?\d*)', text)
                if nums:
                    result["nvidia_reward"] = float(nums[-1])
            return idx, result
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                result["nvidia_raw"] = f"Error: {e}"
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    
    return idx, result


# =============================================================================
# GPT-2 PERPLEXITY SCORER (LOCAL)
# =============================================================================

def load_gpt2():
    """Load GPT-2 model for perplexity scoring."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    
    print("  Loading GPT-2...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"  Loaded on {device}")
    return model, tokenizer, device


def compute_perplexity(task: str, prompt: str, response: str, model, tokenizer, device) -> float:
    """Compute conditional perplexity: P(response | formatted_prompt).
    
    Higher perplexity = response is more surprising/unexpected given the prompt.
    Returns NaN for very short responses where BPE boundaries are ambiguous.
    """
    import torch
    
    response = str(response).strip() if response else ""
    if not response:
        return float("nan")
    
    if task == "aut":
        formatted_prompt = f"What is a use for a {prompt}?"
    elif task == "dpt":
        formatted_prompt = f"How would you {prompt.rstrip('.')}?"
    else:
        formatted_prompt = f"Complete the sentence: {prompt}"
    
    try:
        full_text = f"{formatted_prompt} {response}"
        prompt_only = f"{formatted_prompt} "
        
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
        prompt_ids = tokenizer(prompt_only, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
        
        prompt_len = prompt_ids.shape[1]
        
        if full_ids.shape[1] <= prompt_len:
            return float("nan")
        
        labels = full_ids.clone()
        labels[0, :prompt_len] = -100
        
        with torch.no_grad():
            outputs = model(full_ids, labels=labels)
            loss = outputs.loss.item()
        
        return np.exp(loss)
    except:
        return float("nan")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(y_true, scores, higher_is_better=True):
    """Compute AUC and Spearman for invalid detection."""
    mask = ~(pd.isna(y_true) | pd.isna(scores))
    y, s = np.array(y_true[mask]), np.array(scores[mask])
    
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {"n": len(y), "error": "Insufficient data"}
    
    try:
        # For quality scores: higher = valid, so we negate for AUC (label=1 is invalid)
        auc = roc_auc_score(y, -s if higher_is_better else s)
        rho, _ = spearmanr(y, s)
        
        prec, rec, _ = precision_recall_curve(y, -s if higher_is_better else s)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        best_f1 = float(np.max(f1[:-1])) if len(f1) > 1 else np.nan
        
        return {"n": len(y), "auc": auc, "spearman": rho, "best_f1": best_f1,
                "n_valid": int(len(y) - y.sum()), "n_invalid": int(y.sum())}
    except Exception as e:
        return {"n": len(y), "error": str(e)}


def print_result(name, r):
    print(f"\n{name}:")
    if "error" in r:
        print(f"  ERROR: {r['error']} (n={r.get('n', 0)})")
    else:
        print(f"  AUC: {r['auc']:.3f} | Spearman: {r['spearman']:.3f} | F1: {r['best_f1']:.3f}")
        print(f"  n={r['n']} (valid={r['n_valid']}, invalid={r['n_invalid']})")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_task_data(task: str, script_dir: Path, augmented: bool = True) -> pd.DataFrame:
    filename = "train_stratified_augmented.csv" if augmented else "train_stratified.csv"
    path = script_dir / task / filename
    
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    
    df = pd.read_csv(path)
    
    if "augment_type" in df.columns:
        df = df.rename(columns={"augment_type": "source"})
    if "source" not in df.columns:
        df["source"] = ""
    df["source"] = df["source"].fillna("").replace("", "original")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Score creativity responses (v5 - task-specific 1-5 validity)")
    parser.add_argument("--task", "-t", choices=["aut", "dpt", "meta", "all"], default="aut")
    parser.add_argument("--augmented", "-a", action="store_true", default=True)
    parser.add_argument("--original", "-o", action="store_true")
    parser.add_argument("--limit", "-n", type=int, default=200, help="Max rows per task (default 200)")
    parser.add_argument("--workers", "-w", type=int, default=CONFIG["workers"])
    parser.add_argument("--rps", type=float, default=CONFIG["rps"])
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-nvidia", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    args = parser.parse_args()
    
    augmented = not args.original
    run_gemini = CONFIG["run_gemini"] and not args.skip_gemini
    run_nvidia = CONFIG["run_nvidia"] and not args.skip_nvidia
    run_ppl = CONFIG["run_perplexity"] and not args.skip_ppl
    
    dataset_type = "augmented" if augmented else "original"
    
    print("=" * 60)
    print("Invalid Detection v5 - Task-Specific 1-5 Validity Prompts")
    print("=" * 60)
    print(f"\nDataset: {dataset_type.upper()}")
    print(f"Scorers: Gemini={'✓' if run_gemini else '✗'} NVIDIA={'✓' if run_nvidia else '✗'} PPL={'✓' if run_ppl else '✗'}")
    
    script_dir = Path(__file__).parent
    tasks = ["aut", "dpt", "meta"] if args.task == "all" else [args.task]
    all_results = []
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task.upper()}")
        print("=" * 60)
        
        try:
            df = load_task_data(task, script_dir, augmented=augmented)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        
        n_valid = (df['label']==0).sum()
        n_invalid = (df['label']==1).sum()
        print(f"Loaded {len(df)} rows | Valid: {n_valid} | Invalid: {n_invalid}")
        
        # Limit rows
        if args.limit and args.limit > 0 and len(df) > args.limit:
            n_each = args.limit // 2
            df = pd.concat([
                df[df['label']==0].sample(n=min(n_each, n_valid), random_state=42),
                df[df['label']==1].sample(n=min(n_each, n_invalid), random_state=42)
            ]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Limited to {len(df)} rows")
        
        # Baselines
        df["wordcount"] = df["response"].astype(str).str.split().str.len()
        
        # Initialize columns
        df["gemini_quality"] = None
        df["gemini_rationale"] = ""
        df["nvidia_reward"] = None
        df["perplexity"] = None
        
        n = len(df)
        limiter = RateLimiter(args.rps)
        
        # --- Gemini (1-5 validity) ---
        if run_gemini and GEMINI_API_KEY:
            print("\n--- Gemini (1-5 Validity) ---")
            client = init_gemini()
            if client:
                tasks_list = [(i, task, row["prompt"], row["response"], client, limiter) 
                              for i, row in df.iterrows()]
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futures = {ex.submit(score_gemini, t): t[0] for t in tasks_list}
                    for f in tqdm(as_completed(futures), total=n, desc="Gemini"):
                        idx, res = f.result()
                        for k, v in res.items():
                            df.at[idx, k] = v
                
                # Show distribution
                print(f"  Validity distribution: {df['gemini_quality'].value_counts().sort_index().to_dict()}")
        
        # --- NVIDIA ---
        if run_nvidia and NVIDIA_API_KEY:
            print("\n--- NVIDIA Reward ---")
            tasks_list = [(i, task, row["prompt"], row["response"], limiter) 
                          for i, row in df.iterrows()]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(score_nvidia, t): t[0] for t in tasks_list}
                for f in tqdm(as_completed(futures), total=n, desc="NVIDIA"):
                    idx, res = f.result()
                    for k, v in res.items():
                        df.at[idx, k] = v
        
        # --- GPT-2 Perplexity ---
        if run_ppl:
            print("\n--- GPT-2 Perplexity ---")
            model, tokenizer, device = load_gpt2()
            ppls = []
            for _, row in tqdm(df.iterrows(), total=n, desc="Perplexity"):
                prompt = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
                response = str(row["response"]) if pd.notna(row["response"]) else ""
                ppls.append(compute_perplexity(task, prompt, response, model, tokenizer, device))
            df["perplexity"] = ppls
        
        # --- Evaluate ---
        print(f"\n{'='*50}")
        print(f"RESULTS: {task.upper()}")
        print("=" * 50)
        
        wc = evaluate(df["label"], df["wordcount"], higher_is_better=True)
        print_result("Wordcount (baseline)", wc)
        all_results.append({"task": task, "dataset": dataset_type, "metric": "wordcount", **wc})
        
        if run_gemini:
            # Validity score: higher = valid, so higher_is_better=True for detecting invalids
            gem = evaluate(df["label"], df["gemini_quality"], higher_is_better=True)
            print_result("Gemini Validity (1-5)", gem)
            all_results.append({"task": task, "dataset": dataset_type, "metric": "gemini", **gem})
            
            # Show mean by label
            print(f"  Mean validity by label:")
            print(f"    Valid (label=0): {df[df['label']==0]['gemini_quality'].mean():.2f}")
            print(f"    Invalid (label=1): {df[df['label']==1]['gemini_quality'].mean():.2f}")
        
        if run_nvidia:
            nv = evaluate(df["label"], df["nvidia_reward"], higher_is_better=True)
            print_result("NVIDIA Reward", nv)
            all_results.append({"task": task, "dataset": dataset_type, "metric": "nvidia", **nv})
        
        if run_ppl:
            ppl = evaluate(df["label"], df["perplexity"], higher_is_better=False)
            print_result("GPT-2 Perplexity", ppl)
            all_results.append({"task": task, "dataset": dataset_type, "metric": "perplexity", **ppl})
        
        # Save
        out_path = script_dir / task / f"{task}_scored_{dataset_type}_v5.csv"
        df.to_csv(out_path, index=False)
        print(f"\n✓ Saved: {out_path}")
    
    # Summary
    if all_results:
        summary = pd.DataFrame(all_results)
        summary_path = script_dir / f"results_summary_{dataset_type}_v5.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n✓ Summary: {summary_path}")
        
        # Print comparison table
        print(f"\n{'='*60}")
        print("SUMMARY: AUC by Task/Metric")
        print("=" * 60)
        pivot = summary.pivot(index='task', columns='metric', values='auc')
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
