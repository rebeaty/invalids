#!/usr/bin/env python3
"""
Generate English-Only Metaphor Invalid Detection Dataset
=========================================================

Filters the Bilingual Creative Metaphor Task data to English only,
then creates a balanced dataset for invalid response detection.

NO TRAIN/TEST SPLITS - just generates a single scored dataset.

Label assignment (based on na_count):
- na_count=5 → High-confidence INVALID (all raters marked NA)
- na_count=1 → High-confidence VALID (only 1 rater marked NA)
- na_count 2-4 → Excluded (ambiguous)

Augmentation types (from Laverghetta et al. preprocess_data.py):
1. random_multi_word - Replace 1-N words with random dictionary words
2. mispell - Replace 1-N characters in words with random letters  
3. strip - Truncate up to 75% of the response
4. didnt_understand - Response from a different prompt (wrong task)
5. dont_know - Random "I don't know" variation

Usage:
    python generate_metaphor_english.py \
        --input Metaphor_Rated_Data_Cleaned.csv \
        --output metaphor_english_invalid.jsonl
"""

import argparse
import json
import random
import string
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from math import ceil
import pandas as pd
from pathlib import Path


RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================================
# DONT KNOW ARRAY (from dont_know_array.py)
# ============================================================================

DONT_KNOW_ARRAY = [
    "i don't know",
    "idk", 
    "I don't know",
    "no idea",
    "not sure",
    "can't think of anything",
    "nothing comes to mind",
    "I have no idea",
    "I'm not sure",
    "beats me",
    "no clue",
    "dunno",
    "i cant think of one",
    "??",
    "???",
    "n/a",
    "NA",
    "none",
    "-",
    "pass",
    "skip",
    "next",
    "I don't understand",
    "I don't get it",
    "what?",
    "huh?",
    "um",
    "uh",
    "idk what to say",
    "no answer",
    "blank",
    "nothing",
    "...",
    "?",
    "nope",
    "no",
]

# Simple English word list for nonsensical replacement
RANDOM_WORDS = [
    "banana", "umbrella", "telescope", "refrigerator", "saxophone",
    "volcano", "spaghetti", "helicopter", "kangaroo", "dinosaur",
    "watermelon", "lighthouse", "caterpillar", "strawberry", "penguin",
    "elephant", "butterfly", "chocolate", "submarine", "newspaper",
    "calculator", "hamburger", "trampoline", "microscope", "pineapple",
    "crocodile", "skateboard", "thunderstorm", "marshmallow", "rhinoceros",
    "toothbrush", "basketball", "grasshopper", "sunflower", "motorcycle",
    "accordion", "jellyfish", "parachute", "xylophone", "cauliflower",
    "adventure", "brilliant", "celebrate", "delicious", "enormous",
    "fantastic", "gorgeous", "happiness", "incredible", "jubilant",
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MetaphorItem:
    """A single metaphor item with prompt, response, and label."""
    prompt: str
    response: str
    label: int  # 0 = valid, 1 = invalid
    source: str  # 'human_valid', 'human_invalid', or augment type
    original_response: Optional[str] = None
    na_count: Optional[int] = None


# ============================================================================
# LANGUAGE FILTERING
# ============================================================================

def is_english_prompt(prompt: str) -> bool:
    """Check if prompt is English (not Spanish)."""
    spanish_starters = ('El ', 'La ', 'Los ', 'Las ')
    return not prompt.startswith(spanish_starters)


def clean_prompt(prompt: str) -> str:
    """Remove trailing '...' from prompts."""
    prompt = prompt.strip()
    if prompt.endswith("..."):
        prompt = prompt[:-3].strip()
    return prompt


def tokenize(text: str) -> List[str]:
    """Simple word tokenizer (matching the paper's word_tokenize behavior)."""
    return text.split()


# ============================================================================
# AUGMENTATION TRANSFORMS (matching preprocess_data.py)
# ============================================================================

def transform_random_multi_word(response: str) -> str:
    """Replace 1-N words with random dictionary words."""
    words = tokenize(response)
    english_words = [w for w in words if w.isalpha()]
    
    if len(english_words) == 0:
        return random.choice(RANDOM_WORDS)
    
    total_replacements = random.randint(1, len(english_words))
    result = response
    
    for word in words:
        if total_replacements == 0:
            break
        if word.isalpha():
            new_word = random.choice(RANDOM_WORDS)
            result = result.replace(word, new_word, 1)
            total_replacements -= 1
    
    return result


def transform_mispell(response: str) -> str:
    """Replace 1-N characters in words with random letters."""
    words = tokenize(response)
    english_words = [w for w in words if w.isalpha()]
    
    if len(english_words) == 0:
        return response
    
    total_replacements = random.randint(1, len(english_words))
    result = response
    
    for word in words:
        if total_replacements == 0:
            break
        if word.isalpha() and len(word) > 0:
            i = random.randint(0, len(word) - 1)
            new_word = word[:i] + random.choice(string.ascii_letters) + word[i + 1:]
            result = result.replace(word, new_word, 1)
            total_replacements -= 1
    
    return result


def transform_strip(response: str) -> str:
    """Truncate up to 75% of the response."""
    if len(response) <= 1:
        return response
    
    # Strip at most 75% - keep at least 25%
    len_to_keep = random.randint(1, ceil(len(response) * 0.75))
    return response[:len_to_keep]


def transform_didnt_understand(prompt_response_map: Dict[str, List[str]], current_prompt: str) -> str:
    """Return a valid response from a different prompt."""
    other_responses = []
    for prompt, responses in prompt_response_map.items():
        if prompt != current_prompt:
            other_responses.extend(responses)
    
    if other_responses:
        return random.choice(other_responses)
    return "something completely different"


def transform_dont_know() -> str:
    """Return a random 'I don't know' variation."""
    return random.choice(DONT_KNOW_ARRAY)


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_english_metaphor_data(
    input_path: str,
    invalid_threshold: int = 5,
    valid_threshold: int = 1
) -> Tuple[List[MetaphorItem], List[MetaphorItem]]:
    """
    Load the metaphor data, filter to English, and split by na_count.
    """
    df = pd.read_csv(input_path)
    
    # Filter to English only
    df_english = df[df['Prompt'].apply(is_english_prompt)].copy()
    
    print(f"Total responses: {len(df)}")
    print(f"English responses: {len(df_english)}")
    print(f"Spanish responses (filtered out): {len(df) - len(df_english)}")
    
    valid_items = []
    invalid_items = []
    
    for _, row in df_english.iterrows():
        prompt = clean_prompt(row['Prompt'])
        response = str(row['Response']) if pd.notna(row['Response']) else ""
        na_count = row['na_count']
        
        if na_count >= invalid_threshold:
            invalid_items.append(MetaphorItem(
                prompt=prompt,
                response=response,
                label=1,
                source="human_invalid",
                na_count=na_count
            ))
        elif na_count <= valid_threshold:
            valid_items.append(MetaphorItem(
                prompt=prompt,
                response=response,
                label=0,
                source="human_valid",
                na_count=na_count
            ))
    
    return valid_items, invalid_items


def augment_to_balance(
    valid_items: List[MetaphorItem],
    invalid_items: List[MetaphorItem],
) -> List[MetaphorItem]:
    """
    Augment valid responses to create invalid samples until balanced.
    
    Following the paper: for each prompt, augment until n_invalid = n_valid.
    Uses all 5 augmentation types evenly.
    """
    # Build prompt -> responses map
    prompt_response_map: Dict[str, List[str]] = {}
    prompt_valid_counts: Dict[str, int] = {}
    prompt_invalid_counts: Dict[str, int] = {}
    
    for item in valid_items:
        if item.prompt not in prompt_response_map:
            prompt_response_map[item.prompt] = []
            prompt_valid_counts[item.prompt] = 0
        prompt_response_map[item.prompt].append(item.response)
        prompt_valid_counts[item.prompt] += 1
    
    for item in invalid_items:
        if item.prompt not in prompt_invalid_counts:
            prompt_invalid_counts[item.prompt] = 0
        prompt_invalid_counts[item.prompt] += 1
    
    # Start with existing items
    all_invalid = list(invalid_items)
    augment_types = ["random_multi_word", "mispell", "strip", "didnt_understand", "dont_know"]
    
    # For each prompt, augment to balance
    for prompt in prompt_valid_counts:
        n_valid = prompt_valid_counts[prompt]
        n_invalid = prompt_invalid_counts.get(prompt, 0)
        target_augment = n_valid - n_invalid
        
        if target_augment <= 0:
            continue
        
        # Get valid responses for this prompt to augment
        prompt_responses = prompt_response_map[prompt]
        
        # Calculate how many of each augment type
        per_type = ceil(target_augment / len(augment_types))
        
        augmented_count = 0
        for aug_type in augment_types:
            for _ in range(per_type):
                if augmented_count >= target_augment:
                    break
                
                # Pick a random valid response to augment
                original = random.choice(prompt_responses)
                
                # Skip if no alphabetic words (can't augment)
                words = tokenize(original)
                if not any(w.isalpha() for w in words):
                    continue
                
                # Apply transform
                if aug_type == "random_multi_word":
                    augmented = transform_random_multi_word(original)
                elif aug_type == "mispell":
                    augmented = transform_mispell(original)
                elif aug_type == "strip":
                    augmented = transform_strip(original)
                elif aug_type == "didnt_understand":
                    augmented = transform_didnt_understand(prompt_response_map, prompt)
                else:  # dont_know
                    augmented = transform_dont_know()
                
                all_invalid.append(MetaphorItem(
                    prompt=prompt,
                    response=augmented,
                    label=1,
                    source=aug_type,
                    original_response=original
                ))
                augmented_count += 1
    
    return all_invalid


def save_dataset(items: List[MetaphorItem], output_path: str):
    """Save dataset to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + '\n')


def print_statistics(valid_items: List[MetaphorItem], invalid_items: List[MetaphorItem]):
    """Print dataset statistics."""
    print(f"\n{'='*50}")
    print("Dataset Statistics")
    print(f"{'='*50}")
    print(f"Valid (label=0): {len(valid_items)}")
    print(f"Invalid (label=1): {len(invalid_items)}")
    print(f"Total: {len(valid_items) + len(invalid_items)}")
    
    # Source breakdown for invalid
    sources = {}
    for item in invalid_items:
        sources[item.source] = sources.get(item.source, 0) + 1
    
    print(f"\nInvalid by source:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate English-only metaphor invalid detection dataset"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV (Metaphor_Rated_Data_Cleaned.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="metaphor_english_invalid.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--invalid-threshold",
        type=int,
        default=5,
        help="na_count >= this is INVALID (default: 5)"
    )
    parser.add_argument(
        "--valid-threshold",
        type=int,
        default=1,
        help="na_count <= this is VALID (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip augmentation, use only human labels"
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    print("=" * 60)
    print("English-Only Metaphor Invalid Detection Dataset")
    print("=" * 60)
    
    # Load data (English only)
    print(f"\nLoading data from {args.input}...")
    valid_items, invalid_items = load_english_metaphor_data(
        args.input,
        invalid_threshold=args.invalid_threshold,
        valid_threshold=args.valid_threshold
    )
    
    print(f"\nAfter filtering:")
    print(f"  Valid (na_count<={args.valid_threshold}): {len(valid_items)}")
    print(f"  Invalid (na_count>={args.invalid_threshold}): {len(invalid_items)}")
    
    # Augment to balance
    if not args.no_augment:
        print(f"\nAugmenting to balance...")
        invalid_items = augment_to_balance(valid_items, invalid_items)
    
    print_statistics(valid_items, invalid_items)
    
    # Combine and shuffle
    all_items = valid_items + invalid_items
    random.shuffle(all_items)
    
    # Save
    save_dataset(all_items, args.output)
    print(f"\n✓ Saved {len(all_items)} items to {args.output}")


if __name__ == "__main__":
    main()
