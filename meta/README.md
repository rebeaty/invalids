# Metaphor Task Data

## Files

| File | Description |
|------|-------------|
| `train_stratified_augmented.csv` | Processed dataset for scoring (English only, balanced) |
| `Metaphor_Rated_Data_Cleaned.csv` | Raw source data (8000 responses, English + Spanish) |
| `generate_data.py` | Script to regenerate dataset from raw |

## Data Format

```csv
prompt,response,label,augment_type
The tall tree is,a giant,0,
The strong wind is,powerfu,1,strip
```

- `prompt`: Metaphor stem (e.g., "The tall tree is")
- `response`: Completion
- `label`: 0=valid, 1=invalid
- `augment_type`: Empty for human data, else augmentation type

## Label Assignment

From raw data (`na_count` = number of raters who marked NA):
- **Valid (label=0)**: `na_count=1` (only 1 of 5 raters marked NA)
- **Invalid (label=1)**: `na_count=5` (all 5 raters marked NA) + augmented

## Regenerate Dataset

```bash
python generate_data.py \
    --input Metaphor_Rated_Data_Cleaned.csv \
    --output train_stratified_augmented.csv
```

## Augmentation Types

Invalid responses are augmented to balance the dataset:
- `random_multi_word` - Random dictionary word substitutions
- `mispell` - Character-level typos  
- `strip` - Truncated to 25-75% length
- `didnt_understand` - Response from different prompt
- `dont_know` - Random "I don't know" variations

## Statistics

- Total: 1560 items (50/50 balanced)
- Valid: 780 (all human-labeled)
- Invalid: 780 (51 human + 729 augmented)
