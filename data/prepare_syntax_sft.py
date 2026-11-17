"""
prepare_syntax_sft.py
Loads syntax_sft_examples.json, formats into Qwen chat template,
splits 90/10 train/val, saves HuggingFace Dataset to disk.
"""
import json
import os
from datasets import Dataset
from transformers import AutoTokenizer

DATA_PATH    = "data/syntax_sft_examples.json"
OUTPUT_DIR   = "data/syntax_sft_processed"
MODEL_PATH   = "Qwen/Qwen2.5-3B-Instruct"
VAL_SPLIT    = 0.10
SEED         = 42

def format_example(example, tokenizer):
    messages = [
        {"role": "user",      "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading data...")
    with open(DATA_PATH) as f:
        raw = json.load(f)
    print(f"  Loaded {len(raw)} examples")

    dataset = Dataset.from_list(raw)
    dataset = dataset.map(lambda ex: format_example(ex, tokenizer))
    dataset = dataset.remove_columns(["input", "output"])

    split = dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    train_ds = split["train"]
    val_ds   = split["test"]

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_ds.save_to_disk(os.path.join(OUTPUT_DIR, "train"))
    val_ds.save_to_disk(os.path.join(OUTPUT_DIR, "val"))

    print(f"\nSample formatted example:\n{train_ds[0]['text']}")
    print(f"\nDatasets saved to {OUTPUT_DIR} ✓")

if __name__ == "__main__":
    main()
