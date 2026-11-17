"""
prepare_trajectory_sft.py
Tokenises trajectory_sft_examples.json and saves train/val splits as HuggingFace datasets.

Each saved example includes:
  input_ids, attention_mask, labels   — standard fields
  reasoning_start, reasoning_end      — absolute token indices of reasoning content
  tool_start, tool_end                — absolute token indices of tool content

These indices are stored at preparation time so training needs no tag searching.
Ground truth is always well-formed so no format checking is needed here.
"""

import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH   = "data/trajectory_sft_examples.json"
OUT_DIR     = "data/trajectory_sft_processed"
MAX_SEQ_LEN = 512
VAL_FRAC    = 0.10

SYSTEM_PROMPT = (
    "You are a personal assistant with access to the following tools: "
    "zoom, email, text, calendar, contacts, clock, stop. "
    "Think through each step carefully before acting."
)

REASONING_OPEN  = "<reasoning>"
REASONING_CLOSE = "</reasoning>"
TOOL_OPEN       = "<tool>"
TOOL_CLOSE      = "</tool>"


def char_to_token(token_ids, tokenizer, char_pos):
    """Binary search: find the first token index whose decoded prefix reaches char_pos."""
    lo, hi = 0, len(token_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        prefix = tokenizer.decode(token_ids[:mid], skip_special_tokens=False)
        if len(prefix) < char_pos:
            lo = mid + 1
        else:
            hi = mid
    return lo


def find_content_span(token_ids, tokenizer, open_tag, close_tag):
    """
    Return (content_start, content_end) token indices for text between open_tag and close_tag.
    content_start = first token after open_tag
    content_end   = first token of close_tag (exclusive end)
    Returns (None, None) if either tag is missing.
    """
    full_text  = tokenizer.decode(token_ids, skip_special_tokens=False)
    open_char  = full_text.find(open_tag)
    close_char = full_text.find(close_tag)

    if open_char == -1 or close_char == -1 or close_char <= open_char:
        return None, None

    content_start_char = open_char + len(open_tag)
    content_end_char   = close_char

    start_tok = char_to_token(token_ids, tokenizer, content_start_char)
    end_tok   = char_to_token(token_ids, tokenizer, content_end_char)
    return start_tok, end_tok


def tokenise_example(example, tokenizer):
    # build full sequence and prompt-only sequence
    full_text = tokenizer.apply_chat_template(
        [{"role": "system",    "content": SYSTEM_PROMPT},
         {"role": "user",      "content": example["input"]},
         {"role": "assistant", "content": example["output"]}],
        tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": example["input"]}],
        tokenize=False, add_generation_prompt=True
    )

    full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)

    # mask input tokens from loss
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # truncate
    full_ids = full_ids[:MAX_SEQ_LEN]
    labels   = labels[:MAX_SEQ_LEN]

    # find spans in the output portion only, then convert to absolute indices
    output_ids = full_ids[prompt_len:]

    r_start, r_end = find_content_span(output_ids, tokenizer, REASONING_OPEN,  REASONING_CLOSE)
    t_start, t_end = find_content_span(output_ids, tokenizer, TOOL_OPEN,       TOOL_CLOSE)

    def to_abs(idx):
        return (prompt_len + idx) if idx is not None else -1

    return {
        "input_ids":        full_ids,
        "attention_mask":   [1] * len(full_ids),
        "labels":           labels,
        "reasoning_start":  to_abs(r_start),
        "reasoning_end":    to_abs(r_end),
        "tool_start":       to_abs(t_start),
        "tool_end":         to_abs(t_end),
    }


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        raw = json.load(f)
    print(f"  {len(raw)} examples loaded")

    print("Tokenising and finding tag spans...")
    tokenised = [tokenise_example(ex, tokenizer) for ex in raw]

    # report how many examples have valid spans
    n_ok = sum(1 for t in tokenised if t["reasoning_start"] != -1 and t["tool_start"] != -1)
    print(f"  Valid spans (both tags found): {n_ok}/{len(tokenised)}")

    # filter examples where output was fully truncated (labels all -100)
    before = len(tokenised)
    tokenised = [t for t in tokenised if any(l != -100 for l in t["labels"])]
    print(f"  Filtered {before - len(tokenised)} fully-masked examples")

    split_idx  = int(len(tokenised) * (1 - VAL_FRAC))
    train_data = tokenised[:split_idx]
    val_data   = tokenised[split_idx:]
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_data).save_to_disk(f"{OUT_DIR}/train")
    Dataset.from_list(val_data).save_to_disk(f"{OUT_DIR}/val")
    print(f"Saved to {OUT_DIR}/train and {OUT_DIR}/val")


if __name__ == "__main__":
    main()
