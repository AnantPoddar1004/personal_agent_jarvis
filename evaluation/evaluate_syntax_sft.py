"""
evaluate_syntax_sft.py
Evaluates the fine-tuned adapter on the validation set.
- Exact match on: tool name, function, all fields except subject/body
- Semantic match (cosine >= 0.80) on: email subject, email body, text body
- Hard 0 on malformed/unparseable outputs
Prints a full per-tool and per-field accuracy report.
"""
import os
import re
import json
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
BASE_MODEL_PATH  = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH     = "outputs/adapter"
DATA_DIR         = "data/syntax_sft_processed"
RESULTS_PATH     = "outputs/eval_results.json"
EMBED_MODEL      = "all-MiniLM-L6-v2"
SEM_THRESHOLD    = 0.80
MAX_NEW_TOKENS   = 80
BATCH_SIZE       = 16   # inference batch size

# Fields that use semantic matching instead of exact match
# format: tool_name -> list of 0-indexed field positions (after tool|func|)
SEMANTIC_FIELDS = {
    "email": {1: "subject", 2: "body"},   # email|send|TO|SUBJECT|BODY
    "text":  {1: "body"},                  # text|send|TO|BODY
}

# --------------------------------------------------------------------------
# PARSING
# --------------------------------------------------------------------------
EXPECTED_FIELDS = {
    "zoom":     3,   # topic | datetime | duration
    "email":    3,   # to | subject | body
    "text":     2,   # to | body
    "calendar": 3,   # datetime | duration | title
    "contacts": 1,   # name
    "clock":    0,   # no fields
}

def parse_tool_call(raw: str):
    """
    Returns (tool, func, fields_list) or None if malformed.
    """
    match = re.search(r"<tool>(.*?)</tool>", raw, re.DOTALL)
    if not match:
        return None
    content = match.group(1).strip()
    parts = content.split("|")
    if len(parts) < 2:
        return None
    tool = parts[0].strip()
    func = parts[1].strip()
    fields = [p.strip() for p in parts[2:]]

    expected = EXPECTED_FIELDS.get(tool)
    if expected is None:
        return None
    if len(fields) != expected:
        return None

    return tool, func, fields

# --------------------------------------------------------------------------
# SCORING
# --------------------------------------------------------------------------
def score_example(pred_raw: str, gold_raw: str, embed_model) -> dict:
    result = {
        "parse_ok": False,
        "tool_match": False,
        "func_match": False,
        "field_scores": {},
        "overall": False,
    }

    pred = parse_tool_call(pred_raw)
    gold = parse_tool_call(gold_raw)

    if gold is None:
        # Shouldn't happen with our dataset, but guard anyway
        return result
    if pred is None:
        return result  # hard 0

    result["parse_ok"] = True
    g_tool, g_func, g_fields = gold
    p_tool, p_func, p_fields = pred

    result["tool_match"] = (p_tool == g_tool)
    result["func_match"] = (p_func == g_func)

    if not result["tool_match"] or not result["func_match"]:
        return result  # hard 0 — wrong tool

    sem_positions = SEMANTIC_FIELDS.get(g_tool, {})
    all_fields_ok = True

    for i, (p_val, g_val) in enumerate(zip(p_fields, g_fields)):
        field_name = sem_positions.get(i, f"field_{i}")
        if i in sem_positions:
            # Semantic match
            vecs = embed_model.encode([p_val, g_val])
            sim = float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])
            ok = sim >= SEM_THRESHOLD
            result["field_scores"][field_name] = {"match": ok, "score": round(sim, 4), "type": "semantic"}
        else:
            # Exact match
            ok = (p_val == g_val)
            result["field_scores"][field_name] = {"match": ok, "type": "exact"}

        if not ok:
            all_fields_ok = False

    result["overall"] = all_fields_ok
    return result

# --------------------------------------------------------------------------
# INFERENCE
# --------------------------------------------------------------------------
def run_inference(model, tokenizer, prompts: list) -> list:
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=200).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,        # greedy — deterministic eval
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for j, gen in enumerate(generated):
            # Decode only the new tokens
            input_len = inputs["input_ids"].shape[1]
            new_tokens = gen[input_len:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return outputs

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    # Load embedding model
    print("Loading sentence embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    # Load tokenizer + model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"  # left-pad for generation batching

    print("Loading base model + adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    # Load val set — recover original input/output from chat-formatted text
    print("Loading validation data...")
    val_ds = load_from_disk(os.path.join(DATA_DIR, "val"))

    # Re-load original val pairs from JSON for clean scoring
    with open("data/syntax_sft_examples.json") as f:
        all_data = json.load(f)

    # Rebuild val indices using same seed as prepare_syntax_sft.py
    from datasets import Dataset as HFDataset
    full_ds = HFDataset.from_list(all_data)
    split = full_ds.train_test_split(test_size=0.10, seed=42)
    val_raw = split["test"]

    print(f"Evaluating {len(val_raw)} validation examples...")

    # Build prompts using chat template
    prompts = []
    for ex in val_raw:
        messages = [{"role": "user", "content": ex["input"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # Run inference
    print("Running inference...")
    predictions = run_inference(model, tokenizer, prompts)

    # Score
    print("Scoring...")
    results = []
    for ex, pred in tqdm(zip(val_raw, predictions), total=len(val_raw)):
        score = score_example(pred, ex["output"], embed_model)
        score["input"]     = ex["input"]
        score["gold"]      = ex["output"]
        score["predicted"] = pred
        results.append(score)

    # ---------- REPORT ----------
    total = len(results)
    parse_ok    = sum(r["parse_ok"]   for r in results)
    tool_match  = sum(r["tool_match"] for r in results)
    overall_ok  = sum(r["overall"]    for r in results)

    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"Total examples  : {total}")
    print(f"Parse success   : {parse_ok}  ({100*parse_ok/total:.1f}%)")
    print(f"Correct tool    : {tool_match} ({100*tool_match/total:.1f}%)")
    print(f"Overall correct : {overall_ok} ({100*overall_ok/total:.1f}%)")

    # Per-tool breakdown
    tools = ["zoom","email","text","calendar","contacts","clock"]
    print("\nPer-tool accuracy:")
    for tool in tools:
        tool_results = [r for r in results if r["gold"].startswith(f"<tool>{tool}")]
        if not tool_results:
            continue
        n = len(tool_results)
        correct = sum(r["overall"] for r in tool_results)
        print(f"  {tool:<12}: {correct}/{n} ({100*correct/n:.1f}%)")

    # Per-field semantic score summary
    print("\nSemantic field scores (email subject, email body, text body):")
    for tool, fields in SEMANTIC_FIELDS.items():
        for idx, fname in fields.items():
            scores = [
                r["field_scores"][fname]["score"]
                for r in results
                if fname in r.get("field_scores", {})
                and r["field_scores"][fname].get("type") == "semantic"
            ]
            if scores:
                print(f"  {tool}.{fname:<12}: avg cosine = {np.mean(scores):.4f}  "
                      f"pass@0.80 = {100*sum(s>=SEM_THRESHOLD for s in scores)/len(scores):.1f}%")

    print("="*60)

    # Save full results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
