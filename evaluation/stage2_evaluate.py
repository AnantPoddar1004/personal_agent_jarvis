"""
stage2_evaluate.py
Evaluates the stage2 adapter on the validation set.

Three-tier scoring:
  1. format_ok      — output contains <reasoning>...</reasoning> then <tool>...</tool>
                      in correct order, no extra content outside tags
  2. reasoning_cos  — semantic similarity of reasoning content (only if format_ok)
  3. tool_score     — exact match + cosine for text fields (only if format_ok)

Examples that fail format_ok get tool_score=0 and reasoning_cos=0.
"""

import json, re
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, TaskType
from sentence_transformers import SentenceTransformer

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "outputs/stage2/adapter"
RAW_DATA    = "data/sft_stage2_data.json"
OUTPUT_FILE = "outputs/stage2/eval_results_stage2.json"

MAX_NEW_TOKENS = 250
EVAL_SAMPLES   = 500
COSINE_THRESH  = 0.80
COSINE_FIELDS  = {"email_subject", "email_body", "text_body"}

SYSTEM_PROMPT = (
    "You are a personal assistant with access to the following tools: "
    "zoom, email, text, calendar, contacts, clock, stop. "
    "Think through each step carefully before acting."
)

# ── format checking ────────────────────────────────────────────────────────────

def check_format(text):
    """
    Returns (format_ok, reasoning_content, tool_content).
    format_ok is True only when:
      - exactly one <reasoning>...</reasoning> block exists
      - exactly one <tool>...</tool> block exists
      - reasoning comes before tool
      - no content outside these two blocks (ignoring whitespace)
    """
    r_matches = list(re.finditer(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL))
    t_matches = list(re.finditer(r"<tool>(.*?)</tool>",           text, re.DOTALL))

    if len(r_matches) != 1 or len(t_matches) != 1:
        return False, "", ""

    r_match = r_matches[0]
    t_match = t_matches[0]

    # reasoning must come before tool
    if r_match.end() > t_match.start():
        return False, "", ""

    # nothing meaningful outside the two blocks
    before_reasoning = text[:r_match.start()].strip()
    between          = text[r_match.end():t_match.start()].strip()
    after_tool       = text[t_match.end():].strip()

    if before_reasoning or between or after_tool:
        return False, "", ""

    return True, r_match.group(1).strip(), t_match.group(1).strip()

# ── tool scoring ───────────────────────────────────────────────────────────────

def parse_tool(tool_str):
    parts  = tool_str.split("|")
    result = {
        "tool":     parts[0].strip() if len(parts) > 0 else "",
        "function": parts[1].strip() if len(parts) > 1 else "",
    }
    params = parts[2:] if len(parts) > 2 else []
    tool   = result["tool"]

    if tool == "zoom" and len(params) >= 3:
        result.update({"topic": params[0], "start_time": params[1], "duration": params[2]})
    elif tool == "email" and len(params) >= 3:
        result.update({"to": params[0], "email_subject": params[1],
                       "email_body": "|".join(params[2:])})
    elif tool == "text" and len(params) >= 2:
        result.update({"to": params[0], "text_body": "|".join(params[1:])})
    elif tool == "calendar" and len(params) >= 3:
        result.update({"datetime": params[0], "duration": params[1], "title": params[2]})
    elif tool == "contacts" and len(params) >= 1:
        result.update({"name": params[0]})
    return result

def score_tool(pred_str, gold_str, encoder):
    """
    Returns (exact_score, cosine_score) both in [0, 1].
    exact_score: 1.0 only if tool, function, and all non-cosine fields match exactly.
    cosine_score: mean cosine similarity over cosine fields (email/text body+subject).
    """
    pred = parse_tool(pred_str)
    gold = parse_tool(gold_str)

    if pred.get("tool") != gold.get("tool"):
        return 0.0, 0.0
    if pred.get("function") != gold.get("function"):
        return 0.0, 0.0

    exact_fields  = [k for k in gold if k not in ("tool", "function") and k not in COSINE_FIELDS]
    cosine_fields = [k for k in gold if k in COSINE_FIELDS]

    # exact match on non-text fields
    exact_score = 1.0
    for f in exact_fields:
        if pred.get(f, "").strip() != gold.get(f, "").strip():
            exact_score = 0.0
            break

    # cosine on text fields
    cosine_score = 1.0
    if cosine_fields:
        sims = []
        for f in cosine_fields:
            pv = pred.get(f, "")
            gv = gold.get(f, "")
            if not pv or not gv:
                sims.append(0.0)
                continue
            pe = encoder.encode([pv])[0]
            ge = encoder.encode([gv])[0]
            sim = float(np.dot(pe, ge) / (np.linalg.norm(pe) * np.linalg.norm(ge) + 1e-9))
            sims.append(sim)
        cosine_score = float(np.mean(sims))

    return exact_score, cosine_score

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    base  = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    print("Loading sentence encoder...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading val data...")
    with open(RAW_DATA) as f:
        raw_all = json.load(f)
    split_idx = int(len(raw_all) * 0.9)
    val_raw   = raw_all[split_idx: split_idx + EVAL_SAMPLES]
    print(f"  Evaluating {len(val_raw)} examples")

    results = []
    format_scores, reasoning_scores, tool_exact_scores, tool_cosine_scores = [], [], [], []

    for i, ex in enumerate(val_raw):
        if i % 50 == 0:
            print(f"  {i}/{len(val_raw)}...")

        # generate
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": ex["input"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        # ── tier 1: format check ───────────────────────────────────────────────
        fmt_ok, pred_reasoning, pred_tool = check_format(generated)
        _, gold_reasoning, gold_tool      = check_format(ex["output"])
        format_scores.append(int(fmt_ok))

        if not fmt_ok:
            reasoning_scores.append(0.0)
            tool_exact_scores.append(0.0)
            tool_cosine_scores.append(0.0)
            results.append({
                "input": ex["input"],
                "generated": generated,
                "format_ok": False,
                "reasoning_cosine": 0.0,
                "tool_exact": 0.0,
                "tool_cosine": 0.0,
            })
            continue

        # ── tier 2: reasoning cosine ───────────────────────────────────────────
        pe = encoder.encode([pred_reasoning])[0]
        ge = encoder.encode([gold_reasoning])[0]
        r_sim = float(np.dot(pe, ge) / (np.linalg.norm(pe) * np.linalg.norm(ge) + 1e-9))
        reasoning_scores.append(r_sim)

        # ── tier 3: tool scoring ───────────────────────────────────────────────
        exact, cosine = score_tool(pred_tool, gold_tool, encoder)
        tool_exact_scores.append(exact)
        tool_cosine_scores.append(cosine)

        results.append({
            "input":            ex["input"],
            "generated":        generated,
            "gold_reasoning":   gold_reasoning,
            "gold_tool":        gold_tool,
            "pred_reasoning":   pred_reasoning,
            "pred_tool":        pred_tool,
            "format_ok":        True,
            "reasoning_cosine": r_sim,
            "tool_exact":       exact,
            "tool_cosine":      cosine,
        })

    summary = {
        "n_eval":                       len(results),
        "format_ok_pct":                float(np.mean(format_scores)),
        "reasoning_cosine_mean":        float(np.mean(reasoning_scores)),
        "reasoning_cosine_above_thresh":float(np.mean([s >= COSINE_THRESH for s in reasoning_scores])),
        "tool_exact_mean":              float(np.mean(tool_exact_scores)),
        "tool_cosine_mean":             float(np.mean(tool_cosine_scores)),
        "tool_exact_pct_perfect":       float(np.mean([s >= 1.0 for s in tool_exact_scores])),
    }

    print("\n── Evaluation Results ───────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
