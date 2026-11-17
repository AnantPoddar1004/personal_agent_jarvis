"""
train_trajectory_sft.py
QLoRA fine-tuning of Qwen2.5-3B-Instruct on trajectory-augmented SFT data.

Loss = 1.0 * loss_tool + 0.3 * loss_reasoning
Both are cross-entropy. Token boundaries come from stored indices in the
dataset — no tag searching at training time.

Trains from the ORIGINAL base model.
"""

import json, math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = "Qwen/Qwen2.5-3B-Instruct"
DATA_DIR     = "data/trajectory_sft_processed"
OUTPUT_DIR   = "outputs/trajectory_sft"
CKPT_DIR     = "checkpoints/trajectory_sft"
METRICS_FILE = f"{OUTPUT_DIR}/trajectory_sft_training_metrics.json"

# ── hyperparams ────────────────────────────────────────────────────────────────
EPOCHS       = 2
BATCH_SIZE   = 4
GRAD_ACCUM   = 4        # effective batch = 16
LR           = 2e-4
WARMUP_FRAC  = 0.05
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
SAVE_STEPS   = 200
LOG_STEPS    = 10

# loss weights
W_TOOL      = 1.0
W_REASONING = 0.3

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)


# ── loss function ──────────────────────────────────────────────────────────────

def compute_weighted_loss(logits, labels, reasoning_starts, reasoning_ends,
                          tool_starts, tool_ends):
    """
    Weighted cross-entropy using pre-stored tag boundary indices.

    reasoning_starts/ends and tool_starts/ends are tensors of shape (B,)
    containing absolute token indices into the sequence. Value -1 means
    the span was not found (truncated) — those tokens get W_TOOL weight
    as a fallback so they still contribute to loss.

    Returns: total_loss, loss_tool (scalar), loss_reasoning (scalar)
    """
    B, T, V = logits.shape

    # shift by 1 for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    T1 = T - 1

    # per-token cross-entropy, ignore_index=-100 handles masked input tokens
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, T1)

    weight_mask    = torch.zeros(B, T1, device=logits.device)
    tool_mask      = torch.zeros(B, T1, device=logits.device)
    reasoning_mask = torch.zeros(B, T1, device=logits.device)

    for b in range(B):
        lbl  = shift_labels[b]
        r_s  = reasoning_starts[b].item()
        r_e  = reasoning_ends[b].item()
        t_s  = tool_starts[b].item()
        t_e  = tool_ends[b].item()

        has_reasoning = (r_s != -1 and r_e != -1 and r_s < r_e)
        has_tool      = (t_s != -1 and t_e != -1 and t_s < t_e)

        for pos in range(T1):
            if lbl[pos] == -100:
                continue  # masked input token

            # shift: stored indices are into full_ids, shift_labels starts at 1
            in_reasoning = has_reasoning and (r_s <= pos + 1 < r_e)
            in_tool      = has_tool      and (t_s <= pos + 1 < t_e)

            if in_reasoning:
                weight_mask[b, pos]    = W_REASONING
                reasoning_mask[b, pos] = 1.0
            elif in_tool:
                weight_mask[b, pos] = W_TOOL
                tool_mask[b, pos]   = 1.0
            else:
                # structural tokens (<reasoning>, </reasoning>, <tool>, </tool>, newlines)
                # treat at tool weight so they still contribute
                weight_mask[b, pos] = W_TOOL

    weighted_loss = (per_token_loss * weight_mask).sum() / weight_mask.sum().clamp(min=1)

    loss_tool      = (per_token_loss * tool_mask).sum()      / tool_mask.sum().clamp(min=1)
    loss_reasoning = (per_token_loss * reasoning_mask).sum() / reasoning_mask.sum().clamp(min=1)

    return weighted_loss, loss_tool.item(), loss_reasoning.item()


# ── collator ───────────────────────────────────────────────────────────────────

def collate_fn(batch, pad_id):
    max_len    = max(len(b["input_ids"]) for b in batch)
    B          = len(batch)
    input_ids  = torch.full((B, max_len), pad_id,  dtype=torch.long)
    attention  = torch.zeros(B, max_len,            dtype=torch.long)
    labels_t   = torch.full((B, max_len), -100,     dtype=torch.long)

    r_starts = torch.zeros(B, dtype=torch.long)
    r_ends   = torch.zeros(B, dtype=torch.long)
    t_starts = torch.zeros(B, dtype=torch.long)
    t_ends   = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        input_ids[i, :l] = torch.tensor(b["input_ids"])
        attention[i,  :l] = torch.tensor(b["attention_mask"])
        labels_t[i,   :l] = torch.tensor(b["labels"])
        r_starts[i] = b["reasoning_start"]
        r_ends[i]   = b["reasoning_end"]
        t_starts[i] = b["tool_start"]
        t_ends[i]   = b["tool_end"]

    return {
        "input_ids":        input_ids,
        "attention_mask":   attention,
        "labels":           labels_t,
        "reasoning_starts": r_starts,
        "reasoning_ends":   r_ends,
        "tool_starts":      t_starts,
        "tool_ends":        t_ends,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_ds = load_from_disk(f"{DATA_DIR}/train")
    val_ds   = load_from_disk(f"{DATA_DIR}/val")
    pad_id   = tokenizer.pad_token_id

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )

    total_steps  = math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    metrics = {
        "train_loss": [], "train_loss_tool": [], "train_loss_reasoning": [],
        "val_loss":   [], "val_loss_tool":   [], "val_loss_reasoning":   [],
        "steps": [],
    }

    global_step = 0
    optimizer.zero_grad()

    print(f"\nTraining {EPOCHS} epochs — {total_steps} optimizer steps")
    print(f"W_TOOL={W_TOOL}  W_REASONING={W_REASONING}\n")

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = ep_tool = ep_reason = ep_n = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)
            r_starts  = batch["reasoning_starts"].to(device)
            r_ends    = batch["reasoning_ends"].to(device)
            t_starts  = batch["tool_starts"].to(device)
            t_ends    = batch["tool_ends"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits  = outputs.logits

            loss, l_tool, l_reason = compute_weighted_loss(
                logits, labels, r_starts, r_ends, t_starts, t_ends
            )
            (loss / GRAD_ACCUM).backward()

            ep_loss   += loss.item()
            ep_tool   += l_tool
            ep_reason += l_reason
            ep_n      += 1

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_STEPS == 0:
                    avg        = ep_loss   / ep_n
                    avg_tool   = ep_tool   / ep_n
                    avg_reason = ep_reason / ep_n
                    lr_now     = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                          f"loss={avg:.4f}  tool={avg_tool:.4f}  "
                          f"reasoning={avg_reason:.4f}  lr={lr_now:.2e}")
                    metrics["steps"].append(global_step)
                    metrics["train_loss"].append(avg)
                    metrics["train_loss_tool"].append(avg_tool)
                    metrics["train_loss_reasoning"].append(avg_reason)
                    ep_loss = ep_tool = ep_reason = ep_n = 0.0

                if global_step % SAVE_STEPS == 0:
                    ckpt = f"{CKPT_DIR}/step_{global_step}"
                    model.save_pretrained(ckpt)
                    print(f"  Checkpoint → {ckpt}")

        # ── validation ─────────────────────────────────────────────────────────
        model.eval()
        v_loss = v_tool = v_reason = v_n = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["labels"].to(device)
                r_starts  = batch["reasoning_starts"].to(device)
                r_ends    = batch["reasoning_ends"].to(device)
                t_starts  = batch["tool_starts"].to(device)
                t_ends    = batch["tool_ends"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                loss, lt, lr = compute_weighted_loss(
                    outputs.logits, labels, r_starts, r_ends, t_starts, t_ends
                )
                v_loss   += loss.item()
                v_tool   += lt
                v_reason += lr
                v_n      += 1

        vl = v_loss / v_n
        vt = v_tool / v_n
        vr = v_reason / v_n
        print(f"\n── Epoch {epoch+1} val: loss={vl:.4f}  tool={vt:.4f}  reasoning={vr:.4f} ──\n")
        metrics["val_loss"].append(vl)
        metrics["val_loss_tool"].append(vt)
        metrics["val_loss_reasoning"].append(vr)

    # save
    adapter_dir = f"{OUTPUT_DIR}/adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nAdapter saved → {adapter_dir}")

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_FILE}")


if __name__ == "__main__":
    main()
