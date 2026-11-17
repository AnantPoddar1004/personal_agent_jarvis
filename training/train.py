"""
train.py
QLoRA fine-tuning of Qwen 2.5 3B Instruct using TRL SFTTrainer.
Saves adapter weights to outputs/adapter
Saves training plots  to outputs/plots/
"""
import os
import json
import torch
import matplotlib
matplotlib.use("Agg")  # headless — no display needed on RunPod
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
MODEL_PATH   = "Qwen/Qwen2.5-3B-Instruct"
DATA_DIR     = "data/processed"
OUTPUT_DIR   = "outputs/adapter"
PLOTS_DIR    = "outputs/plots"
LOG_DIR      = "logs"
CKPT_DIR     = "checkpoints"
METRICS_PATH = "outputs/training_metrics.json"

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODS  = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

MAX_SEQ_LEN  = 256
BATCH_SIZE   = 8
GRAD_ACCUM   = 2        # effective batch = 16
LR           = 2e-4
EPOCHS       = 3
WARMUP_RATIO = 0.05
SAVE_STEPS   = 100
EVAL_STEPS   = 100
LOGGING_STEPS= 25

# --------------------------------------------------------------------------
# METRICS COLLECTOR CALLBACK
# --------------------------------------------------------------------------
class MetricsCallback(TrainerCallback):
    """Collects train loss, eval loss, learning rate at every log step."""
    def __init__(self):
        self.train_loss   = []   # (step, loss)
        self.eval_loss    = []   # (step, loss)
        self.learning_rate= []   # (step, lr)
        self.grad_norm    = []   # (step, grad_norm)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_loss.append((step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_loss.append((step, logs["eval_loss"]))
        if "learning_rate" in logs:
            self.learning_rate.append((step, logs["learning_rate"]))
        if "grad_norm" in logs:
            self.grad_norm.append((step, logs["grad_norm"]))

# --------------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------------
def plot_all(metrics: MetricsCallback, plots_dir: str, total_steps: int):
    os.makedirs(plots_dir, exist_ok=True)

    # ── 1. Loss curves (train + eval on same axes) ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    if metrics.train_loss:
        steps, vals = zip(*metrics.train_loss)
        ax.plot(steps, vals, label="Train loss", color="#2196F3", linewidth=1.5)
    if metrics.eval_loss:
        steps, vals = zip(*metrics.eval_loss)
        ax.plot(steps, vals, label="Eval loss", color="#F44336",
                linewidth=2, linestyle="--", marker="o", markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, "loss_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 2. Learning rate schedule ────────────────────────────────────────────
    if metrics.learning_rate:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps, vals = zip(*metrics.learning_rate)
        ax.plot(steps, vals, color="#4CAF50", linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning Rate Schedule (Cosine with Warmup)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(plots_dir, "learning_rate.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 3. Gradient norm ────────────────────────────────────────────────────
    if metrics.grad_norm:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps, vals = zip(*metrics.grad_norm)
        ax.plot(steps, vals, color="#FF9800", linewidth=1, alpha=0.8)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Gradient norm")
        ax.set_title("Gradient Norm During Training")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(plots_dir, "grad_norm.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 4. Dashboard — all four panels in one image ──────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("QLoRA Fine-Tuning Dashboard — Qwen 2.5 3B", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Panel A: train loss
    ax_tl = fig.add_subplot(gs[0, 0])
    if metrics.train_loss:
        s, v = zip(*metrics.train_loss)
        ax_tl.plot(s, v, color="#2196F3", linewidth=1.5)
    ax_tl.set_title("Train Loss"); ax_tl.set_xlabel("Step"); ax_tl.set_ylabel("Loss")
    ax_tl.grid(True, alpha=0.3)

    # Panel B: eval loss
    ax_el = fig.add_subplot(gs[0, 1])
    if metrics.eval_loss:
        s, v = zip(*metrics.eval_loss)
        ax_el.plot(s, v, color="#F44336", linewidth=2, marker="o", markersize=4)
    ax_el.set_title("Eval Loss"); ax_el.set_xlabel("Step"); ax_el.set_ylabel("Loss")
    ax_el.grid(True, alpha=0.3)

    # Panel C: learning rate
    ax_lr = fig.add_subplot(gs[1, 0])
    if metrics.learning_rate:
        s, v = zip(*metrics.learning_rate)
        ax_lr.plot(s, v, color="#4CAF50", linewidth=1.5)
    ax_lr.set_title("Learning Rate"); ax_lr.set_xlabel("Step"); ax_lr.set_ylabel("LR")
    ax_lr.grid(True, alpha=0.3)

    # Panel D: grad norm
    ax_gn = fig.add_subplot(gs[1, 1])
    if metrics.grad_norm:
        s, v = zip(*metrics.grad_norm)
        ax_gn.plot(s, v, color="#FF9800", linewidth=1, alpha=0.8)
    ax_gn.set_title("Gradient Norm"); ax_gn.set_xlabel("Step"); ax_gn.set_ylabel("Norm")
    ax_gn.grid(True, alpha=0.3)

    path = os.path.join(plots_dir, "dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # -- Tokenizer --
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Quantization config (4-bit QLoRA) --
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # -- Model --
    print("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # -- LoRA --
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -- Data --
    print("Loading datasets...")
    train_ds = load_from_disk(os.path.join(DATA_DIR, "train"))
    val_ds   = load_from_disk(os.path.join(DATA_DIR, "val"))
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Only compute loss on the assistant turn, not the user prompt
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # -- Metrics callback --
    metrics_cb = MetricsCallback()

    # -- Training args --
    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        tf32=True,
        logging_dir=LOG_DIR,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=4,
        group_by_length=True,
    )

    # -- Trainer --
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        callbacks=[metrics_cb],
    )

    # -- Train --
    print("\nStarting training...")
    train_result = trainer.train()

    # -- Save adapter --
    print(f"\nSaving adapter to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Adapter saved ✓")

    # -- Save raw metrics JSON (useful for later analysis) --
    metrics_data = {
        "train_loss":    metrics_cb.train_loss,
        "eval_loss":     metrics_cb.eval_loss,
        "learning_rate": metrics_cb.learning_rate,
        "grad_norm":     metrics_cb.grad_norm,
        "train_runtime_seconds": train_result.metrics.get("train_runtime", None),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", None),
        "total_steps":   trainer.state.global_step,
        "epochs":        EPOCHS,
        "final_train_loss": train_result.metrics.get("train_loss", None),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics JSON saved to {METRICS_PATH} ✓")

    # -- Plot --
    print(f"\nGenerating plots in {PLOTS_DIR}...")
    plot_all(metrics_cb, PLOTS_DIR, trainer.state.global_step)

    # -- Summary --
    print("\n" + "="*55)
    print("TRAINING COMPLETE")
    print("="*55)
    runtime = train_result.metrics.get("train_runtime", 0)
    print(f"  Total steps     : {trainer.state.global_step}")
    print(f"  Runtime         : {runtime/60:.1f} min")
    print(f"  Final train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Adapter saved   : {OUTPUT_DIR}")
    print(f"  Plots saved     : {PLOTS_DIR}/")
    print(f"    - loss_curves.png")
    print(f"    - learning_rate.png")
    print(f"    - grad_norm.png")
    print(f"    - dashboard.png   ← all 4 panels combined")
    print("="*55)

if __name__ == "__main__":
    main()
