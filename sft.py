"""
SFT Fine-tuning script for Qwen2.5-3B-Instruct
Personal Assistant Tool-Calling Training

Requirements:
    pip install transformers datasets peft trl accelerate bitsandbytes torch

Data format expected (JSON file):
    [
        {"input": "I have to send a text saying Happy Birthday to 9830487592.", "output": "<tool>text|send|9830487592|Happy Birthday</tool>"},
        ...
    ]
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ──────────────────────────────────────────────
# CONFIG — edit these values before running
# ──────────────────────────────────────────────

DATA_PATH       = "data.json"           # Path to your JSON data file
OUTPUT_DIR      = "./sft_output"        # Where to save checkpoints & final model
MODEL_ID        = "Qwen/Qwen2.5-3B-Instruct"

# Training hyperparameters
EPOCHS          = 3
BATCH_SIZE      = 4                     # Per-device batch size
GRAD_ACCUM      = 4                     # Effective batch = BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE   = 2e-4
MAX_SEQ_LEN     = 256                   # Tool calls are short; increase if needed
WARMUP_RATIO    = 0.05
SAVE_STEPS      = 100
LOGGING_STEPS   = 10

# LoRA hyperparameters
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

# Set to True if your GPU supports bfloat16 (A100, H100, RTX 3090+)
USE_BF16        = True
# Set to True to load model in 4-bit (saves VRAM, ~6GB for 3B)
USE_4BIT        = True

# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a personal assistant with access to the following tools:
- zoom|schedule : Schedule a Zoom meeting. Params: topic (string) | start_time (datetime) | duration (int minutes)
- email|send : Send an email. Params: to (string) | subject (string) | body (string)
- text|send : Send a text message. Params: to (string) | body (string)
- calendar|add_event : Add a calendar event. Params: datetime (datetime) | duration (int minutes) | title (string)
- contacts|get_info : Look up a contact. Params: name (string)
- time|get_time : Get the current date and time. No params.

When the user asks you to perform a task, respond ONLY with the correct tool call in this exact format:
<tool>[tool_name]|[function_name]|[param1]|[param2]|...</tool>"""


# ──────────────────────────────────────────────
# LOAD & FORMAT DATA
# ──────────────────────────────────────────────

def load_data(path: str) -> Dataset:
    """Load JSON data and format into chat-style prompt/completion pairs."""
    with open(path, "r") as f:
        raw = json.load(f)

    formatted = []
    for item in raw:
        # Build a chat-formatted string using Qwen's chat template markers.
        # The model is trained to predict ONLY the assistant turn (output).
        conversation = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{item['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{item['output']}<|im_end|>"
        )
        formatted.append({"text": conversation})

    return Dataset.from_list(formatted)


# ──────────────────────────────────────────────
# MODEL & TOKENIZER
# ──────────────────────────────────────────────

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # Required for causal LM training

    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )
    model.config.use_cache = False          # Required for gradient checkpointing
    model.enable_input_require_grads()      # Required for PEFT with gradient checkpointing

    return model, tokenizer


# ──────────────────────────────────────────────
# LORA CONFIG
# ──────────────────────────────────────────────

def apply_lora(model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Target the attention + MLP projection layers in Qwen2.5
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train():
    print("── Loading data ──")
    dataset = load_data(DATA_PATH)
    print(f"   {len(dataset)} examples loaded")

    # Optional: split off a small validation set
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset  = split["test"]

    print("── Loading model & tokenizer ──")
    model, tokenizer = load_model_and_tokenizer()

    print("── Applying LoRA ──")
    model = apply_lora(model)

    # DataCollator that masks everything EXCEPT the assistant turn,
    # so loss is computed only on the tool call output.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=USE_BF16,
        fp16=not USE_BF16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",               # Change to "wandb" if you use W&B
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args,
    )

    print("── Starting training ──")
    trainer.train()

    print("── Saving final model ──")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"   Model saved to {OUTPUT_DIR}/final")


# ──────────────────────────────────────────────
# INFERENCE HELPER — test your trained model
# ──────────────────────────────────────────────

def run_inference(prompt: str, model_path: str = f"{OUTPUT_DIR}/final"):
    """Quick inference test after training."""
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,        # Greedy decoding — deterministic tool calls
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nInput:  {prompt}")
    print(f"Output: {response}")
    return response


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    train()

    # Uncomment to test after training:
    # run_inference("Send a text to 9830487592 saying Happy Birthday")
    # run_inference("What is the current time?")
    # run_inference("Schedule a 30 minute zoom called Team Sync at 2026-03-05 14:00:00.000000")