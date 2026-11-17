"""
merge_adapter.py
Merges the QLoRA adapter into full model weights for downstream use.
Output is a standard HuggingFace model at outputs/merged_model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH    = "outputs/adapter"
MERGED_PATH     = "outputs/merged_model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

print("Loading base model in bf16 (no quantization for clean merge)...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)

print("Merging adapter into base weights...")
merged = model.merge_and_unload()

print(f"Saving merged model to {MERGED_PATH}...")
merged.save_pretrained(MERGED_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_PATH)

print("Merge complete ✓")
print(f"Merged model saved to: {MERGED_PATH}")
print("This model is now a standard Qwen checkpoint ready for downstream use.")
