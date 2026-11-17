"""
GRPO RL Training Loop
Trains Qwen2.5-3B-Instruct using Group Relative Policy Optimization.
Rewards come from tool input validation in simulated_tools_and_rewards.py.

Usage:
    python training/train_rule_grpo.py --model_path outputs/trajectory_sft/adapter --tasks data/assistant_tasks.json

Requirements:
    pip install transformers datasets peft trl accelerate torch
"""

import json
import re
import argparse
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

from simulated_tools_and_rewards import (
    zoom_schedule,
    email_send,
    text_send,
    calendar_add_event,
    contacts_get_info,
    clock_get_date_time,
    stop,
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

MODEL_ID            = "Qwen/Qwen2.5-3B-Instruct"
RUN_DIR             = "outputs/rule_grpo"
OUTPUT_DIR          = f"{RUN_DIR}/checkpoints"
TRAJECTORY_DIR      = f"{RUN_DIR}/trajectories"
METRICS_FILE        = f"{RUN_DIR}/metrics/training_metrics_rl.json"

# GRPO hyperparameters
G                   = 4         # Number of rollouts per task (group size)
EPOCHS              = 3         # Number of full passes over task dataset
LEARNING_RATE       = 1e-5      # Lower than SFT — RL updates are noisier
KL_COEFF            = 0.05      # KL penalty weight vs reference model
MAX_STEPS           = 8         # Max tool calls per rollout before penalty
MAX_NEW_TOKENS      = 128
SAVE_EVERY_N_EPOCHS = 1

# Reward constants (must match simulated_tools_and_rewards.py)
REWARD_NO_STOP      = -2.0      # Never emitted stop token
REWARD_MAX_STEPS    = -0.5      # Hit step limit without stop
REPEAT_PENALTY      = -1.5      # Penalty for calling the same tool twice in a row
STEP_DECAY          = 0.95      # Reward decay per step — rewards earlier completion

SYSTEM_PROMPT = """You are a personal assistant with access to the following tools:
- zoom|schedule : Schedule a Zoom meeting. Params: topic (string) | start_time (datetime) | duration (int minutes)
- email|send : Send an email. Params: to (string) | subject (string) | body (string)
- text|send : Send a text message. Params: to (string) | body (string)
- calendar|add_event : Add a calendar event. Params: datetime (datetime) | duration (int minutes) | title (string)
- contacts|get_info : Look up a contact. Params: name (string)
- clock|get_date_time : Get the current date and time. No params.
- stop : Call this when the task is fully completed.

At each step, reason about what to do next, then output exactly one tool call:
<tool>[tool_name]|[function_name]|[param1]|[param2]|...</tool>
When done: <tool>stop</tool>"""


# ──────────────────────────────────────────────
# DEVICE SETUP
# ──────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────

def load_policy_and_reference(model_path: str):
    """
    Loads the SFT model as the policy (trainable).
    Makes a frozen copy as the reference model for KL penalty.
    """
    if not (Path(model_path) / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing starting adapter: {model_path}")
    device, dtype = get_device()
    device_map = "auto" if device == "cuda" else {"": device}

    print(f"Loading base model ({device}, {dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for batch generation

    from transformers import BitsAndBytesConfig
    policy_bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        quantization_config=policy_bnb,  # ← change from torch_dtype=dtype
        trust_remote_code=True,
    )

    # Load SFT LoRA adapter if present
    adapter_path = Path(model_path) / "adapter_config.json"
    if adapter_path.exists():
        print("  Loading SFT LoRA adapter onto policy model...")
        policy = PeftModel.from_pretrained(base, model_path, is_trainable=True)
    else:
        print("  No adapter found — applying fresh LoRA to base model...")
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
        )
        policy = get_peft_model(base, lora_cfg)

    policy.train()
    policy.gradient_checkpointing_enable()
    policy.print_trainable_parameters()

    # Reference model — same weights, fully frozen, no LoRA updates
    print("  Creating frozen reference model...")
    # ref_base = AutoModelForCausalLM.from_pretrained(
    #     MODEL_ID,
    #     device_map=device_map,
    #     torch_dtype=dtype,     # ← and here
    #     trust_remote_code=True,
    # )
    from transformers import BitsAndBytesConfig
    ref_bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    ref_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        quantization_config=ref_bnb,
        trust_remote_code=True,
    )
    if adapter_path.exists():
        reference = PeftModel.from_pretrained(ref_base, model_path)
    else:
        reference = ref_base

    for param in reference.parameters():
        param.requires_grad = False
    reference.eval()

    return policy, reference, tokenizer, device


# ──────────────────────────────────────────────
# TOOL DISPATCHER
# ──────────────────────────────────────────────

def _execute_tool(tool_call: str):
    """
    Internal helper — parses and routes tool call to simulated_tools_and_rewards.py.
    Returns (result, reward, is_stop).
    """
    tool_call = tool_call.strip()

    if tool_call == "stop":
        result, reward = stop()
        return result, reward, True

    parts      = tool_call.split("|")
    tool_name  = parts[0].strip() if len(parts) > 0 else ""
    func_name  = parts[1].strip() if len(parts) > 1 else ""
    params     = [p.strip() for p in parts[2:]]

    try:
        if tool_name == "clock" and func_name == "get_date_time":
            result, reward = clock_get_date_time()

        elif tool_name == "contacts" and func_name == "get_info":
            result, reward = contacts_get_info(params[0] if params else "")

        elif tool_name == "zoom" and func_name == "schedule":
            result, reward = zoom_schedule(
                params[0] if len(params) > 0 else "",
                params[1] if len(params) > 1 else "",
                params[2] if len(params) > 2 else "",
            )

        elif tool_name == "email" and func_name == "send":
            result, reward = email_send(
                params[0] if len(params) > 0 else "",
                params[1] if len(params) > 1 else "",
                params[2] if len(params) > 2 else "",
            )

        elif tool_name == "text" and func_name == "send":
            result, reward = text_send(
                params[0] if len(params) > 0 else "",
                params[1] if len(params) > 1 else "",
            )

        elif tool_name == "calendar" and func_name == "add_event":
            result, reward = calendar_add_event(
                params[0] if len(params) > 0 else "",
                params[1] if len(params) > 1 else "",
                params[2] if len(params) > 2 else "",
            )

        else:
            result = f"[Unknown tool: {tool_call}]"
            reward = -1.0

    except Exception as e:
        result = f"[Dispatch error: {e}]"
        reward = -1.0

    return result, reward, False


def dispatch_tool(tool_call: str, prev_tool_call: str = None):
    """
    Parses tool call string and routes to simulated_tools_and_rewards.py.
    Applies REPEAT_PENALTY if this call is identical to the previous one.
    Returns (result, reward, is_stop).
    """
    # Fix 1: penalize immediate repeat tool calls
    if prev_tool_call is not None and tool_call.strip() == prev_tool_call.strip():
        result, _, is_stop = _execute_tool(tool_call)
        return result, REPEAT_PENALTY, is_stop
    return _execute_tool(tool_call)


# ──────────────────────────────────────────────
# TOOL CALL PARSER
# ──────────────────────────────────────────────

def extract_tool_call(text: str):
    match = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    return match.group(1).strip() if match else None


# ──────────────────────────────────────────────
# SINGLE ROLLOUT
# Runs one full task trajectory, returns
# list of (prompt, response, reward) per step
# ──────────────────────────────────────────────

def run_rollout(model, tokenizer, device: str, task: str):
    """
    Generates one complete trajectory for a task.
    Returns:
        steps   : list of dicts with prompt/response/tool/reward per step
        total_reward : sum of all step rewards
        completed    : whether stop was reached
    """
    context = f"<input>{task}</input>"
    steps = []
    total_reward = 0.0
    completed = False
    prev_tool_call = None  # Fix 1: track previous tool call for repeat detection

    for step_num in range(MAX_STEPS):
        # Build prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": context},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,       # sample during RL rollouts for diversity
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Parse and execute tool
        tool_call = extract_tool_call(response_text)
        if tool_call is None:
            # No tool call found — penalize and end
            steps.append({
                "prompt": prompt_text,
                "response": response_text,
                "response_ids": new_ids,
                "tool_call": None,
                "tool_result": None,
                "reward": -1.0,
            })
            total_reward += -1.0
            break

        # Fix 1: pass prev_tool_call for repeat detection
        tool_result, reward, is_stop = dispatch_tool(tool_call, prev_tool_call)
        prev_tool_call = tool_call

        # Fix 3: apply step decay — reward earlier steps more
        decayed_reward = reward * (STEP_DECAY ** step_num)
        total_reward += decayed_reward

        steps.append({
            "prompt": prompt_text,
            "response": response_text,
            "response_ids": new_ids,
            "tool_call": tool_call,
            "tool_result": str(tool_result) if tool_result else None,
            "reward": decayed_reward,
        })

        if is_stop:
            completed = True
            break

        # Append step to context
        reasoning = re.sub(r"<tool>.*?</tool>", "", response_text, flags=re.DOTALL).strip()
        new_step = f"<step>{reasoning} {tool_result}</step>"
        context = context + new_step

    # Fix 2: flat penalty + proportional penalty for never stopping
    if not completed:
        total_reward += REWARD_NO_STOP
        total_reward += -0.1 * len(steps)

    return steps, total_reward, completed


# ──────────────────────────────────────────────
# GRPO LOSS
# Computes policy gradient loss with KL penalty
# ──────────────────────────────────────────────

def compute_grpo_loss(policy, reference, tokenizer, device, group_steps, advantages):
    """
    group_steps : list of step dicts from G rollouts (flattened)
    advantages  : list of per-step advantage values (same length)

    GRPO loss = -mean(advantage * log_prob) + KL_COEFF * KL(policy || reference)
    """
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    count = 0

    for step, advantage in zip(group_steps, advantages):
        if step["response_ids"] is None or len(step["response_ids"]) == 0:
            continue

        # Tokenize prompt
        prompt_ids = tokenizer(
            step["prompt"], return_tensors="pt"
        ).input_ids.to(device)

        response_ids = step["response_ids"].unsqueeze(0).to(device)

        # Full sequence = prompt + response
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)

        # Policy log probs over response tokens
        policy_out  = policy(full_ids)
        policy_logits = policy_out.logits[:, prompt_ids.shape[1]-1:-1, :]
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        token_logprobs = policy_logprobs[0, torch.arange(response_ids.shape[1]), response_ids[0]]
        policy_seq_logprob = token_logprobs.mean()

        # Reference log probs (no grad)
        with torch.no_grad():
            ref_out = reference(full_ids)
            ref_logits = ref_out.logits[:, prompt_ids.shape[1]-1:-1, :]
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_token_logprobs = ref_logprobs[0, torch.arange(response_ids.shape[1]), response_ids[0]]
            ref_seq_logprob = ref_token_logprobs.mean()

        # KL divergence (approximate): KL ≈ policy_logprob - ref_logprob
        kl = policy_seq_logprob - ref_seq_logprob

        # GRPO step loss
        adv_tensor = torch.tensor(advantage, dtype=torch.float32).to(device)
        step_loss = -(adv_tensor * policy_seq_logprob) + KL_COEFF * kl
        total_loss = total_loss + step_loss
        count += 1

    return total_loss / max(count, 1)


# ──────────────────────────────────────────────
# GRPO ADVANTAGE COMPUTATION
# Normalize rewards within a group
# ──────────────────────────────────────────────

def compute_advantages(group_rewards: list) -> list:
    """
    Normalizes rewards within a group of G rollouts.
    advantage_i = (reward_i - mean) / (std + eps)
    """
    import statistics
    if len(group_rewards) < 2:
        return [0.0] * len(group_rewards)
    mean = sum(group_rewards) / len(group_rewards)
    std  = statistics.stdev(group_rewards) if len(group_rewards) > 1 else 1.0
    eps  = 1e-8
    return [(r - mean) / (std + eps) for r in group_rewards]


# ──────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────

def train(args):
    # Load tasks
    with open(args.tasks) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks")

    # Load models
    policy, reference, tokenizer, device = load_policy_and_reference(args.model_path)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=LEARNING_RATE,
    )

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TRAJECTORY_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{RUN_DIR}/metrics").mkdir(parents=True, exist_ok=True)
    Path(f"{RUN_DIR}/final").mkdir(parents=True, exist_ok=True)

    global_step = 0
    all_metrics = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print(f"{'='*60}")

        epoch_rewards   = []
        epoch_losses    = []
        epoch_trajs     = []

        for task_idx, task in enumerate(tasks):
            torch.cuda.empty_cache()
            print(f"\n[Task {task_idx+1}/{len(tasks)}] {task}")

            # ── 1. ROLLOUT — generate G trajectories ──
            group_rollouts = []
            group_rewards  = []

            for g in range(G):
                steps, total_reward, completed = run_rollout(
                    policy, tokenizer, device, task
                )
                group_rollouts.append(steps)
                group_rewards.append(total_reward)
                print(f"  Rollout {g+1}/{G} | reward={total_reward:.2f} | "
                      f"steps={len(steps)} | completed={completed}")

            # ── 2. ADVANTAGE — normalize within group ──
            advantages = compute_advantages(group_rewards)

            # Assign per-step advantages (all steps in a rollout share
            # the rollout-level advantage — simple and effective)
            flat_steps      = []
            flat_advantages = []
            for rollout_steps, adv in zip(group_rollouts, advantages):
                for step in rollout_steps:
                    flat_steps.append(step)
                    flat_advantages.append(adv)

            # ── 3. POLICY UPDATE — GRPO loss ──
            optimizer.zero_grad()
            loss = compute_grpo_loss(
                policy, reference, tokenizer, device,
                flat_steps, flat_advantages
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, policy.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            global_step += 1

            loss_val = loss.item()
            mean_reward = sum(group_rewards) / len(group_rewards)
            epoch_rewards.append(mean_reward)
            epoch_losses.append(loss_val)

            print(f"  Loss={loss_val:.4f} | MeanReward={mean_reward:.2f} | "
                  f"Advantages={[f'{a:.2f}' for a in advantages]}")

            # Record best rollout trajectory for saving
            best_idx = group_rewards.index(max(group_rewards))
            epoch_trajs.append({
                "task":         task,
                "best_reward":  max(group_rewards),
                "mean_reward":  mean_reward,
                "all_rewards":  group_rewards,
                "best_trajectory": [
                    {
                        "step_num":    i+1,
                        "tool_call":   s["tool_call"],
                        "tool_result": s["tool_result"],
                        "reward":      s["reward"],
                        "response":    s["response"],
                    }
                    for i, s in enumerate(group_rollouts[best_idx])
                ],
            })

        # ── End of epoch summary ──
        avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        avg_loss   = sum(epoch_losses)  / max(len(epoch_losses), 1)
        print(f"\nEpoch {epoch} Summary | AvgReward={avg_reward:.3f} | AvgLoss={avg_loss:.4f}")

        # ── Save checkpoint ──
        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            ckpt_path = f"{OUTPUT_DIR}/epoch_{epoch}"
            policy.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

        # ── Save trajectories (timestamped per epoch) ──
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_path = f"{TRAJECTORY_DIR}/epoch{epoch}_{ts}.json"
        run_record = {
            "epoch":        epoch,
            "timestamp":    datetime.now().isoformat(),
            "model_path":   args.model_path,
            "avg_reward":   avg_reward,
            "avg_loss":     avg_loss,
            "trajectories": epoch_trajs,
        }
        with open(traj_path, "w") as f:
            json.dump(run_record, f, indent=2)
        print(f"  Trajectories saved → {traj_path}")

        # ── Save performance metrics ──
        all_metrics.append({
            "epoch":      epoch,
            "timestamp":  datetime.now().isoformat(),
            "avg_reward": avg_reward,
            "avg_loss":   avg_loss,
            "task_rewards": epoch_rewards,
            "task_losses":  epoch_losses,
        })
        with open(METRICS_FILE, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"  Metrics saved → {METRICS_FILE}")

    # ── Final save ──
    final_path = f"{RUN_DIR}/final"
    policy.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal model saved → {final_path}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/trajectory_sft/adapter",
                        help="Path to SFT fine-tuned model / LoRA adapter")
    parser.add_argument("--tasks",      type=str, default="data/assistant_tasks.json",
                        help="Path to JSON file (list of task strings)")
    args = parser.parse_args()
    train(args)
