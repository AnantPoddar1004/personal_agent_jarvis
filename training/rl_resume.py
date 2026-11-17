"""
rl_resume.py — Resume RL training from a checkpoint epoch.

Imports everything from rl_train.py unchanged, then overrides only:
  - START_EPOCH  : which epoch to start from (default: 3)
  - EPOCHS       : total epochs (must match original, default: 3)
  - model_path   : path to the checkpoint to resume from

Usage (from your qwen-finetune directory):
    !python rl_resume.py \
        --model_path rl_run_002/checkpoints/epoch_2 \
        --tasks tasks.json \
        --start_epoch 3

rl_train.py is NOT modified.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import torch

# ── Import everything from the original training script ──
# Make sure rl_train.py is in the same directory (or on PYTHONPATH).
import rl_train as rl

# ──────────────────────────────────────────────
# RESUME BLOCK — only thing that changes
# ──────────────────────────────────────────────

def train_resume(args):
    """
    Identical to rl_train.train() but starts the epoch loop at args.start_epoch
    and loads the existing metrics file (if any) so history is preserved.
    """
    # Load tasks
    with open(args.tasks) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks")
    print(f"Resuming from epoch {args.start_epoch} / {rl.EPOCHS}")
    print(f"Checkpoint: {args.model_path}")

    # Load models from the checkpoint (epoch_2 adapter in your case)
    policy, reference, tokenizer, device = rl.load_policy_and_reference(args.model_path)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=rl.LEARNING_RATE,
    )

    # Create output dirs (already exist, but safe to call again)
    Path(rl.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(rl.TRAJECTORY_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{rl.RUN_DIR}/metrics").mkdir(parents=True, exist_ok=True)
    Path(f"{rl.RUN_DIR}/final").mkdir(parents=True, exist_ok=True)

    # Load existing metrics so we append rather than overwrite
    if Path(rl.METRICS_FILE).exists():
        with open(rl.METRICS_FILE) as f:
            all_metrics = json.load(f)
        print(f"Loaded existing metrics ({len(all_metrics)} epochs recorded)")
    else:
        all_metrics = []

    global_step = 0

    # ── Epoch loop — starts at start_epoch instead of 1 ──
    for epoch in range(args.start_epoch, rl.EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{rl.EPOCHS}")
        print(f"{'='*60}")

        epoch_rewards = []
        epoch_losses  = []
        epoch_trajs   = []

        for task_idx, task in enumerate(tasks):
            torch.cuda.empty_cache()
            print(f"\n[Task {task_idx+1}/{len(tasks)}] {task}")

            # 1. ROLLOUT
            group_rollouts = []
            group_rewards  = []

            for g in range(rl.G):
                steps, total_reward, completed = rl.run_rollout(
                    policy, tokenizer, device, task
                )
                group_rollouts.append(steps)
                group_rewards.append(total_reward)
                print(f"  Rollout {g+1}/{rl.G} | reward={total_reward:.2f} | "
                      f"steps={len(steps)} | completed={completed}")

            # 2. ADVANTAGE
            advantages = rl.compute_advantages(group_rewards)

            flat_steps      = []
            flat_advantages = []
            for rollout_steps, adv in zip(group_rollouts, advantages):
                for step in rollout_steps:
                    flat_steps.append(step)
                    flat_advantages.append(adv)

            # 3. POLICY UPDATE
            optimizer.zero_grad()
            loss = rl.compute_grpo_loss(
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

            loss_val    = loss.item()
            mean_reward = sum(group_rewards) / len(group_rewards)
            epoch_rewards.append(mean_reward)
            epoch_losses.append(loss_val)

            print(f"  Loss={loss_val:.4f} | MeanReward={mean_reward:.2f} | "
                  f"Advantages={[f'{a:.2f}' for a in advantages]}")

            best_idx = group_rewards.index(max(group_rewards))
            epoch_trajs.append({
                "task":        task,
                "best_reward": max(group_rewards),
                "mean_reward": mean_reward,
                "all_rewards": group_rewards,
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

        # ── End of epoch ──
        avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        avg_loss   = sum(epoch_losses)  / max(len(epoch_losses), 1)
        print(f"\nEpoch {epoch} Summary | AvgReward={avg_reward:.3f} | AvgLoss={avg_loss:.4f}")

        # Save checkpoint
        if epoch % rl.SAVE_EVERY_N_EPOCHS == 0:
            ckpt_path = f"{rl.OUTPUT_DIR}/epoch_{epoch}"
            policy.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

        # Save trajectories
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_path = f"{rl.TRAJECTORY_DIR}/epoch{epoch}_{ts}.json"
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

        # Append to existing metrics (not overwrite)
        all_metrics.append({
            "epoch":        epoch,
            "timestamp":    datetime.now().isoformat(),
            "avg_reward":   avg_reward,
            "avg_loss":     avg_loss,
            "task_rewards": epoch_rewards,
            "task_losses":  epoch_losses,
        })
        with open(rl.METRICS_FILE, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"  Metrics saved → {rl.METRICS_FILE}")

    # Final save
    final_path = f"{rl.RUN_DIR}/final"
    policy.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal model saved → {final_path}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="rl_run_002/checkpoints/epoch_2",
        help="Path to checkpoint to resume from (e.g. rl_run_002/checkpoints/epoch_2)",
    )
    parser.add_argument(
        "--tasks", type=str, default="data/tasks.json",
        help="Path to JSON task list",
    )
    parser.add_argument(
        "--start_epoch", type=int, default=3,
        help="Epoch to resume from (inclusive). Default: 3",
    )
    args = parser.parse_args()
    train_resume(args)
