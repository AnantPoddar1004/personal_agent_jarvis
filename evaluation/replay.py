"""Display the saved epoch-three clock rollout. Run from any directory."""
import json
from pathlib import Path

trace = json.loads((Path(__file__).resolve().parents[1] / "results/clock-trace.json").read_text())
print(trace["task"])
for step in trace["best_trajectory"]:
    print(f'{step["step_num"]}. {step["tool_call"]}   reward {step["reward"]:+.3f}')
print(f'Total reward: {trace["best_reward"]:.3f}')
