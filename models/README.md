# Trained LoRA adapters

Both adapters load directly on **Qwen/Qwen2.5-3B-Instruct**. Each archive includes the adapter, PEFT configuration, and tokenizer. The published configurations identify the base model by its Hub ID.

| Archive | Training provenance |
| --- | --- |
| [sft.tar.gz](https://github.com/AnantPoddar1004/personal_agent_jarvis/releases/download/research-v1/sft.tar.gz) | Weighted supervised training, archived as outputs/trajectory_sft/adapter |
| [rule-grpo.tar.gz](https://github.com/AnantPoddar1004/personal_agent_jarvis/releases/download/research-v1/rule-grpo.tar.gz) | Final adapter from rule-based GRPO run 002 |

From the repository root, download and verify both archives.

```sh
mkdir -p models/assets
gh release download research-v1 --pattern '*.tar.gz' --dir models/assets
(cd models/assets && shasum -a 256 -c ../SHA256SUMS)
tar -xzf models/assets/sft.tar.gz -C models/assets
tar -xzf models/assets/rule-grpo.tar.gz -C models/assets
```

Load either model in the training environment by choosing its directory.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter = "models/assets/rule-grpo"  # or models/assets/sft
tokenizer = AutoTokenizer.from_pretrained(adapter)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, adapter).eval()
```

To run the SFT evaluator against the released weights from a fresh checkout:

```sh
mkdir -p outputs/trajectory_sft
cp -R models/assets/sft outputs/trajectory_sft/adapter
python evaluation/evaluate_trajectory_sft.py
```
